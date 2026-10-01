"""Job cancellation / failure must stop every task and release worker resources.

Covers the whole cleanup chain: JobManager state finalization, the scheduler's
termination fan-out, terminal-state guards on late worker reports, and the
worker's per-job task handles + cooperative cancellation.
"""

import shutil
import tempfile
import time
import unittest

from backend.common import constants as C
from backend.common.config import ClusterConfig
from backend.common.logbus import LogBus
from backend.common.storage import Storage
from backend.master.fault_tolerance import FaultTolerance
from backend.master.job_manager import JobManager
from backend.master.metrics import Metrics
from backend.master.registry import WorkerRegistry
from backend.master.scheduler import Scheduler
from backend.master.shuffle import ShuffleCoordinator
from backend.worker.executor import Executor


def _submit(jm, **over):
    payload = {
        "name": "t", "mapper": "wordcount_mapper", "reducer": "count_reducer",
        "num_map_tasks": 3, "num_reduce_tasks": 2, "input_rows": 100, "params": {},
    }
    payload.update(over)
    return jm.submit(payload)


class _ClusterFixture(unittest.TestCase):
    """Master-side components wired the same way as in Master.__init__."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = Storage(self.tmp)
        self.config = ClusterConfig(max_attempts=2)
        self.logbus = LogBus(self.storage)
        self.jm = JobManager(self.storage, self.config, self.logbus)
        self.registry = WorkerRegistry(self.storage, self.config)
        self.shuffle = ShuffleCoordinator(self.storage, self.jm, self.registry, self.logbus)
        self.ft = FaultTolerance(self.storage, self.jm, self.config, self.logbus)
        self.metrics = Metrics(self.storage)
        self.scheduler = Scheduler(
            self.storage, self.jm, self.registry, self.shuffle,
            self.ft, self.metrics, self.config, self.logbus,
        )
        self.jm.on_terminal = self.scheduler.terminate_job_tasks
        self.job = _submit(self.jm)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _maps(self):
        return self.jm.tasks_for(self.job.job_id, C.TASK_MAP)


class TestCancelCleanup(_ClusterFixture):
    def test_cancel_finalizes_every_task(self):
        maps = self._maps()
        self.jm.update_task(self.job.job_id, maps[0].task_id,
                            status=C.TASK_RUNNING, worker_id="w1")
        self.jm.update_task(self.job.job_id, maps[1].task_id,
                            status=C.TASK_ASSIGNED, worker_id="w2")
        self.jm.update_task(self.job.job_id, maps[2].task_id,
                            status=C.TASK_RETRYING, retry_after_ms=10 ** 15)

        self.jm.cancel(self.job)

        statuses = {t.status for t in self.jm.tasks_for(self.job.job_id)}
        self.assertEqual(statuses, {C.TASK_CANCELLED})
        # Scheduler capacity accounting sees nothing left on the workers.
        self.assertEqual(self.scheduler._count_running_on("w1"), 0)
        self.assertEqual(self.scheduler._count_running_on("w2"), 0)

    def test_finalize_reports_inflight_pairs(self):
        maps = self._maps()
        self.jm.update_task(self.job.job_id, maps[0].task_id,
                            status=C.TASK_RUNNING, worker_id="w1",
                            stats={"speculative_workers": ["w2"]})
        inflight = self.jm.finalize_tasks(self.job.job_id, reason="job cancelled")
        self.assertEqual(sorted(inflight), [(maps[0].task_id, "w1"),
                                            (maps[0].task_id, "w2")])
        # Idempotent: a second pass finds nothing left to stop.
        self.assertEqual(self.jm.finalize_tasks(self.job.job_id), [])

    def test_late_reports_do_not_resurrect_tasks(self):
        maps = self._maps()
        tid = maps[0].task_id
        self.jm.update_task(self.job.job_id, tid, status=C.TASK_RUNNING, worker_id="w1")
        self.jm.cancel(self.job)

        self.scheduler.on_task_status({
            "job_id": self.job.job_id, "task_id": tid, "worker_id": "w1",
            "progress": 0.9, "records_processed": 50,
        })
        task = self.jm.get_task(self.job.job_id, tid)
        self.assertEqual(task.status, C.TASK_CANCELLED)
        self.assertEqual(task.records_processed, 0)

        self.scheduler.on_task_complete({
            "job_id": self.job.job_id, "task_id": tid, "worker_id": "w1",
            "status": C.TASK_SUCCEEDED, "records_processed": 100,
        })
        self.assertEqual(self.jm.get_task(self.job.job_id, tid).status, C.TASK_CANCELLED)


class TestFailCleanup(_ClusterFixture):
    def _exhaust(self, task_id):
        retried = True
        while retried:
            task = self.jm.get_task(self.job.job_id, task_id)
            retried = self.ft.handle_task_failure(self.job, task, "boom", "w1")

    def test_job_failure_stops_sibling_tasks(self):
        maps = self._maps()
        self.jm.update_task(self.job.job_id, maps[1].task_id,
                            status=C.TASK_RUNNING, worker_id="w2")
        self._exhaust(maps[0].task_id)

        self.assertEqual(self.jm.get_job(self.job.job_id).status, C.JOB_FAILED)
        # The failed task keeps its FAILED state; everything else is cancelled.
        self.assertEqual(self.jm.get_task(self.job.job_id, maps[0].task_id).status,
                         C.TASK_FAILED)
        for t in self.jm.tasks_for(self.job.job_id):
            if t.task_id != maps[0].task_id:
                self.assertEqual(t.status, C.TASK_CANCELLED, t.task_id)

    def test_completion_after_failure_is_ignored(self):
        maps = self._maps()
        self.jm.update_task(self.job.job_id, maps[1].task_id,
                            status=C.TASK_RUNNING, worker_id="w2")
        self._exhaust(maps[0].task_id)
        # A sibling that was still running finally reports success: too late.
        self.scheduler.on_task_complete({
            "job_id": self.job.job_id, "task_id": maps[1].task_id, "worker_id": "w2",
            "status": C.TASK_SUCCEEDED, "records_processed": 10,
        })
        self.assertEqual(self.jm.get_task(self.job.job_id, maps[1].task_id).status,
                         C.TASK_CANCELLED)
        self.assertEqual(self.jm.get_job(self.job.job_id).status, C.JOB_FAILED)


class TestExecutorCancellation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # Unreachable master: report posts fail fast and are swallowed.
        self.ex = Executor("w-test", self.tmp, "http://127.0.0.1:9",
                           ClusterConfig(), exec_mode="thread")

    def tearDown(self):
        self.ex.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _spec(self, job_id, task_id, rows=400000):
        return {
            "task_id": task_id, "job_id": job_id, "kind": C.TASK_MAP,
            "mapper": "wordcount_mapper", "reducer": "count_reducer",
            "params": {}, "partition_count": 2,
            "records": ["hello world foo bar"] * rows,
        }

    def _wait_empty(self, timeout=20.0):
        deadline = time.time() + timeout
        while time.time() < deadline and self.ex.running_count:
            time.sleep(0.05)
        return self.ex.running_count == 0

    def test_thread_backend_cancel_stops_task(self):
        self.assertTrue(self.ex.start_task(self._spec("job-a", "m-0000")))
        self.assertTrue(self.ex.cancel("m-0000", job_id="job-a"))
        self.assertTrue(self._wait_empty(), "cancelled task did not stop")

    def test_same_task_id_across_jobs_coexists(self):
        self.assertTrue(self.ex.start_task(self._spec("job-a", "m-0000")))
        self.assertTrue(self.ex.start_task(self._spec("job-b", "m-0000")))
        self.assertEqual(self.ex.running_count, 2)

        # Cancelling job-a's copy must not touch job-b's.
        self.assertTrue(self.ex.cancel("m-0000", job_id="job-a"))
        handle_b = self.ex._handles.get("job-b/m-0000")
        self.assertIsNotNone(handle_b)
        self.assertFalse(handle_b["cancel"].is_set())
        self.assertTrue(self.ex.cancel("m-0000", job_id="job-b"))
        self.assertTrue(self._wait_empty(), "cancelled tasks did not stop")


if __name__ == "__main__":
    unittest.main()
