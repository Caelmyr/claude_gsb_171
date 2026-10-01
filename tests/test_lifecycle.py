"""Tests for job termination and worker-side task cancellation."""

import shutil
import tempfile
import time
import unittest

from backend.common import constants as C
from backend.common.config import ClusterConfig
from backend.common.logbus import LogBus
from backend.common.storage import Storage
from backend.master.job_manager import JobManager
from backend.master.scheduler import Scheduler
from backend.tasks import registry
from backend.worker.executor import Executor


class TestJobTermination(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = Storage(self.tmp)
        self.config = ClusterConfig()
        self.logbus = LogBus(self.storage)
        self.jm = JobManager(self.storage, self.config, self.logbus)
        self.job = self.jm.submit({
            "name": "terminate",
            "mapper": "wordcount_mapper",
            "reducer": "count_reducer",
            "num_map_tasks": 3,
            "num_reduce_tasks": 2,
            "input_rows": 100,
            "params": {},
        })
        self.tasks = self.jm.tasks_for(self.job.job_id)
        self.tasks[0].worker_id = "w-1"
        self.tasks[1].worker_id = "w-2"
        self.jm.update_task(self.job.job_id, self.tasks[0].task_id,
                           status=C.TASK_RUNNING, worker_id="w-1")
        self.jm.update_task(self.job.job_id, self.tasks[1].task_id,
                           status=C.TASK_ASSIGNED, worker_id="w-2")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cancel_marks_every_open_task_terminal_and_reports_remote_work(self):
        remote_ids = []

        def on_terminal(job, remote):
            self.assertEqual(job.status, C.JOB_CANCELLED)
            remote_ids.extend((t.task_id, t.worker_id) for t in remote)

        self.jm.on_terminal(on_terminal)
        cancelled = self.jm.cancel(self.job)

        self.assertEqual(cancelled.status, C.JOB_CANCELLED)
        self.assertEqual(
            {t.status for t in self.jm.tasks_for(self.job.job_id)},
            {C.TASK_CANCELLED},
        )
        self.assertEqual(
            sorted(remote_ids),
            sorted([(self.tasks[0].task_id, "w-1"), (self.tasks[1].task_id, "w-2")]),
        )

    def test_failure_marks_open_tasks_cancelled_without_duplicate_handler(self):
        calls = []
        self.jm.on_terminal(lambda job, remote: calls.append(1))

        failed = self.jm.fail(self.job, "fatal")

        self.assertEqual(failed.status, C.JOB_FAILED)
        self.assertTrue(all(t.status == C.TASK_CANCELLED for t in self.jm.tasks_for(self.job.job_id)))
        self.assertEqual(len(calls), 1)
        # A second failure must not rerun cancellation or overwrite the first state.
        self.jm.fail(self.job, "later")
        self.assertEqual(len(calls), 1)


class TestSchedulerCancellation(unittest.TestCase):
    class Response:
        def __init__(self, data=None, ok=True):
            self.data = data or {}
            self.ok = ok

    class Client:
        def __init__(self, owner):
            self.owner = owner

        def post(self, url, body=None, timeout=None):
            return self.owner.post(url, body or {}, timeout)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = Storage(self.tmp)
        self.config = ClusterConfig(metric_interval_sec=0.01)
        self.logbus = LogBus(self.storage)
        self.jm = JobManager(self.storage, self.config, self.logbus)
        self.job = self.jm.submit({
            "name": "scheduler-cancel",
            "mapper": "wordcount_mapper",
            "reducer": "count_reducer",
            "num_map_tasks": 2,
            "num_reduce_tasks": 1,
            "input_rows": 50,
            "params": {},
        })
        self.task = self.jm.tasks_for(self.job.job_id, C.TASK_MAP)[0]
        self.jm.update_task(self.job.job_id, self.task.task_id,
                           status=C.TASK_RUNNING, worker_id="w-1")
        self.requests = []
        self.workers = [type("Worker", (), {
            "worker_id": "w-1",
            "address": "http://worker-1",
        })()]

        class Registry:
            def __init__(self, workers):
                self._workers = workers
            def get(self, worker_id):
                return self._workers[0] if worker_id == "w-1" else None
            def alive(self):
                return list(self._workers)
            def all(self):
                return list(self._workers)
            def save(self, worker):
                pass
            def task_finished(self, *args):
                pass

        self.scheduler = Scheduler(
            self.storage, self.jm, Registry(self.workers), None, None, None,
            self.config, self.logbus,
        )
        self.scheduler.client = self.Client(self)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def post(self, url, body, timeout):
        self.requests.append((url, body))
        return self.Response({"cancelled": 1})

    def test_cancel_requests_whole_job_on_worker(self):
        self.jm.cancel(self.job)
        deadline = time.time() + 2.0
        while time.time() < deadline and not self.requests:
            time.sleep(0.01)
        self.assertEqual(self.requests, [
            ("http://worker-1/job/" + self.job.job_id + "/cancel", {}),
        ])


class TestExecutorCancellation(unittest.TestCase):
    @staticmethod
    def _slow_mapper(records, params):
        for _ in range(10_000_000):
            yield ("x", 1)

    def setUp(self):
        registry.register_mapper("test_slow_mapper", self._slow_mapper)
        self.tmp = tempfile.mkdtemp()
        self.executor = Executor(
            worker_id="w-test",
            data_root=self.tmp,
            master_url="http://127.0.0.1:1",
            config=ClusterConfig(),
            exec_mode="thread",
        )

    def tearDown(self):
        self.executor.shutdown()
        time.sleep(0.05)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _wait_for_count(self, expected):
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if self.executor.running_count == expected:
                return
            time.sleep(0.02)
        self.assertEqual(self.executor.running_count, expected)

    def test_same_task_id_in_different_jobs_is_independent(self):
        spec = {
            "task_id": "m-0000",
            "kind": C.TASK_MAP,
            "mapper": "test_slow_mapper",
            "reducer": "count_reducer",
            "params": {},
            "partition_count": 1,
            "records": list(range(1000)),
        }
        first = dict(spec, job_id="job-a")
        second = dict(spec, job_id="job-b")
        self.assertTrue(self.executor.start_task(first))
        self.assertTrue(self.executor.start_task(second))
        self._wait_for_count(2)

        self.assertEqual(self.executor.cancel("job-a", "m-0000"), 1)
        self._wait_for_count(1)
        self.assertEqual(self.executor.running_execution_keys(), ["job-b:m-0000"])

    def test_cancel_job_stops_all_its_tasks(self):
        for i in range(2):
            self.assertTrue(self.executor.start_task({
                "job_id": "job-c",
                "task_id": f"m-{i:04d}",
                "kind": C.TASK_MAP,
                "mapper": "test_slow_mapper",
                "reducer": "count_reducer",
                "params": {},
                "partition_count": 1,
                "records": list(range(1000)),
            }))
        self.assertEqual(self.executor.cancel("job-c"), 2)
        self._wait_for_count(0)


class TestProcessExecutorCancellation(unittest.TestCase):
    def test_process_job_cancel_terminates_child(self):
        from backend.tasks import registry

        def slow_mapper(records, params):
            import time
            while True:
                yield ("x", 1)
                time.sleep(0.001)

        registry.register_mapper("test_process_slow_mapper", slow_mapper)
        tmp = tempfile.mkdtemp()
        executor = Executor(
            worker_id="w-process",
            data_root=tmp,
            master_url="http://127.0.0.1:1",
            config=ClusterConfig(),
            exec_mode="process",
        )
        try:
            self.assertTrue(executor.start_task({
                "job_id": "job-process",
                "task_id": "m-0000",
                "kind": C.TASK_MAP,
                "mapper": "test_process_slow_mapper",
                "reducer": "count_reducer",
                "params": {},
                "partition_count": 1,
                "records": list(range(10)),
            }))
            time.sleep(0.2)
            self.assertEqual(executor.cancel("job-process"), 1)
            deadline = time.time() + 5.0
            while time.time() < deadline and executor.running_count:
                time.sleep(0.02)
            self.assertEqual(executor.running_count, 0)
        finally:
            executor.shutdown()
            time.sleep(0.1)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
