"""Fault tolerance: task retries, worker-death reassignment, speculation.

This module turns failures into *recoverable events*:

* a failed task is retried up to ``max_attempts`` with exponential backoff, and
  only then marks the job failed;
* a worker that stops heartbeating has every in-flight task reassigned to other
  workers (the original is treated as a lost attempt, not a permanent failure);
* stragglers — tasks running much longer than the median — are detected and a
  speculative duplicate is launched on another worker, winner takes all.

Every decision is recorded both as a structured ``FaultEvent`` document (for the
fault-recovery page) and as a log line (for the log-search page).
"""

from __future__ import annotations

import statistics
from typing import Optional

from backend.common import constants as C
from backend.common.ids import new_id
from backend.common.jsonutil import now_ms
from backend.common.logbus import LogBus
from backend.common.models import FaultEvent, Job, Task, WorkerRecord
from backend.common.storage import Storage
from backend.master.job_manager import JobManager


class FaultTolerance:
    def __init__(
        self,
        storage: Storage,
        job_manager: JobManager,
        config,
        logbus: LogBus,
    ) -> None:
        self.storage = storage
        self.job_manager = job_manager
        self.config = config
        self.logbus = logbus

    # ------------------------------------------------------------------
    def _record(self, job: Job, kind: str, message: str, task: Optional[Task] = None,
                worker_id: str = "", detail: Optional[dict] = None) -> FaultEvent:
        event = FaultEvent(
            fault_id=new_id("fault"),
            job_id=job.job_id,
            task_id=task.task_id if task else "",
            worker_id=worker_id,
            kind=kind,
            message=message,
            attempt=task.attempts if task else 0,
            created_ms=now_ms(),
            detail=detail or {},
        )
        self.storage.write(event.to_dict(), "jobs", job.job_id, "faults", f"{event.fault_id}.json")
        self.logbus.warn(
            job.job_id, f"[{kind}] {message}",
            task_id=task.task_id if task else "job", worker_id=worker_id,
        )
        return event

    # ------------------------------------------------------------------
    def handle_task_failure(self, job: Job, task: Task, error: str, worker_id: str = "") -> bool:
        """Return True if the task was queued for retry, False if the job is doomed."""
        # Another failure can terminate the whole job while a duplicate/old attempt
        # report is in flight; never resurrect or reschedule a terminal job.
        current_job = self.job_manager.get_job(job.job_id)
        if current_job is None or current_job.is_terminal:
            return False
        job = current_job
        max_attempts = int(self.config.max_attempts)
        if task.attempts < max_attempts:
            backoff_ms = int(self.config.retry_backoff_base_sec * (2 ** task.attempts))
            next_attempts = task.attempts + 1
            retry_after_ms = now_ms() + backoff_ms

            def queue_retry(t: Task) -> None:
                t.status = C.TASK_RETRYING
                t.worker_id = None
                t.error = error
                t.attempts = next_attempts
                t.retry_after_ms = retry_after_ms
                t.progress = 0.0
                t.records_processed = 0
                t.records_emitted = 0

            queued = self.job_manager.apply_task(
                job.job_id, task.task_id, queue_retry, require_active_job=True,
            )
            if queued is None:
                return False
            self._record(
                job, "task_failed", f"task {task.task_id} failed ({error}); retrying",
                task=queued, worker_id=worker_id,
                detail={"attempt": next_attempts, "max_attempts": max_attempts,
                        "backoff_ms": backoff_ms},
            )
            return True

        final_attempts = task.attempts + 1

        def mark_failed(t: Task) -> None:
            t.status = C.TASK_FAILED
            t.error = error
            t.attempts = final_attempts

        failed = self.job_manager.apply_task(
            job.job_id, task.task_id, mark_failed, require_active_job=True,
        )
        if failed is None:
            return False
        self._record(
            job, "task_failed", f"task {task.task_id} exhausted {max_attempts} attempts",
            task=failed, worker_id=worker_id,
        )
        self.job_manager.fail(job, f"task {task.task_id} failed after {max_attempts} attempts: {error}")
        return False

    def handle_worker_death(self, worker: WorkerRecord) -> int:
        """Reassign every in-flight task on a dead worker. Returns count."""
        reassigned = 0
        for job in self.job_manager.list_jobs():
            if job.is_terminal:
                continue
            for task in self.job_manager.tasks_for(job.job_id):
                speculative_workers = list((task.stats or {}).get("speculative_workers", []))
                on_dead_worker = (
                    task.worker_id == worker.worker_id
                    or worker.worker_id in speculative_workers
                )
                if on_dead_worker and task.status in C.TASK_ACTIVE_STATES:
                    self._record(
                        job, "worker_dead",
                        f"worker {worker.name} lost; reassigning task {task.task_id}",
                        task=task, worker_id=worker.worker_id,
                    )

                    def reassign(t: Task, worker_id=worker.worker_id) -> None:
                        if t.worker_id == worker_id:
                            t.worker_id = None
                        stats = dict(t.stats or {})
                        stats["speculative_workers"] = [
                            wid for wid in stats.get("speculative_workers", []) if wid != worker_id
                        ]
                        t.stats = stats
                        t.status = C.TASK_RETRYING
                        t.error = f"worker {worker.name} died"
                        t.retry_after_ms = 0

                    self.job_manager.apply_task(job.job_id, task.task_id, reassign)
                    reassigned += 1
        return reassigned

    def find_stragglers(self, job: Job) -> list[Task]:
        """Tasks running far longer than the median, still awaiting a duplicate."""
        if not self.config.speculative_execution:
            return []
        tasks = self.job_manager.tasks_for(job.job_id)
        running = [t for t in tasks if t.status == C.TASK_RUNNING]
        finished = [t for t in tasks if t.status == C.TASK_SUCCEEDED and t.duration_ms > 0]
        if not running or len(finished) < 2:
            return []
        median = statistics.median(t.duration_ms for t in finished)
        threshold = median * float(self.config.speculation_threshold)
        stragglers: list[Task] = []
        for t in running:
            elapsed = now_ms() - (t.started_ms or now_ms())
            if elapsed > threshold and not (t.stats or {}).get("speculated"):
                stragglers.append(t)
        return stragglers

    def mark_speculated(self, job: Job, task: Task) -> None:
        stats = dict(task.stats or {})
        stats["speculated"] = True
        self.job_manager.update_task(job.job_id, task.task_id, stats=stats)

    def list_faults(self, job_id: str) -> list[dict]:
        from backend.common.storage import list_files, read_json
        root = self.storage.path("jobs", job_id, "faults")
        faults = []
        for path in list_files(root, suffix=".json"):
            doc = read_json(path)
            if doc:
                faults.append(doc)
        faults.sort(key=lambda d: d.get("created_ms", 0))
        return faults
