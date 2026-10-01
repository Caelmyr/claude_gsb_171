"""Job and task data layer with lifecycle transitions.

The JobManager is the Master's in-memory index over the JSON store: it loads
jobs/tasks on boot, keeps them warm, and persists every mutation atomically.
It also owns the job lifecycle state machine
``PENDING -> MAP -> SHUFFLE -> REDUCE -> SUCCEEDED / FAILED / CANCELLED``.
Task-level transitions (retry, reassignment) are performed through the same
``update_task`` helper so the scheduler and fault-tolerance code share one
serialisation path.
"""

from __future__ import annotations

import os
import threading
from typing import Callable, Optional

from backend.common import constants as C
from backend.common.jsonutil import now_ms
from backend.common.logbus import LogBus
from backend.common.models import Job, Task, new_job
from backend.common.storage import Storage, list_files, list_subdirs, read_json
from backend.master.shard_planner import ShardPlanner
from backend.tasks.registry import has_mapper, has_reducer
from backend.tasks.samples import input_kind_for


class JobManager:
    def __init__(self, storage: Storage, config, logbus: LogBus) -> None:
        self.storage = storage
        self.config = config
        self.logbus = logbus
        self.planner = ShardPlanner(storage, config)
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, dict[str, Task]] = {}
        self._lock = threading.RLock()
        self._terminal_handlers: list[Callable[[Job, list[Task]], None]] = []
        self._load()

    def on_terminal(self, handler: Callable[[Job, list[Task]], None]) -> None:
        """Register a callback invoked after a job enters a terminal state."""
        self._terminal_handlers.append(handler)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _job_path(self, job_id: str) -> list[str]:
        return ["jobs", job_id, "job.json"]

    def _task_path(self, job_id: str, task_id: str) -> list[str]:
        return ["jobs", job_id, "tasks", f"{task_id}.json"]

    def _load(self) -> None:
        jobs_root = self.storage.path("jobs")
        for job_dir in list_subdirs(jobs_root):
            job_id = os.path.basename(job_dir)
            doc = self.storage.read("jobs", job_id, "job.json")
            if doc:
                self._jobs[job_id] = Job.from_dict(doc)
            tasks: dict[str, Task] = {}
            for path in list_files(os.path.join(job_dir, "tasks"), suffix=".json"):
                tdoc = read_json(path)
                if tdoc:
                    task = Task.from_dict(tdoc)
                    tasks[task.task_id] = task
            self._tasks[job_id] = tasks

    def save_job(self, job: Job) -> None:
        self.storage.write(job.to_dict(), *self._job_path(job.job_id))

    def save_task(self, job_id: str, task: Task) -> None:
        self.storage.write(task.to_dict(), *self._task_path(job_id, task.task_id))

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    def submit(self, payload: dict) -> Job:
        name = str(payload.get("name") or "untitled").strip() or "untitled"
        mapper = str(payload.get("mapper") or "")
        reducer = str(payload.get("reducer") or "")
        if not has_mapper(mapper):
            raise ValueError(f"unknown mapper: {mapper!r}")
        if not has_reducer(reducer):
            raise ValueError(f"unknown reducer: {reducer!r}")

        defaults = payload.get("_defaults") or {}
        num_map = int(payload.get("num_map_tasks", defaults.get("num_map_tasks", 8)))
        num_reduce = int(payload.get("num_reduce_tasks", defaults.get("num_reduce_tasks", 4)))
        input_rows = int(payload.get("input_rows", defaults.get("input_rows", 12000)))
        params = dict(payload.get("params") or {})
        params["input_kind"] = input_kind_for(mapper)

        job = new_job(name, mapper, reducer, num_map, num_reduce, input_rows, params)

        with self._lock:
            plan = self.planner.plan(job)
            job.num_map_tasks = len(plan["map_tasks"])
            job.map_task_ids = [t.task_id for t in plan["map_tasks"]]
            job.reduce_task_ids = [t.task_id for t in plan["reduce_tasks"]]
            job.status = C.JOB_MAP
            job.started_ms = now_ms()
            job.stats["total_records"] = plan["total_records"] + 1
            job.stats["input_kind"] = params["input_kind"]

            self._jobs[job.job_id] = job
            self._tasks[job.job_id] = {}
            for task in plan["map_tasks"] + plan["reduce_tasks"]:
                self._tasks[job.job_id][task.task_id] = task
                self.save_task(job.job_id, task)
            self.save_job(job)

        self.logbus.info(
            job.job_id, f"job submitted: {job.num_map_tasks} map / {job.num_reduce_tasks} reduce, "
                        f"{plan['total_records']} records", task_id="submit",
        )
        return job

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_ms, reverse=True)

    def get_task(self, job_id: str, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(job_id, {}).get(task_id)

    def tasks_for(self, job_id: str, kind: str = "") -> list[Task]:
        with self._lock:
            tasks = list(self._tasks.get(job_id, {}).values())
        if kind:
            tasks = [t for t in tasks if t.kind == kind]
        return sorted(tasks, key=lambda t: (t.kind, t.index))

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def set_job_status(self, job: Job, status: str) -> Job:
        with self._lock:
            job.status = status
            if status == C.JOB_MAP and not job.started_ms:
                job.started_ms = now_ms()
            if status in C.JOB_TERMINAL_STATES and not job.finished_ms:
                job.finished_ms = now_ms()
            self.save_job(job)
        return job

    def update_task(self, job_id: str, task_id: str, **fields) -> Optional[Task]:
        with self._lock:
            task = self._tasks.get(job_id, {}).get(task_id)
            if task is None:
                return None
            for key, value in fields.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            task.last_update_ms = now_ms()
            self.save_task(job_id, task)
            return task

    def update_job(self, job: Job, **fields) -> Job:
        with self._lock:
            for key, value in fields.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            self.save_job(job)
        return job

    def apply_task(self, job_id: str, task_id: str, fn: Callable[[Task], None],
                   require_active_job: bool = False) -> Optional[Task]:
        """Run ``fn(task)`` under the manager lock and persist the result."""
        with self._lock:
            job = self._jobs.get(job_id)
            task = self._tasks.get(job_id, {}).get(task_id)
            if job is None or task is None:
                return None
            if require_active_job and (job.is_terminal or task.status in C.TASK_TERMINAL_STATES):
                return None
            fn(task)
            task.last_update_ms = now_ms()
            self.save_task(job_id, task)
            return task

    def apply_job(self, job_id: str, fn: Callable[[Job], None]) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            fn(job)
            self.save_job(job)
            return job

    def mark_task_dispatched(self, job: Job, task: Task, worker_id: str,
                             speculative: bool = False) -> Optional[Task]:
        """Record that a worker accepted a task.

        Returns ``None`` if the job or task became terminal while the dispatch
        HTTP request was in flight.  The caller must then cancel the work on the
        worker instead of leaving an unmanaged process running.
        """
        with self._lock:
            task = self._tasks.get(job.job_id, {}).get(task.task_id)
            current_job = self._jobs.get(job.job_id)
            if (task is None or current_job is None or current_job.is_terminal
                    or task.status in C.TASK_TERMINAL_STATES):
                return None
            task.status = C.TASK_ASSIGNED
            task.assigned_ms = now_ms()
            if speculative:
                stats = dict(task.stats or {})
                stats.setdefault("speculative_workers", []).append(worker_id)
                task.stats = stats
            elif not task.worker_id:
                task.worker_id = worker_id
            task.last_update_ms = now_ms()
            self.save_task(job.job_id, task)
            return task

    def terminate(self, job: Job, status: str, error: str = "") -> tuple[Optional[Job], list[Task]]:
        """Atomically terminate a job and stop all of its still-open tasks.

        Already completed tasks are preserved. Every pending, assigned, running
        or retrying task is moved to a terminal state so it can neither be
        scheduled nor counted as occupied node capacity.  The returned tasks are
        the executions that remote workers may still need to kill.
        """
        if status not in C.JOB_TERMINAL_STATES:
            raise ValueError(f"invalid terminal job status: {status!r}")

        with self._lock:
            current = self._jobs.get(job.job_id)
            if current is None:
                return None, []
            if current.is_terminal:
                return current, []

            remote: list[Task] = []
            now = now_ms()
            for task in self._tasks.get(job.job_id, {}).values():
                if task.status in C.TASK_TERMINAL_STATES:
                    continue

                if task.worker_id and task.status in (C.TASK_ASSIGNED, C.TASK_RUNNING):
                    remote.append(task)

                task_status = C.TASK_FAILED if status == C.JOB_FAILED and task.error else C.TASK_CANCELLED
                task.status = task_status
                task.finished_ms = now
                task.last_update_ms = now
                if task_status == C.TASK_CANCELLED:
                    task.progress = 0.0
                    task.error = error
                elif error:
                    task.error = task.error or error
                task.retry_after_ms = 0
                self.save_task(job.job_id, task)

            current.status = status
            if error:
                current.error = error
            if not current.finished_ms:
                current.finished_ms = now_ms()
            self.save_job(current)
            terminal_job = current
            remote_tasks = list(remote)

        for handler in self._terminal_handlers:
            try:
                handler(terminal_job, remote_tasks)
            except Exception:  # noqa: BLE001 - handlers must not roll back state
                import traceback
                traceback.print_exc()
        return terminal_job, remote_tasks

    def cancel(self, job: Job) -> Job:
        cancelled, _ = self.terminate(job, C.JOB_CANCELLED, "job cancelled")
        if cancelled is not None:
            self.logbus.warn(cancelled.job_id, "job cancelled; all open tasks stopped", task_id="job")
        return cancelled or job

    def fail(self, job: Job, error: str) -> Job:
        failed, _ = self.terminate(job, C.JOB_FAILED, error)
        if failed is not None:
            self.logbus.error(failed.job_id, f"job failed: {error}", task_id="job")
        return failed or job

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------
    def stage_progress(self, job: Job) -> dict:
        tasks = self.tasks_for(job.job_id)
        progress: dict = {}
        for stage, kind in ((C.STAGE_MAP, C.TASK_MAP), (C.STAGE_REDUCE, C.TASK_REDUCE)):
            stage_tasks = [t for t in tasks if t.kind == kind]
            counts = {C.TASK_PENDING: 0, C.TASK_ASSIGNED: 0, C.TASK_RUNNING: 0,
                      C.TASK_RETRYING: 0, C.TASK_SUCCEEDED: 0, C.TASK_FAILED: 0,
                      C.TASK_CANCELLED: 0}
            for t in stage_tasks:
                counts[t.status] = counts.get(t.status, 0) + 1
            total = len(stage_tasks)
            done = counts[C.TASK_SUCCEEDED]
            progress[stage] = {
                "total": total,
                "done": done,
                "pct": round(done / total * 100.0, 1) if total else 0.0,
                "counts": counts,
            }
        return progress

    def job_summary(self, job: Job) -> dict:
        tasks = self.tasks_for(job.job_id)
        by_status: dict[str, int] = {}
        for t in tasks:
            by_status[t.status] = by_status.get(t.status, 0) + 1
        return {
            "job_id": job.job_id,
            "name": job.name,
            "mapper": job.mapper,
            "reducer": job.reducer,
            "status": job.status,
            "num_map_tasks": job.num_map_tasks,
            "num_reduce_tasks": job.num_reduce_tasks,
            "input_rows": job.input_rows,
            "created_ms": job.created_ms,
            "started_ms": job.started_ms,
            "finished_ms": job.finished_ms,
            "error": job.error,
            "params": job.params,
            "stats": job.stats,
            "task_status": by_status,
            "stage_progress": self.stage_progress(job),
        }
