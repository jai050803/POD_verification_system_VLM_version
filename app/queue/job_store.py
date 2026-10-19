import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import pg8000.native

from app.config import get_settings
from app.queue.models import Job, JobStatus

logger = logging.getLogger(__name__)


class JobStore:
    def __init__(self, database_url: Optional[str] = None):
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}
        self._db_url = database_url or get_settings().DATABASE_URL
        self._db_conn: Optional[pg8000.native.Connection] = None
        self._init_db()

    def _get_connection(self) -> Optional[pg8000.native.Connection]:
        if not self._db_url:
            return None
        try:
            if self._db_conn is not None:
                try:
                    self._db_conn.run("SELECT 1")
                    return self._db_conn
                except Exception:
                    try:
                        self._db_conn.close()
                    except Exception:
                        pass
                    self._db_conn = None

            parsed = urlparse(self._db_url)
            self._db_conn = pg8000.native.Connection(
                user=parsed.username or "postgres",
                password=parsed.password or "",
                host=parsed.hostname or "localhost",
                port=parsed.port or 5432,
                database=parsed.path.lstrip("/") or "pod_verification",
            )
            return self._db_conn
        except Exception as e:
            logger.warning(f"Failed to connect to PostgreSQL database: {e}")
            self._db_conn = None
            return None

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            if not conn:
                return
            try:
                conn.run("""
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id VARCHAR(255) PRIMARY KEY,
                        status VARCHAR(50) NOT NULL,
                        webhook_url TEXT NOT NULL,
                        reference_id TEXT,
                        doc_type_hint TEXT,
                        created_at DOUBLE PRECISION NOT NULL,
                        updated_at DOUBLE PRECISION NOT NULL,
                        result JSONB,
                        error TEXT,
                        pages_processed INTEGER DEFAULT 0,
                        webhook_delivered BOOLEAN DEFAULT FALSE
                    );
                """)
                rows = conn.run("""
                    SELECT job_id, status, webhook_url, reference_id, doc_type_hint,
                           created_at, updated_at, result, error, pages_processed, webhook_delivered
                    FROM jobs
                """)
                for row in rows:
                    job = Job(
                        job_id=row[0],
                        status=JobStatus(row[1]),
                        webhook_url=row[2],
                        reference_id=row[3],
                        doc_type_hint=row[4],
                        created_at=row[5],
                        updated_at=row[6],
                        result=row[7] if isinstance(row[7], dict) else (json.loads(row[7]) if row[7] else None),
                        error=row[8],
                        pages_processed=row[9] or 0,
                        webhook_delivered=bool(row[10]),
                    )
                    self._jobs[job.job_id] = job
            except Exception as e:
                logger.warning(f"Error initializing jobs table from PostgreSQL: {e}")

    def _persist_job(self, job: Job) -> None:
        # Assumes caller holds self._lock
        conn = self._get_connection()
        if not conn:
            return
        try:
            result_json = json.dumps(job.result) if job.result is not None else None
            conn.run("""
                INSERT INTO jobs (
                    job_id, status, webhook_url, reference_id, doc_type_hint,
                    created_at, updated_at, result, error, pages_processed, webhook_delivered
                ) VALUES (
                    :job_id, :status, :webhook_url, :reference_id, :doc_type_hint,
                    :created_at, :updated_at, :result, :error, :pages_processed, :webhook_delivered
                )
                ON CONFLICT (job_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    webhook_url = EXCLUDED.webhook_url,
                    reference_id = EXCLUDED.reference_id,
                    doc_type_hint = EXCLUDED.doc_type_hint,
                    created_at = EXCLUDED.created_at,
                    updated_at = EXCLUDED.updated_at,
                    result = EXCLUDED.result,
                    error = EXCLUDED.error,
                    pages_processed = EXCLUDED.pages_processed,
                    webhook_delivered = EXCLUDED.webhook_delivered;
            """,
                job_id=job.job_id,
                status=job.status.value,
                webhook_url=job.webhook_url,
                reference_id=job.reference_id,
                doc_type_hint=job.doc_type_hint,
                created_at=job.created_at,
                updated_at=job.updated_at,
                result=result_json,
                error=job.error,
                pages_processed=job.pages_processed,
                webhook_delivered=job.webhook_delivered,
            )
        except Exception as e:
            logger.error(f"Failed to persist job {job.job_id} to database: {e}")

    def create(
        self,
        job_id: str,
        webhook_url: str,
        reference_id: Optional[str] = None,
        doc_type_hint: Optional[str] = None,
    ) -> Job:
        now = time.time()
        job = Job(
            job_id=job_id,
            status=JobStatus.QUEUED,
            webhook_url=webhook_url,
            reference_id=reference_id,
            doc_type_hint=doc_type_hint,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._persist_job(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def set_processing(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job.status = JobStatus.PROCESSING
            job.updated_at = time.time()
            self._persist_job(job)
            return job

    def set_completed(self, job_id: str, result: dict) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job.status = JobStatus.COMPLETED
            job.result = result
            job.updated_at = time.time()
            self._persist_job(job)
            return job

    def set_failed(self, job_id: str, error: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job.status = JobStatus.FAILED
            job.error = error
            job.updated_at = time.time()
            self._persist_job(job)
            return job

    def set_webhook_failed(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job.status = JobStatus.WEBHOOK_FAILED
            job.updated_at = time.time()
            self._persist_job(job)
            return job

    def set_webhook_delivered(self, job_id: str, delivered: bool = True) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job.webhook_delivered = delivered
            job.updated_at = time.time()
            self._persist_job(job)
            return job

    def set_pages_processed(self, job_id: str, pages: int) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job.pages_processed = pages
            job.updated_at = time.time()
            self._persist_job(job)
            return job

    def list_all(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())


# Singleton instance
job_store = JobStore()
