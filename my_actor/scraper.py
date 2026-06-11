from __future__ import annotations

from dataclasses import dataclass
import logging
import multiprocessing as mp
import queue as queue_module
import sys
import time
from typing import Any

from apify import Actor
from jobspy import scrape_jobs
import pandas as pd
from requests import exceptions as requests_exceptions
from requests.adapters import HTTPAdapter, Retry
from urllib3.exceptions import MaxRetryError

from .schema import ActorInput, JobPost

DEFAULT_MAX_PARALLEL_BOARDS = 3
BOARD_PRIORITY = ("linkedin", "indeed", "zip_recruiter", "google", "glassdoor")
GLOBAL_TIMEOUT_SECONDS = 180
BOARD_TIMEOUT_SECONDS = 30
JOBSPY_PATCHED = False
EXPECTED_BOARD_ERROR_MARKERS = (
    "connection refused",
    "failed to establish a new connection",
    "max retries exceeded",
    "maxretryerror",
    "httpsconnectionpool",
    "read timed out",
    "connect timeout",
)


@dataclass(frozen=True)
class BoardError:
    error_type: str
    message: str
    expected: bool


@dataclass(frozen=True)
class ScrapeOutcome:
    jobs: list[dict[str, Any]]
    failed_boards: list[str]


def _clean_value(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return value


def _clean_str(value: Any, max_length: int | None = None) -> str:
    cleaned = _clean_value(value)
    if cleaned == "":
        return ""
    text = str(cleaned).strip()
    if max_length is not None:
        return text[:max_length]
    return text


def _clean_number_or_empty(value: Any) -> float | str:
    cleaned = _clean_value(value)
    if cleaned == "":
        return ""
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return _clean_str(cleaned)


def _clean_bool(value: Any) -> bool:
    cleaned = _clean_value(value)
    if cleaned == "":
        return False
    if isinstance(cleaned, str):
        return cleaned.strip().lower() in {"1", "true", "yes", "y", "remote"}
    return bool(cleaned)


def _dedup_key(job: dict[str, Any]) -> str:
    if job["job_url"]:
        return job["job_url"].strip().lower()
    if job["id"]:
        return f"{job['source']}:{job['id']}".strip().lower()
    return f"{job['source']}:{job['title']}:{job['company']}:{job['location']}".strip().lower()


def _is_expected_board_exception(exc: BaseException) -> bool:
    expected_types = (
        requests_exceptions.ConnectionError,
        requests_exceptions.Timeout,
        MaxRetryError,
        ConnectionRefusedError,
        TimeoutError,
    )
    current: BaseException | None = exc
    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        if isinstance(current, expected_types):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__

    message = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in message for marker in EXPECTED_BOARD_ERROR_MARKERS)


def _is_expected_board_error_message(error_type: str, message: str) -> bool:
    text = f"{error_type}: {message}".lower()
    return any(marker in text for marker in EXPECTED_BOARD_ERROR_MARKERS)


def _format_board_error(error: BoardError) -> str:
    message = " ".join(error.message.split())
    if len(message) > 300:
        message = f"{message[:297]}..."
    return f"{error.error_type}: {message}" if message else error.error_type


def _row_to_job(row: pd.Series) -> dict[str, Any]:
    job = JobPost(
        id=_clean_str(row.get("id", "")),
        title=_clean_str(row.get("title", "")),
        company=_clean_str(row.get("company", "")),
        location=_clean_str(row.get("location", "")),
        job_type=_clean_str(row.get("job_type", "")),
        date_posted=_clean_str(row.get("date_posted", "")),
        salary_min=_clean_number_or_empty(row.get("min_amount", "")),
        salary_max=_clean_number_or_empty(row.get("max_amount", "")),
        salary_currency=_clean_str(row.get("currency", "")),
        description=_clean_str(row.get("description", ""), max_length=3000),
        job_url=_clean_str(row.get("job_url", "")),
        source=_clean_str(row.get("site", "")),
        is_remote=_clean_bool(row.get("is_remote", False)),
        ai_score=None,
        ai_verdict=None,
        ai_missing_skills=None,
        keywords=None,
    )
    return job.model_dump()


def _scrape_query(inp: ActorInput, search_term: str, location: str, board: str) -> pd.DataFrame | None:
    return scrape_jobs(
        site_name=[board],
        search_term=search_term,
        location=location,
        results_wanted=inp.results_wanted,
        hours_old=inp.hours_old,
        country_indeed=inp.country_indeed,
        linkedin_fetch_description=False,
        verbose=0,
    )


def _suppress_jobspy_log_noise() -> None:
    for logger_name in (
        "JobSpy",
        "JobSpy:Indeed",
        "JobSpy:LinkedIn",
        "JobSpy:Glassdoor",
        "JobSpy:ZipRecruiter",
        "JobSpy:Google",
    ):
        logger = logging.getLogger(logger_name)
        logger.disabled = True
        logger.propagate = False


def _patch_jobspy_runtime() -> None:
    global JOBSPY_PATCHED
    if JOBSPY_PATCHED:
        return

    from jobspy.linkedin import LinkedIn
    from jobspy.util import RequestsRotating

    original_process_job = LinkedIn._process_job

    def safe_process_job(self: Any, job_card: Any, job_id: str, full_descr: bool) -> Any:
        try:
            return original_process_job(self, job_card, job_id, full_descr)
        except ValueError as exc:
            if "Invalid country string" in str(exc):
                return None
            raise

    safe_process_job._jobhunt_safe = True  # type: ignore[attr-defined]
    LinkedIn._process_job = safe_process_job

    def limited_setup_session(self: Any, has_retry: bool, delay: int) -> None:
        if has_retry:
            retries = Retry(
                total=1,
                connect=1,
                read=1,
                status=1,
                status_forcelist=[429, 500, 502, 503, 504],
                backoff_factor=0,
            )
            adapter = HTTPAdapter(max_retries=retries)
            self.mount("http://", adapter)
            self.mount("https://", adapter)

    limited_setup_session._jobhunt_limited = True  # type: ignore[attr-defined]
    RequestsRotating.setup_session = limited_setup_session
    JOBSPY_PATCHED = True


def _ordered_boards(boards: list[str]) -> list[str]:
    priority_boards = [board for board in BOARD_PRIORITY if board in boards]
    remaining_boards = [board for board in boards if board not in BOARD_PRIORITY]
    return priority_boards + remaining_boards


def _max_parallel_boards(inp: ActorInput) -> int:
    value = getattr(inp, "max_parallel_boards", DEFAULT_MAX_PARALLEL_BOARDS)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return DEFAULT_MAX_PARALLEL_BOARDS


def _scrape_query_process(
    inp_data: dict[str, Any],
    search_term: str,
    location: str,
    board: str,
    result_queue: Any,
) -> None:
    try:
        _suppress_jobspy_log_noise()
        _patch_jobspy_runtime()
        inp = ActorInput.model_validate(inp_data)
        df = _scrape_query(inp, search_term, location, board)
        records = [] if df is None or df.empty else df.to_dict("records")
        result_queue.put(("ok", records))
    except BaseException as exc:
        result_queue.put(
            (
                "error",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "expected": _is_expected_board_exception(exc),
                },
            )
        )


def _terminate_process(process: mp.Process) -> None:
    if not process.is_alive():
        return
    process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)


def _multiprocessing_context() -> Any:
    if sys.platform == "darwin" and "spawn" in mp.get_all_start_methods():
        return mp.get_context("spawn")
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context()


def _run_scrape_query_with_timeout(
    inp: ActorInput,
    search_term: str,
    location: str,
    board: str,
    timeout_seconds: float,
) -> tuple[pd.DataFrame | None, BoardError | None]:
    ctx = _multiprocessing_context()
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_scrape_query_process,
        args=(inp.model_dump(), search_term, location, board, result_queue),
    )
    process.daemon = True
    process.start()

    try:
        status, payload = result_queue.get(timeout=max(1, timeout_seconds))
    except queue_module.Empty:
        _terminate_process(process)
        return None, BoardError("Timeout", f"timed out after {int(timeout_seconds)} seconds", True)
    finally:
        result_queue.close()

    process.join(timeout=2)
    _terminate_process(process)

    if status == "error":
        if isinstance(payload, dict):
            error_type = str(payload.get("type") or "BoardError")
            message = str(payload.get("message") or "")
            expected = bool(payload.get("expected")) or _is_expected_board_error_message(error_type, message)
            return None, BoardError(error_type, message, expected)
        message = str(payload)
        return None, BoardError("BoardError", message, _is_expected_board_error_message("BoardError", message))

    return pd.DataFrame.from_records(payload), None


def _seconds_remaining(deadline: float) -> float:
    return deadline - time.monotonic()


def _deadline_expired(deadline: float) -> bool:
    return _seconds_remaining(deadline) <= 0


def _scrape_board(
    inp: ActorInput,
    search_term: str,
    location: str,
    board: str,
    deadline: float,
) -> tuple[pd.DataFrame | None, BoardError | None]:
    remaining = _seconds_remaining(deadline)
    if remaining <= 0:
        return None, BoardError("Timeout", "global scrape timeout reached", True)

    timeout_seconds = min(BOARD_TIMEOUT_SECONDS, remaining)
    Actor.log.info(f"Scraping board '{board}' for '{search_term}' in '{location}'")
    df, error = _run_scrape_query_with_timeout(inp, search_term, location, board, timeout_seconds)
    if error:
        reason = _format_board_error(error)
        if error.expected:
            Actor.log.warning(f"Board unavailable/blocked; continuing. Board: {board}; reason: {reason}")
        else:
            Actor.log.warning(f"Board failed; continuing. Board: {board}; reason: {reason}")
        return None, error

    if df is None or df.empty:
        Actor.log.warning(f"Board returned 0 results. Board: {board}; query: '{search_term}'; location: '{location}'")
        return None, None

    return df, None


def _scrape_frames(
    inp: ActorInput,
    search_term: str,
    location: str,
    boards: list[str],
    disabled_boards_this_run: set[str],
    failed_boards: set[str],
    jobs: list[dict[str, Any]],
    seen: set[str],
    deadline: float,
) -> None:
    max_parallel_boards = _max_parallel_boards(inp)

    for i in range(0, len(boards), max_parallel_boards):
        if len(jobs) >= inp.max_results_total:
            Actor.log.info(f"Reached max_total_results={inp.max_results_total}; stopping scrape")
            break
        if _deadline_expired(deadline):
            Actor.log.warning("Global scrape timeout reached; returning partial results")
            break

        board_batch = [board for board in boards[i : i + max_parallel_boards] if board not in disabled_boards_this_run]
        if not board_batch:
            continue

        Actor.log.info(f"Scraping board batch: {board_batch}")
        for board in board_batch:
            if len(jobs) >= inp.max_results_total:
                Actor.log.info(f"Reached max_total_results={inp.max_results_total}; stopping scrape")
                break
            if _deadline_expired(deadline):
                Actor.log.warning("Global scrape timeout reached; returning partial results")
                break
            if board in disabled_boards_this_run:
                Actor.log.warning(f"Skipping disabled board for remainder of run. Board: {board}")
                continue

            df, error = _scrape_board(inp, search_term, location, board, deadline)
            if error:
                disabled_boards_this_run.add(board)
                failed_boards.add(board)
                Actor.log.warning(f"Marked board unavailable for remainder of run. Board: {board}")
                if board == "indeed":
                    Actor.log.warning("Indeed failed; disabling Indeed for this run")
                continue
            if df is not None and not df.empty:
                _append_deduped_jobs(df, jobs, seen, inp.max_results_total)


def _scrape_sync(inp: ActorInput) -> ScrapeOutcome:
    _patch_jobspy_runtime()
    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    disabled_boards_this_run: set[str] = set()
    failed_boards: set[str] = set()
    deadline = time.monotonic() + GLOBAL_TIMEOUT_SECONDS

    search_terms = inp.effective_search_terms()
    locations = inp.effective_locations()
    boards = _ordered_boards(inp.boards)

    for search_term in search_terms:
        if len(jobs) >= inp.max_results_total:
            Actor.log.info(f"Reached max_total_results={inp.max_results_total}; stopping scrape")
            return ScrapeOutcome(jobs=jobs, failed_boards=sorted(failed_boards))
        for location in locations:
            if len(jobs) >= inp.max_results_total:
                Actor.log.info(f"Reached max_total_results={inp.max_results_total}; stopping scrape")
                return ScrapeOutcome(jobs=jobs, failed_boards=sorted(failed_boards))
            if _deadline_expired(deadline):
                Actor.log.warning("Global scrape timeout reached; returning partial results")
                return ScrapeOutcome(jobs=jobs, failed_boards=sorted(failed_boards))

            active_boards = [board for board in boards if board not in disabled_boards_this_run]
            if not active_boards:
                Actor.log.warning("All selected boards are unavailable; returning partial results")
                return ScrapeOutcome(jobs=jobs, failed_boards=sorted(failed_boards))

            Actor.log.info(f"Scraping '{search_term}' in '{location}' from {boards}")
            jobs_before_combo = len(jobs)
            _scrape_frames(
                inp,
                search_term,
                location,
                active_boards,
                disabled_boards_this_run,
                failed_boards,
                jobs,
                seen,
                deadline,
            )

            if len(jobs) == jobs_before_combo:
                Actor.log.info(f"No jobs found for '{search_term}' in '{location}'")

    if len(jobs) > inp.max_results_total:
        jobs = jobs[: inp.max_results_total]

    Actor.log.info(f"Applying max_total_results={inp.max_results_total}, returning {len(jobs)} jobs")

    return ScrapeOutcome(jobs=jobs, failed_boards=sorted(failed_boards))


def scrape(inp: ActorInput) -> list[dict[str, Any]]:
    return scrape_with_metadata(inp).jobs


def scrape_with_metadata(inp: ActorInput) -> ScrapeOutcome:
    return _scrape_sync(inp)


def _append_deduped_jobs(
    df: pd.DataFrame,
    jobs: list[dict[str, Any]],
    seen: set[str],
    limit: int,
) -> None:
    for _, row in df.fillna("").iterrows():
        if len(jobs) >= limit:
            break
        try:
            job = _row_to_job(row)
        except Exception:
            Actor.log.exception("Skipping malformed job row")
            continue

        key = _dedup_key(job)
        if key in seen:
            continue
        seen.add(key)
        jobs.append(job)
