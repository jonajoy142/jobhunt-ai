from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from apify import Actor, Event
from pydantic import ValidationError

from .schema import ActorInput
from .scraper import scrape_with_metadata

CHARGED_EVENT_NAME = "job-result"
QA_MAX_RESULTS_TOTAL = 10
QA_RESULTS_WANTED = 5
QA_SEARCH_TERM_LIMIT = 1
QA_LOCATION_LIMIT = 1
RUN_TIMEOUT_GUARD_SECONDS = 270
NO_RESULTS_ITEM = {
    "status": "no_results",
    "message": "No jobs found or selected boards were unavailable. Try fewer filters or another board.",
}


async def on_aborting() -> None:
    Actor.log.warning("Actor aborting; exiting after short SDK persistence delay")
    await asyncio.sleep(1)
    await Actor.exit()


async def _push_jobs(jobs: list[dict[str, Any]]) -> int:
    if not jobs:
        return 0
    await Actor.push_data(jobs, charged_event_name=CHARGED_EVENT_NAME)
    return len(jobs)


async def _push_no_results(failed_boards: list[str]) -> None:
    await Actor.push_data({**NO_RESULTS_ITEM, "failed_boards": failed_boards})


def _is_apify_qa_run() -> bool:
    origin = os.getenv("APIFY_META_ORIGIN", "").strip().upper()
    return origin in {"TEST", "QA"} or "TEST" in origin or "QA" in origin


def _apply_qa_caps(inp: ActorInput) -> ActorInput:
    search_terms = inp.effective_search_terms()[:QA_SEARCH_TERM_LIMIT]
    locations = inp.effective_locations()[:QA_LOCATION_LIMIT]
    capped = {
        **inp.model_dump(),
        "search_term": search_terms[0],
        "search_terms": search_terms,
        "location": locations[0],
        "locations": locations,
        "hours_old": min(inp.hours_old, 720),
        "results_wanted": min(inp.results_wanted, QA_RESULTS_WANTED),
        "max_results_total": min(inp.max_results_total, QA_MAX_RESULTS_TOTAL),
        "boards": ["linkedin"],
        "enable_ai_scoring": False,
    }
    return ActorInput.model_validate(capped)


async def main() -> None:
    async with Actor:
        run_deadline = time.monotonic() + RUN_TIMEOUT_GUARD_SECONDS
        Actor.on(Event.ABORTING, on_aborting)

        raw = await Actor.get_input() or {}
        try:
            inp = ActorInput.model_validate(raw)
        except ValidationError:
            Actor.log.exception("Invalid Actor input")
            raise

        if _is_apify_qa_run():
            inp = _apply_qa_caps(inp)
            Actor.log.warning(
                "Apify QA/test run detected; capped input to "
                f"max_results_total={inp.max_results_total}, results_wanted={inp.results_wanted}, "
                f"search_terms={len(inp.effective_search_terms())}, locations={len(inp.effective_locations())}, "
                f"boards={inp.boards}, enable_ai_scoring={inp.enable_ai_scoring}"
            )

        search_terms = inp.effective_search_terms()
        locations = inp.effective_locations()
        Actor.log.info(
            f"Scraping {len(search_terms)} search term(s) x {len(locations)} location(s) from {inp.boards}"
        )
        scrape_outcome = scrape_with_metadata(inp)
        jobs = scrape_outcome.jobs
        Actor.log.info(f"Scraped {len(jobs)} jobs after dedup")

        if not jobs:
            await _push_no_results(scrape_outcome.failed_boards)
            Actor.log.info(
                "No jobs were found or all selected boards were unavailable; "
                f"wrote no_results dataset item with failed_boards={scrape_outcome.failed_boards}"
            )
            return

        pushed_count = 0
        if inp.enable_ai_scoring:
            from .scorer import evaluate_job

            Actor.log.info(
                f"Laya Decision Model & JEV Valuation enabled; evaluating {len(jobs)} jobs locally"
            )

            scored_jobs: list[dict[str, Any]] = []
            for index, job in enumerate(jobs):
                if time.monotonic() >= run_deadline:
                    remaining_jobs = jobs[index:]
                    if inp.min_ai_score > 0 or inp.min_jev_score > 0:
                        Actor.log.warning(
                            "Run timeout guard reached; dropping remaining unscored jobs because score filters are active"
                        )
                    else:
                        Actor.log.warning(
                            f"Run timeout guard reached; pushing {len(remaining_jobs)} remaining job(s) unscored"
                        )
                        pushed_count += await _push_jobs(remaining_jobs)
                        scored_jobs.extend(remaining_jobs)
                    break

                eval_metrics = evaluate_job(
                    job,
                    resume_text=inp.resume_text,
                    anthropic_api_key=inp.anthropic_api_key,
                    hours_old=inp.hours_old,
                )
                job.update(eval_metrics)

                if inp.min_ai_score > 0 and (job.get("ai_score") or 0) < inp.min_ai_score:
                    continue
                if inp.min_jev_score > 0 and (job.get("jev_score") or 0.0) < inp.min_jev_score:
                    continue

                scored_jobs.append(job)
                await Actor.push_data(job, charged_event_name=CHARGED_EVENT_NAME)
                pushed_count += 1

            jobs = scored_jobs
            if inp.min_ai_score > 0 or inp.min_jev_score > 0:
                Actor.log.info(
                    f"{len(jobs)} jobs matched score thresholds (min_ai_score={inp.min_ai_score}, min_jev_score={inp.min_jev_score})"
                )
        else:
            if inp.min_ai_score > 0 or inp.min_jev_score > 0:
                Actor.log.warning("Score threshold filters are ignored when AI scoring is disabled")
            pushed_count = await _push_jobs(jobs)

        Actor.log.info(f"Done. {pushed_count} jobs pushed to dataset.")
