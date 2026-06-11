from __future__ import annotations

import json
from typing import Any

import anthropic
from apify import Actor

SYSTEM = """You are a senior technical recruiter. Given a job description and candidate resume/skills,
return ONLY valid JSON (no markdown, no explanation):
{
  "score": <0-100 integer, how well candidate fits this role>,
  "verdict": "<apply_now | consider | skip>",
  "missing_skills": ["skill1", "skill2"]
}
Score 80+ = strong match. 60-79 = possible fit. Below 60 = significant gaps."""

VALID_VERDICTS = {"apply_now", "consider", "skip", "error"}


def create_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key, timeout=10.0, max_retries=0)


def _score_value(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, score))


def _missing_skills(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(skill).strip() for skill in value if str(skill).strip()]


def _keyword_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    keywords: list[str] = []
    for keyword in value:
        item = str(keyword).strip()
        if item and item not in keywords:
            keywords.append(item)
        if len(keywords) == 10:
            break
    return keywords


def score_job(client: anthropic.Anthropic, job: dict[str, Any], resume_text: str) -> dict[str, Any]:
    prompt = f"""
RESUME/SKILLS:
{resume_text[:2000]}

JOB TITLE: {job['title']} at {job['company']}
JOB DESCRIPTION:
{job['description'][:2000]}
"""
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text
        result = json.loads(text)
        verdict = str(result.get("verdict", "")).strip()
        if verdict not in VALID_VERDICTS:
            verdict = "skip"

        return {
            "ai_score": _score_value(result.get("score", 0)),
            "ai_verdict": verdict,
            "ai_missing_skills": _missing_skills(result.get("missing_skills", [])),
        }
    except Exception:
        Actor.log.exception("AI scoring failed for a job")
        return {"ai_score": 0, "ai_verdict": "error", "ai_missing_skills": []}


def extract_keywords(client: anthropic.Anthropic, job: dict[str, Any]) -> list[str]:
    prompt = f"""Extract the 10 most important technical skills and keywords from this job description as a JSON array of strings. Return only the array.

JOB TITLE: {job['title']} at {job['company']}
JOB DESCRIPTION:
{job['description'][:2000]}
"""
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=160,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text
        return _keyword_list(json.loads(text))
    except Exception:
        Actor.log.exception("Keyword extraction failed for a job")
        return []
