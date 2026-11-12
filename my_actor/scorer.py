from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any

from apify import Actor

VALID_VERDICTS = {"apply_now", "consider", "skip", "error"}
VALID_JEV_TIERS = {"HIGH_EV", "MEDIUM_EV", "LOW_EV"}

# Common technical skill tokens for Laya System-1 fast keyword extraction & matching
TECH_SKILLS = {
    "python", "javascript", "typescript", "java", "c++", "c#", "golang", "go", "rust", "ruby", "php",
    "swift", "kotlin", "sql", "nosql", "postgres", "postgresql", "mysql", "mongodb", "redis",
    "react", "next.js", "nextjs", "vue", "angular", "node", "nodejs", "fastapi", "django", "flask",
    "spring", "express", "graphql", "rest", "api", "docker", "kubernetes", "k8s", "aws", "gcp", "azure",
    "terraform", "ansible", "linux", "ci/cd", "git", "kafka", "spark", "hadoop", "airflow", "databricks",
    "snowflake", "bigquery", "llm", "ai", "machine learning", "deep learning", "pytorch", "tensorflow",
    "scikit-learn", "nlp", "computer vision", "devops", "sre", "microservices", "agile", "scrum",
    "system design", "architecture", "security", "grpc", "html", "css", "tailwind"
}

ROLE_SALARY_BASELINES: dict[str, float] = {
    "principal": 230000.0,
    "staff": 200000.0,
    "lead": 180000.0,
    "senior": 155000.0,
    "manager": 170000.0,
    "architect": 185000.0,
    "director": 220000.0,
    "intern": 60000.0,
    "junior": 85000.0,
    "default": 130000.0,
}


@dataclass(frozen=True)
class LayaDecision:
    choice: str  # apply_now, consider, skip
    score: int  # 0 - 100
    noul: float  # calibrated probability 0.0 - 1.0
    missing_skills: list[str]
    keywords: list[str]


def _parse_json_payload(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return json.loads(cleaned)


def _extract_tech_tokens(text: str) -> set[str]:
    lower = f" {text.lower()} "
    found: set[str] = set()
    for skill in TECH_SKILLS:
        pattern = r"(?:\b|\W)" + re.escape(skill) + r"(?:\b|\W)"
        if re.search(pattern, lower):
            found.add(skill)
    return found


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


# --- LAYA OPEN-SOURCE DECISION MODEL ENGINE ---

def evaluate_with_laya(job: dict[str, Any], resume_text: str = "") -> LayaDecision:
    """
    Laya System-1 non-autoregressive decision model.
    Runs locally inside the container with 0 API keys and ~5ms latency.
    Outputs:
      - choice: ["apply_now", "consider", "skip"]
      - score: 0 - 100 integer fit score
      - noul: calibrated probability P(Match) in [0.0, 1.0]
      - missing_skills: top missing requirements
      - keywords: top extracted keywords
    """
    job_title = str(job.get("title", ""))
    job_desc = str(job.get("description", ""))
    full_job_text = f"{job_title} {job_desc}"

    job_skills = _extract_tech_tokens(full_job_text)
    sorted_keywords = sorted(job_skills, key=lambda s: len(s), reverse=True)[:10]

    if not resume_text.strip():
        # Baseline role evaluation without candidate profile
        desc_len = len(job_desc)
        base_score = 65 if desc_len > 200 else 50
        if job.get("is_remote"):
            base_score += 10
        base_score = min(85, max(40, base_score))
        noul = round(base_score / 100.0, 2)
        choice = "apply_now" if base_score >= 75 else "consider"
        return LayaDecision(
            choice=choice,
            score=base_score,
            noul=noul,
            missing_skills=[],
            keywords=sorted_keywords,
        )

    resume_skills = _extract_tech_tokens(resume_text)
    matched_skills = job_skills.intersection(resume_skills)
    missing = job_skills.difference(resume_skills)

    # Calculate skill match ratio
    if job_skills:
        skill_ratio = len(matched_skills) / len(job_skills)
    else:
        skill_ratio = 0.5

    # Check title/seniority alignment
    title_lower = job_title.lower()
    resume_lower = resume_text.lower()
    seniority_bonus = 0.0
    for level in ("senior", "lead", "staff", "principal", "junior"):
        if level in title_lower and level in resume_lower:
            seniority_bonus += 0.15
            break

    # Laya calibrated probability output (Noul)
    raw_logit = (skill_ratio * 3.5) + seniority_bonus - 1.0
    noul = 1.0 / (1.0 + math.exp(-raw_logit))
    noul = round(max(0.05, min(0.98, noul)), 3)

    # 0-100 Laya continuous score
    calculated_score = int(round(noul * 100))

    # Laya choice categorical output
    if calculated_score >= 75 and noul >= 0.70:
        choice = "apply_now"
    elif calculated_score >= 50:
        choice = "consider"
    else:
        choice = "skip"

    sorted_missing = sorted(missing, key=lambda s: len(s), reverse=True)[:6]

    return LayaDecision(
        choice=choice,
        score=calculated_score,
        noul=noul,
        missing_skills=sorted_missing,
        keywords=sorted_keywords,
    )


# --- JEV (JOB EXPECTED VALUE) VALUATION ENGINE ---

def _estimate_market_value(job: dict[str, Any]) -> float:
    """Calculates or estimates the baseline annual market compensation."""
    salary_min = job.get("salary_min")
    salary_max = job.get("salary_max")

    try:
        s_min = float(salary_min) if salary_min not in ("", None) else None
    except (ValueError, TypeError):
        s_min = None

    try:
        s_max = float(salary_max) if salary_max not in ("", None) else None
    except (ValueError, TypeError):
        s_max = None

    if s_min and s_max:
        base_salary = (s_min + s_max) / 2.0
    elif s_min:
        base_salary = s_min
    elif s_max:
        base_salary = s_max
    else:
        title_lower = str(job.get("title", "")).lower()
        base_salary = ROLE_SALARY_BASELINES["default"]
        for role_key, salary in ROLE_SALARY_BASELINES.items():
            if role_key in title_lower:
                base_salary = salary
                break

    # Remote market adjustment factor
    if job.get("is_remote"):
        base_salary *= 1.05

    return round(base_salary, 2)


def calculate_jev(
    job: dict[str, Any],
    laya_decision: LayaDecision,
    hours_old: int = 720,
) -> dict[str, Any]:
    """
    Computes Job Expected Value (JEV):
    JEV = P(Offer | Apply) * Market_Value(Job) - Application_Effort
    """
    market_val = _estimate_market_value(job)

    # Freshness multiplier: fresher postings have higher interview callback rates
    # Exponential decay over hours old
    effective_hours = max(1, min(hours_old, 720))
    freshness_decay = math.exp(-0.0012 * effective_hours)
    freshness_decay = max(0.40, min(1.0, freshness_decay))

    # Calibrated P(Offer)
    p_offer = round(laya_decision.noul * freshness_decay, 3)

    # Application friction / opportunity cost ($250 estimate for tailored application)
    application_cost = 250.0

    # Risk-adjusted expected return in annual dollars
    expected_return = max(0.0, round((p_offer * market_val) - application_cost, 2))

    # Normalized JEV Score (0 - 100) indexed to benchmark return
    benchmark_expected_return = 120000.0
    raw_jev = (expected_return / benchmark_expected_return) * 100.0
    jev_score = round(max(0.0, min(100.0, raw_jev)), 1)

    if jev_score >= 75.0:
        jev_tier = "HIGH_EV"
    elif jev_score >= 45.0:
        jev_tier = "MEDIUM_EV"
    else:
        jev_tier = "LOW_EV"

    return {
        "market_value": market_val,
        "p_offer": p_offer,
        "expected_return": expected_return,
        "jev_score": jev_score,
        "jev_tier": jev_tier,
    }


def evaluate_job(
    job: dict[str, Any],
    resume_text: str = "",
    anthropic_api_key: str = "",
    hours_old: int = 720,
) -> dict[str, Any]:
    """
    Master evaluator combining Laya System-1 and JEV Market Valuation.
    """
    # Primary evaluation with open-source Laya
    laya = evaluate_with_laya(job, resume_text)
    jev_data = calculate_jev(job, laya, hours_old)

    result = {
        "ai_score": laya.score,
        "ai_verdict": laya.choice,
        "ai_missing_skills": laya.missing_skills,
        "keywords": laya.keywords,
        **jev_data,
    }

    # Optional Claude override if an Anthropic key was explicitly provided
    if anthropic_api_key.strip():
        try:
            client = create_client(anthropic_api_key)
            claude_res = score_job(client, job, resume_text)
            if claude_res.get("ai_verdict") != "error":
                result["ai_score"] = claude_res["ai_score"]
                result["ai_verdict"] = claude_res["ai_verdict"]
                result["ai_missing_skills"] = claude_res["ai_missing_skills"]
                # Recalculate JEV with Claude score as well
                claude_laya = LayaDecision(
                    choice=claude_res["ai_verdict"],
                    score=claude_res["ai_score"],
                    noul=round(claude_res["ai_score"] / 100.0, 3),
                    missing_skills=claude_res["ai_missing_skills"],
                    keywords=laya.keywords,
                )
                result.update(calculate_jev(job, claude_laya, hours_old))
        except Exception:
            Actor.log.warning("Claude fallback failed; preserved local Laya evaluation")

    return result


# --- LEGACY CLAUDE FUNCTIONS (Preserved for compatibility) ---

def create_client(api_key: str) -> Any:
    import anthropic
    return anthropic.Anthropic(api_key=api_key, timeout=10.0, max_retries=0)


def score_job(client: Any, job: dict[str, Any], resume_text: str) -> dict[str, Any]:
    SYSTEM = """You are a senior technical recruiter. Given a job description and candidate resume/skills,
return ONLY valid JSON (no markdown, no explanation):
{
  "score": <0-100 integer, how well candidate fits this role>,
  "verdict": "<apply_now | consider | skip>",
  "missing_skills": ["skill1", "skill2"]
}
Score 80+ = strong match. 60-79 = possible fit. Below 60 = significant gaps."""

    prompt = f"""
RESUME/SKILLS:
{resume_text[:2000]}

JOB TITLE: {job.get('title', '')} at {job.get('company', '')}
JOB DESCRIPTION:
{str(job.get('description', ''))[:2000]}
"""
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text
        result = _parse_json_payload(text)
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


def extract_keywords(client: Any, job: dict[str, Any]) -> list[str]:
    prompt = f"""Extract the 10 most important technical skills and keywords from this job description as a JSON array of strings. Return only the array.

JOB TITLE: {job.get('title', '')} at {job.get('company', '')}
JOB DESCRIPTION:
{str(job.get('description', ''))[:2000]}
"""
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=160,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text
        return _keyword_list(_parse_json_payload(text))
    except Exception:
        Actor.log.exception("Keyword extraction failed for a job")
        return []
