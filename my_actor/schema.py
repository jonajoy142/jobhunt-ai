from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SUPPORTED_BOARDS = ("linkedin", "indeed", "glassdoor", "zip_recruiter", "google")
DEFAULT_BOARDS = ("linkedin",)
INPUT_ALIASES = {
    "posted_within_hours": "hours_old",
    "results_wanted_per_board": "results_wanted",
    "max_total_results": "max_results_total",
    "job_boards": "boards",
    "resume_or_key_skills": "resume_text",
    "minimum_ai_score": "min_ai_score",
    "minimum_jev_score": "min_jev_score",
    "indeed_country": "country_indeed",
}


class ActorInput(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    search_term: str = "software engineer"
    search_terms: list[str] = Field(default_factory=list)
    location: str = Field(default="Remote", min_length=1)
    locations: list[str] = Field(default_factory=list)
    hours_old: int = Field(default=720, ge=1)
    results_wanted: int = Field(default=5, ge=1, le=1000)
    max_results_total: int = Field(default=10, ge=1, le=5000)
    boards: list[str] = Field(default_factory=lambda: list(DEFAULT_BOARDS), min_length=1)
    enable_ai_scoring: bool = False
    resume_text: str = ""
    anthropic_api_key: str = ""
    min_ai_score: int = Field(default=0, ge=0, le=100)
    min_jev_score: float = Field(default=0.0, ge=0.0, le=100.0)
    country_indeed: str = Field(default="USA", min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_input_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        for public_name, internal_name in INPUT_ALIASES.items():
            if public_name in normalized:
                normalized[internal_name] = normalized.pop(public_name)
        return normalized

    @field_validator("search_terms", "locations")
    @classmethod
    def normalize_string_list(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            item = value.strip()
            if item and item not in normalized:
                normalized.append(item)
        return normalized

    @field_validator("boards")
    @classmethod
    def validate_boards(cls, boards: list[str]) -> list[str]:
        normalized: list[str] = []
        invalid: list[str] = []

        for board in boards:
            board_name = board.strip().lower()
            if board_name not in SUPPORTED_BOARDS:
                invalid.append(board)
                continue
            if board_name not in normalized:
                normalized.append(board_name)

        if invalid:
            supported = ", ".join(SUPPORTED_BOARDS)
            raise ValueError(f"Unsupported boards: {invalid}. Supported boards: {supported}")
        if not normalized:
            raise ValueError("At least one supported board must be selected")

        return normalized

    @model_validator(mode="after")
    def validate_effective_search(self) -> ActorInput:
        if not self.search_terms and not self.search_term.strip():
            raise ValueError("Provide search_terms or search_term")
        if not self.locations and not self.location.strip():
            raise ValueError("Provide locations or location")
        return self

    def effective_search_terms(self) -> list[str]:
        return self.search_terms or [self.search_term.strip()]

    def effective_locations(self) -> list[str]:
        return self.locations or [self.location.strip()]


class JobPost(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = ""
    title: str = ""
    company: str = ""
    location: str = ""
    job_type: str = ""
    date_posted: str = ""
    salary_min: float | str = ""
    salary_max: float | str = ""
    salary_currency: str = ""
    description: str = ""
    job_url: str = ""
    source: str = ""
    is_remote: bool = False
    ai_score: int | None = Field(default=None, ge=0, le=100)
    ai_verdict: str | None = None
    ai_missing_skills: list[str] | None = None
    keywords: list[str] | None = None
    jev_score: float | None = Field(default=None, ge=0.0, le=100.0)
    jev_tier: str | None = None
    p_offer: float | None = Field(default=None, ge=0.0, le=1.0)
    market_value: float | None = None
    expected_return: float | None = None
