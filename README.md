## What does JobHunt AI do?

**JobHunt AI scrapes job listings from top job boards and evaluates them using the open-source Laya System-1 decision model and the Job Expected Value (JEV) mathematical framework—requiring zero external API keys.** The default stable board is [LinkedIn Jobs](https://www.linkedin.com/jobs/), with optional best-effort support for [Indeed](https://www.indeed.com/), [Google Jobs](https://careers.google.com/), [ZipRecruiter](https://www.ziprecruiter.com/), and [Glassdoor](https://www.glassdoor.com/).

Run it by entering a role, location, freshness window, and selected boards in the Apify Input tab. When AI evaluation is enabled, JobHunt AI runs the **Laya System-1 decision engine locally** (at ~5ms per job) to calculate candidate fit, missing skills, keywords, and offer probabilities $P(\text{Offer})$, which feeds directly into the **JEV (Job Expected Value)** valuation engine:

$$\text{JEV} = P(\text{Offer} \mid \text{Apply}) \times \text{Market Value}(\text{Job}) - \text{Application Effort}$$

Every job is enriched with `jev_score` (0-100), `jev_tier` (`HIGH_EV`, `MEDIUM_EV`, `LOW_EV`), `expected_return` ($), and `market_value` ($).

## Why use JobHunt AI?

Most job scrapers simply dump raw unranked HTML rows into CSVs. JobHunt AI is built for job seekers, recruiters, talent researchers, and founders who want an actionable, high-signal feed prioritized by true career return on application effort.

- **Zero-Key Open-Source AI**: Powered by the **Laya System-1** non-autoregressive decision model running directly inside the container—no Anthropic or OpenAI API keys needed.
- **JEV Decision Framework**: Identifies high-leverage job opportunities by computing the risk-adjusted expected financial and career payoff.
- **Deduplicated Multi-Board Coverage**: Aggregates LinkedIn, Indeed, Google, ZipRecruiter, and Glassdoor without duplicate listings.
- **Instant Latency**: Scores 50 jobs in under 2 seconds.

## How to use JobHunt AI

1. Open the Actor on Apify and go to the Input tab.
2. Enter one fallback `search_term`, or use `search_terms` to search several roles in one run.
3. Set one fallback `location`, or use `locations` to search several markets in one run.
4. Choose how fresh results should be with `posted_within_hours`.
5. Keep the default `job_boards` value as `["linkedin"]` for the most stable and fastest runs.
6. Add Indeed only when you explicitly want best-effort Indeed coverage and can tolerate upstream connection failures or blocking.
7. Optionally enable AI scoring and provide your Anthropic API key plus resume or skills text.
8. Run the Actor and open the Dataset tab to view or export results.

For local development, use the safe default-style input:

```bash
apify run --input '{"search_term":"software engineer","search_terms":[],"location":"Remote","locations":[],"posted_within_hours":720,"results_wanted_per_board":5,"max_total_results":10,"job_boards":["linkedin"],"enable_ai_scoring":false,"resume_or_key_skills":"","anthropic_api_key":"","minimum_ai_score":0,"indeed_country":"USA"}'
```

## Input

Configure the Actor from the Apify Input tab or send JSON through the API.

```json
{
  "search_term": "software engineer",
  "search_terms": [],
  "location": "Remote",
  "locations": [],
  "posted_within_hours": 720,
  "results_wanted_per_board": 5,
  "max_total_results": 10,
  "job_boards": ["linkedin"],
  "enable_ai_scoring": false,
  "resume_or_key_skills": "",
  "anthropic_api_key": "",
  "minimum_ai_score": 0,
  "indeed_country": "USA"
}
```

Provide either `search_terms` or `search_term`. Provide either `locations` or `location`. When list fields are provided, the Actor searches every search term and location combination, then deduplicates jobs across the full combined result set until `max_total_results` is reached. LinkedIn is the recommended default board for stable tests. Indeed, Google Jobs, ZipRecruiter, and Glassdoor are optional best-effort sources and may return zero results or be unavailable due to upstream blocking, rate limits, or network restrictions.

AI scoring is optional. Leave `enable_ai_scoring` as `false` for raw job listings. Set it to `true` only when you also provide `anthropic_api_key` and `resume_or_key_skills`. `minimum_ai_score` only applies when AI scoring is enabled.

The Actor also accepts the previous internal field names for backward compatibility: `hours_old`, `results_wanted`, `max_results_total`, `boards`, `resume_text`, `min_ai_score`, and `country_indeed`.

## Output

The Actor pushes one JSON object per job to the default Apify dataset. You can download the dataset in various formats such as JSON, HTML, CSV, or Excel.

```json
[
  {
    "id": "example-123",
    "title": "Software Engineer",
    "company": "Example Labs",
    "location": "Remote",
    "job_type": "fulltime",
    "date_posted": "2026-06-03",
    "salary_min": 150000,
    "salary_max": 210000,
    "salary_currency": "USD",
    "description": "Build production software systems and improve platform reliability...",
    "job_url": "https://example.com/jobs/software-engineer",
    "source": "linkedin",
    "is_remote": true,
    "ai_score": null,
    "ai_verdict": null,
    "ai_missing_skills": null,
    "keywords": null
  }
]
```

If selected boards fail or return no jobs, the Actor still exits successfully and writes a clear status item:

```json
{
  "status": "no_results",
  "message": "No jobs found or selected boards were unavailable. Try fewer filters or another board.",
  "failed_boards": ["indeed"]
}
```

## Data table

| Field | Description |
| --- | --- |
| `status` | `no_results` when no job rows were produced |
| `message` | Human-readable explanation for no-result runs |
| `failed_boards` | Boards disabled after upstream failures during the run |
| `id` | JobSpy-provided job identifier when available |
| `title` | Job title |
| `company` | Hiring company |
| `location` | Job location |
| `job_type` | Job type when provided by the board |
| `date_posted` | Posted date returned by the source |
| `salary_min` | Minimum salary when available |
| `salary_max` | Maximum salary when available |
| `salary_currency` | Salary currency when available |
| `description` | Job description capped at 3,000 characters |
| `job_url` | Direct job listing URL |
| `source` | Board that returned the job |
| `is_remote` | Whether the job is marked remote |
| `ai_score` | Claude resume-match score from 0 to 100, or null |
| `ai_verdict` | `apply_now`, `consider`, `skip`, `error`, or null |
| `ai_missing_skills` | Skills Claude identified as gaps, or null |
| `keywords` | Top technical skills and keywords extracted by Claude when AI scoring is enabled, or null |

## Pricing / Cost estimation

How much does it cost to scrape job boards with JobHunt AI? Actor pricing should be configured as pay-per-event, with one event per job result pushed to the dataset. The safe default input requests only 5 LinkedIn results and caps the run at 10 total results, which keeps validation and trial runs small.

AI scoring uses your own Anthropic API key. A typical Claude scoring call for one job depends on text length and Anthropic pricing. When AI scoring is enabled, JobHunt AI also makes a second small Claude call for keyword extraction on jobs that pass your score filter. Disable AI scoring when you only need raw job listings.

## Tips or Advanced options

Use the default LinkedIn-only selection for the most stable runs and for Apify Store validation. Add Indeed only when you specifically need it; Indeed can reject or refuse upstream API connections from hosted environments, and the Actor will disable Indeed after the first connection failure in a run instead of retrying it for every query/location.

Use fewer boards, fewer search/location combinations, and lower `results_wanted_per_board` values for faster runs. Use `posted_within_hours` to keep output focused on fresh jobs and reduce duplicate or stale listings. For best AI scoring, paste a concise resume or skills summary with core technologies, seniority, domain experience, and location preferences.

If you only want highly relevant jobs, enable AI scoring and set `minimum_ai_score` to 70 or 80. For broad market research, keep AI scoring disabled and export all results to CSV or Excel.

## FAQ, disclaimers, and support

Is scraping job boards legal? Public web data can often be collected, but you are responsible for using this Actor in compliance with each website's Terms of Service, robots.txt guidance, applicable laws, and Apify policies. Do not collect or store sensitive personal data unless you have a lawful basis and explicit permission.

Why are some boards missing results? Job boards change markup, rate limits, availability, and anti-bot behavior. Indeed in particular can be unavailable or blocked depending on upstream access. JobHunt AI handles per-board failures by continuing with the remaining boards and writing `failed_boards` in no-result runs.

Does this include Upwork, Fiverr, Freelancer, Toptal, Wellfound, RemoteOK, or WeWorkRemotely? Not in V1. Those boards require custom scrapers beyond JobSpy support and should be added only if user demand justifies the maintenance cost.

For bugs, feature requests, or custom job board integrations, use the Issues tab on Apify. Custom solutions are available for teams that need additional boards, scheduled alerting, or downstream workflow integrations.
