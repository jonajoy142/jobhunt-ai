import unittest
from unittest.mock import patch
import pandas as pd
from requests.exceptions import ConnectionError, Timeout
from my_actor.schema import ActorInput
from my_actor.scraper import (
    _clean_str,
    _clean_number_or_empty,
    _clean_bool,
    _dedup_key,
    _ordered_boards,
    _append_deduped_jobs,
    _is_expected_board_exception,
    _is_expected_board_error_message,
)

class TestScraper(unittest.TestCase):
    def test_cleaning_functions(self):
        self.assertEqual(_clean_str(None), "")
        self.assertEqual(_clean_str("  hello world  "), "hello world")
        self.assertEqual(_clean_str("long text here", max_length=4), "long")

        self.assertEqual(_clean_number_or_empty("120000"), 120000.0)
        self.assertEqual(_clean_number_or_empty(150000), 150000.0)
        self.assertEqual(_clean_number_or_empty(None), "")
        self.assertEqual(_clean_number_or_empty(" competitive "), "competitive")

        self.assertTrue(_clean_bool(True))
        self.assertTrue(_clean_bool("1"))
        self.assertTrue(_clean_bool("true"))
        self.assertTrue(_clean_bool("Yes"))
        self.assertTrue(_clean_bool("remote"))
        self.assertFalse(_clean_bool(False))
        self.assertFalse(_clean_bool(None))
        self.assertFalse(_clean_bool("0"))
        self.assertFalse(_clean_bool("no"))

    def test_dedup_key(self):
        job1 = {"job_url": "https://example.com/jobs/1", "id": "1", "source": "linkedin", "title": "Dev", "company": "Co", "location": "US"}
        job2 = {"job_url": "", "id": "abc-456", "source": "indeed", "title": "Dev", "company": "Co", "location": "US"}
        job3 = {"job_url": "", "id": "", "source": "google", "title": "Dev", "company": "Co", "location": "US"}

        self.assertEqual(_dedup_key(job1), "https://example.com/jobs/1")
        self.assertEqual(_dedup_key(job2), "indeed:abc-456")
        self.assertEqual(_dedup_key(job3), "google:dev:co:us")

    def test_ordered_boards(self):
        # Priority order: linkedin, indeed, zip_recruiter, google, glassdoor
        boards = ["glassdoor", "google", "linkedin"]
        ordered = _ordered_boards(boards)
        self.assertEqual(ordered, ["linkedin", "google", "glassdoor"])

    def test_expected_board_exceptions(self):
        self.assertTrue(_is_expected_board_exception(ConnectionError("failed to connect")))
        self.assertTrue(_is_expected_board_exception(Timeout("read timed out")))
        self.assertTrue(_is_expected_board_error_message("ConnectionError", "Max retries exceeded with url"))
        self.assertFalse(_is_expected_board_error_message("KeyError", "missing id column"))

    def test_append_deduped_jobs(self):
        df = pd.DataFrame([
            {"id": "1", "title": "SWE", "company": "A", "location": "Remote", "job_url": "https://a.com/1", "site": "linkedin"},
            {"id": "2", "title": "SWE", "company": "B", "location": "Remote", "job_url": "https://b.com/2", "site": "linkedin"},
            # Duplicate URL:
            {"id": "3", "title": "SWE", "company": "A", "location": "Remote", "job_url": "https://a.com/1", "site": "linkedin"},
        ])
        jobs = []
        seen = set()
        _append_deduped_jobs(df, jobs, seen, limit=10)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["id"], "1")
        self.assertEqual(jobs[1]["id"], "2")

    def test_append_deduped_jobs_respects_limit(self):
        df = pd.DataFrame([
            {"id": str(i), "title": "SWE", "company": f"Co_{i}", "location": "Remote", "job_url": f"https://a.com/{i}", "site": "linkedin"}
            for i in range(10)
        ])
        jobs = []
        seen = set()
        _append_deduped_jobs(df, jobs, seen, limit=3)
        self.assertEqual(len(jobs), 3)

    @patch("my_actor.scraper.time.sleep")
    @patch("my_actor.scraper._run_scrape_query_with_timeout")
    def test_acceptance_scenario_1_linkedin_fails_indeed_succeeds(self, mock_run, mock_sleep):
        # Scenario 1: LinkedIn fails, Indeed succeeds with jobs -> returns Indeed jobs
        from my_actor.scraper import scrape_with_metadata, BoardError
        
        def side_effect(inp, search_term, location, board, timeout):
            if board == "linkedin":
                return None, BoardError("Timeout", "timed out after 30 seconds", True)
            if board == "indeed":
                df = pd.DataFrame([
                    {"id": "ind-1", "title": "Dev", "company": "Co", "location": "Remote", "job_url": "https://indeed.com/1", "site": "indeed"}
                ])
                return df, None
            return None, None

        mock_run.side_effect = side_effect
        inp = ActorInput(search_term="dev", location="Remote", boards=["linkedin", "indeed"], max_results_total=10)
        
        outcome = scrape_with_metadata(inp)
        self.assertEqual(len(outcome.jobs), 1)
        self.assertEqual(outcome.jobs[0]["id"], "ind-1")
        self.assertEqual(outcome.failed_boards, ["linkedin"])

    @patch("my_actor.scraper.time.sleep")
    @patch("my_actor.scraper._run_scrape_query_with_timeout")
    def test_acceptance_scenario_2_linkedin_google_fail_indeed_succeeds(self, mock_run, mock_sleep):
        # Scenario 2: LinkedIn fails, Google fails, Indeed succeeds -> returns Indeed jobs
        from my_actor.scraper import scrape_with_metadata, BoardError

        def side_effect(inp, search_term, location, board, timeout):
            if board == "linkedin":
                return None, BoardError("Timeout", "timed out", True)
            if board == "google":
                return None, BoardError("ConnectionError", "connection refused", True)
            if board == "indeed":
                df = pd.DataFrame([
                    {"id": "ind-1", "title": "Dev", "company": "Co", "location": "Remote", "job_url": "https://indeed.com/1", "site": "indeed"}
                ])
                return df, None
            return None, None

        mock_run.side_effect = side_effect
        inp = ActorInput(search_term="dev", location="Remote", boards=["linkedin", "google", "indeed"], max_results_total=10)

        outcome = scrape_with_metadata(inp)
        self.assertEqual(len(outcome.jobs), 1)
        self.assertEqual(outcome.jobs[0]["id"], "ind-1")
        self.assertEqual(sorted(outcome.failed_boards), ["google", "linkedin"])

    @patch("my_actor.scraper.time.sleep")
    @patch("my_actor.scraper._run_scrape_query_with_timeout")
    def test_acceptance_scenario_3_all_boards_return_zero_jobs(self, mock_run, mock_sleep):
        # Scenario 3: All boards return 0 jobs successfully -> Return no_results, failed_boards = []
        from my_actor.scraper import scrape_with_metadata

        mock_run.return_value = (pd.DataFrame(), None)
        inp = ActorInput(search_term="dev", location="Remote", boards=["indeed", "google"], max_results_total=10)

        outcome = scrape_with_metadata(inp)
        self.assertEqual(len(outcome.jobs), 0)
        self.assertEqual(outcome.failed_boards, [])

    @patch("my_actor.scraper.time.sleep")
    @patch("my_actor.scraper._run_scrape_query_with_timeout")
    def test_acceptance_scenario_4_all_boards_throw_errors(self, mock_run, mock_sleep):
        # Scenario 4: All boards throw errors/timeouts -> Return no_results, failed_boards contains all failed boards
        from my_actor.scraper import scrape_with_metadata, BoardError

        def side_effect(inp, search_term, location, board, timeout):
            return None, BoardError("Timeout", f"{board} timed out", True)

        mock_run.side_effect = side_effect
        inp = ActorInput(search_term="dev", location="Remote", boards=["linkedin", "indeed", "google"], max_results_total=10)

        outcome = scrape_with_metadata(inp)
        self.assertEqual(len(outcome.jobs), 0)
        self.assertEqual(sorted(outcome.failed_boards), ["google", "indeed", "linkedin"])

    @patch("my_actor.scraper.time.sleep")
    @patch("my_actor.scraper._run_scrape_query_with_timeout")
    def test_acceptance_scenario_5_one_succeeds_one_fails_one_zero(self, mock_run, mock_sleep):
        # Scenario 5: One board returns 10 jobs, Another board times out, Another returns 0 -> Return the 10 jobs. Do not return no_results.
        from my_actor.scraper import scrape_with_metadata, BoardError

        def side_effect(inp, search_term, location, board, timeout):
            if board == "indeed":
                df = pd.DataFrame([
                    {"id": f"ind-{i}", "title": "Dev", "company": "Co", "location": "Remote", "job_url": f"https://indeed.com/{i}", "site": "indeed"}
                    for i in range(10)
                ])
                return df, None
            if board == "linkedin":
                return None, BoardError("Timeout", "timed out", True)
            if board == "google":
                return pd.DataFrame(), None
            return None, None

        mock_run.side_effect = side_effect
        inp = ActorInput(search_term="dev", location="Remote", boards=["indeed", "linkedin", "google"], max_results_total=20)

        outcome = scrape_with_metadata(inp)
        self.assertEqual(len(outcome.jobs), 10)
        self.assertEqual(outcome.failed_boards, ["linkedin"])

    @patch("my_actor.scraper.time.sleep")
    @patch("my_actor.scraper._run_scrape_query_with_timeout")
    def test_retry_transient_failure_then_succeed(self, mock_run, mock_sleep):
        # Transient failure on attempt 1, success on attempt 2
        from my_actor.scraper import scrape_with_metadata, BoardError

        attempts = {"linkedin": 0}
        def side_effect(inp, search_term, location, board, timeout):
            attempts[board] += 1
            if attempts[board] == 1:
                return None, BoardError("Timeout", "timed out", True)
            df = pd.DataFrame([
                {"id": "li-1", "title": "Dev", "company": "Co", "location": "Remote", "job_url": "https://linkedin.com/1", "site": "linkedin"}
            ])
            return df, None

        mock_run.side_effect = side_effect
        inp = ActorInput(search_term="dev", location="Remote", boards=["linkedin"], max_results_total=10)

        outcome = scrape_with_metadata(inp)
        self.assertEqual(len(outcome.jobs), 1)
        self.assertEqual(outcome.failed_boards, [])
        self.assertEqual(attempts["linkedin"], 2)
        mock_sleep.assert_called_once()

if __name__ == "__main__":
    unittest.main()
