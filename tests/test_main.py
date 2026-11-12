import os
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from my_actor.schema import ActorInput
from my_actor.main import _is_apify_qa_run, _apply_qa_caps, main
from my_actor.scraper import ScrapeOutcome

class TestMain(unittest.IsolatedAsyncioTestCase):
    def test_qa_origin_detection(self):
        with patch.dict(os.environ, {"APIFY_META_ORIGIN": "QA"}):
            self.assertTrue(_is_apify_qa_run())
        with patch.dict(os.environ, {"APIFY_META_ORIGIN": "TEST"}):
            self.assertTrue(_is_apify_qa_run())
        with patch.dict(os.environ, {"APIFY_META_ORIGIN": "DEVELOPMENT"}):
            self.assertFalse(_is_apify_qa_run())

    def test_apply_qa_caps(self):
        inp = ActorInput(
            search_terms=["dev1", "dev2"],
            locations=["loc1", "loc2"],
            results_wanted=100,
            max_results_total=500,
            boards=["indeed", "linkedin"],
            enable_ai_scoring=True,
        )
        capped = _apply_qa_caps(inp)
        self.assertEqual(capped.effective_search_terms(), ["dev1"])
        self.assertEqual(capped.effective_locations(), ["loc1"])
        self.assertEqual(capped.results_wanted, 5)
        self.assertEqual(capped.max_results_total, 10)
        self.assertEqual(capped.boards, ["linkedin"])
        self.assertFalse(capped.enable_ai_scoring)

    @patch("my_actor.main.Actor")
    @patch("my_actor.main.scrape_with_metadata")
    async def test_main_no_results(self, mock_scrape, mock_actor):
        mock_actor.get_input = AsyncMock(return_value={"search_term": "python", "location": "Remote"})
        mock_actor.push_data = AsyncMock()
        mock_actor.on = MagicMock()
        mock_actor.log = MagicMock()
        mock_scrape.return_value = ScrapeOutcome(jobs=[], failed_boards=["indeed"])

        await main()

        mock_actor.push_data.assert_called_once()
        call_args = mock_actor.push_data.call_args[0][0]
        self.assertEqual(call_args["status"], "no_results")
        self.assertEqual(call_args["failed_boards"], ["indeed"])

    @patch("my_actor.main.Actor")
    @patch("my_actor.main.scrape_with_metadata")
    async def test_main_unscored_push(self, mock_scrape, mock_actor):
        mock_actor.get_input = AsyncMock(return_value={"search_term": "python", "location": "Remote", "enable_ai_scoring": False})
        mock_actor.push_data = AsyncMock()
        mock_actor.on = MagicMock()
        mock_actor.log = MagicMock()
        sample_jobs = [{"id": "1", "title": "SWE", "company": "Tech"}]
        mock_scrape.return_value = ScrapeOutcome(jobs=sample_jobs, failed_boards=[])

        await main()

        mock_actor.push_data.assert_called_once_with(sample_jobs, charged_event_name="job-result")

    @patch("my_actor.main.Actor")
    @patch("my_actor.main.scrape_with_metadata")
    async def test_main_laya_jev_zero_key_flow(self, mock_scrape, mock_actor):
        mock_actor.get_input = AsyncMock(return_value={
            "search_term": "python",
            "location": "Remote",
            "enable_ai_scoring": True,
            "resume_text": "Senior Python Backend Engineer with FastAPI, PostgreSQL, and Docker",
            "anthropic_api_key": "",
            "min_ai_score": 50,
            "min_jev_score": 40.0,
        })
        mock_actor.push_data = AsyncMock()
        mock_actor.on = MagicMock()
        mock_actor.log = MagicMock()

        sample_jobs = [
            {
                "id": "1",
                "title": "Senior Python Developer",
                "company": "TechCorp",
                "description": "Looking for Senior Python Developer with FastAPI and PostgreSQL.",
                "salary_min": 140000,
                "salary_max": 170000,
                "is_remote": True,
            },
            {
                "id": "2",
                "title": "Graphic Designer",
                "company": "DesignStudio",
                "description": "Need Photoshop, Illustrator, and Figma skills.",
                "is_remote": False,
            },
        ]
        mock_scrape.return_value = ScrapeOutcome(jobs=sample_jobs, failed_boards=[])

        await main()

        # Job 1 is strong match with high JEV, Job 2 should be filtered out by min_ai_score / min_jev_score
        self.assertEqual(mock_actor.push_data.call_count, 1)
        pushed_job = mock_actor.push_data.call_args[0][0]
        self.assertEqual(pushed_job["id"], "1")
        self.assertGreaterEqual(pushed_job["ai_score"], 50)
        self.assertIn("jev_score", pushed_job)
        self.assertIn("jev_tier", pushed_job)
        self.assertIn("p_offer", pushed_job)
        self.assertIn("expected_return", pushed_job)
        self.assertIn("market_value", pushed_job)

if __name__ == "__main__":
    unittest.main()
