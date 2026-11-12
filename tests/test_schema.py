import unittest
from pydantic import ValidationError
from my_actor.schema import ActorInput, JobPost, SUPPORTED_BOARDS

class TestActorInput(unittest.TestCase):
    def test_default_values(self):
        inp = ActorInput()
        self.assertEqual(inp.search_term, "software engineer")
        self.assertEqual(inp.location, "Remote")
        self.assertEqual(inp.hours_old, 720)
        self.assertEqual(inp.results_wanted, 5)
        self.assertEqual(inp.max_results_total, 10)
        self.assertEqual(inp.boards, ["linkedin"])
        self.assertFalse(inp.enable_ai_scoring)
        self.assertEqual(inp.min_ai_score, 0)
        self.assertEqual(inp.country_indeed, "USA")

    def test_aliases(self):
        raw = {
            "posted_within_hours": 48,
            "results_wanted_per_board": 15,
            "max_total_results": 30,
            "job_boards": ["indeed", "linkedin"],
            "resume_or_key_skills": "Python, React",
            "minimum_ai_score": 75,
            "indeed_country": "UK",
        }
        inp = ActorInput.model_validate(raw)
        self.assertEqual(inp.hours_old, 48)
        self.assertEqual(inp.results_wanted, 15)
        self.assertEqual(inp.max_results_total, 30)
        self.assertEqual(inp.boards, ["indeed", "linkedin"])
        self.assertEqual(inp.resume_text, "Python, React")
        self.assertEqual(inp.min_ai_score, 75)
        self.assertEqual(inp.country_indeed, "UK")

    def test_effective_search_and_locations(self):
        # Case 1: single term & location
        inp1 = ActorInput(search_term="data engineer", location="New York")
        self.assertEqual(inp1.effective_search_terms(), ["data engineer"])
        self.assertEqual(inp1.effective_locations(), ["New York"])

        # Case 2: multiple terms & locations override single
        inp2 = ActorInput(
            search_terms=["devops", "sre"],
            locations=["Austin", "London"]
        )
        self.assertEqual(inp2.effective_search_terms(), ["devops", "sre"])
        self.assertEqual(inp2.effective_locations(), ["Austin", "London"])

    def test_extra_fields_ignored(self):
        inp = ActorInput.model_validate({"search_term": "dev", "location": "Remote", "url": "https://example.com", "custom_field": 123})
        self.assertEqual(inp.search_term, "dev")

    def test_empty_search_terms_raises(self):
        with self.assertRaises(ValidationError):
            ActorInput.model_validate({"search_term": "   ", "search_terms": []})

    def test_empty_locations_raises(self):
        with self.assertRaises(ValidationError):
            ActorInput.model_validate({"location": "   ", "locations": []})

    def test_unsupported_board_raises(self):
        with self.assertRaises(ValidationError):
            ActorInput.model_validate({"job_boards": ["unknown_board"]})

    def test_board_normalization_and_dedup(self):
        inp = ActorInput.model_validate({"job_boards": [" LinkedIn ", "LINKEDIN", "indeed "]})
        self.assertEqual(inp.boards, ["linkedin", "indeed"])

    def test_empty_boards_raises(self):
        with self.assertRaises(ValidationError):
            ActorInput.model_validate({"job_boards": []})

    def test_score_boundaries(self):
        with self.assertRaises(ValidationError):
            ActorInput(min_ai_score=-1)
        with self.assertRaises(ValidationError):
            ActorInput(min_ai_score=101)
        with self.assertRaises(ValidationError):
            ActorInput(min_jev_score=-1)
        with self.assertRaises(ValidationError):
            ActorInput(min_jev_score=101)

    def test_job_post_model(self):
        post = JobPost(
            id="123",
            title="Senior Dev",
            company="Acme Corp",
            location="Remote",
            salary_min=100000,
            salary_max=150000,
            salary_currency="USD",
            is_remote=True,
            job_url="https://example.com/job/123",
            source="linkedin",
            jev_score=85.5,
            jev_tier="HIGH_EV",
            p_offer=0.75,
            market_value=135000.0,
            expected_return=101000.0,
        )
        dump = post.model_dump()
        self.assertEqual(dump["id"], "123")
        self.assertEqual(dump["salary_min"], 100000)
        self.assertTrue(dump["is_remote"])
        self.assertEqual(dump["jev_score"], 85.5)
        self.assertEqual(dump["jev_tier"], "HIGH_EV")
        self.assertEqual(dump["p_offer"], 0.75)

if __name__ == "__main__":
    unittest.main()
