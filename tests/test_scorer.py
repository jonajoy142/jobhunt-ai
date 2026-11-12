import json
import unittest
from unittest.mock import MagicMock
from my_actor.scorer import (
    score_job,
    extract_keywords,
    _score_value,
    _missing_skills,
    _keyword_list,
    evaluate_with_laya,
    calculate_jev,
    evaluate_job,
)

class TestScorer(unittest.TestCase):
    def test_parse_json_payload_with_codeblocks(self):
        from my_actor.scorer import _parse_json_payload
        raw = "```json\n{\"score\": 85, \"verdict\": \"apply_now\"}\n```"
        self.assertEqual(_parse_json_payload(raw), {"score": 85, "verdict": "apply_now"})
        raw2 = "```\n[\"Python\", \"AWS\"]\n```"
        self.assertEqual(_parse_json_payload(raw2), ["Python", "AWS"])

    def test_helpers(self):
        self.assertEqual(_score_value("85"), 85)
        self.assertEqual(_score_value(150), 100)
        self.assertEqual(_score_value(-10), 0)
        self.assertEqual(_score_value("invalid"), 0)

        self.assertEqual(_missing_skills(["Python", " Docker "]), ["Python", "Docker"])
        self.assertEqual(_missing_skills("not a list"), [])

        keywords = _keyword_list(["Python", "FastAPI", "Python", "SQL"] + [f"skill_{i}" for i in range(15)])
        self.assertEqual(len(keywords), 10)
        self.assertEqual(keywords[0], "Python")
        self.assertEqual(keywords[1], "FastAPI")

    def test_laya_evaluation_with_resume(self):
        job = {
            "title": "Senior Python Backend Engineer",
            "company": "Tech Innovations",
            "description": "We are seeking a Senior Python Engineer proficient in FastAPI, PostgreSQL, Docker, and Kubernetes.",
            "is_remote": True,
        }
        # Strong match
        resume_strong = "Senior Software Engineer with 8 years of Python, FastAPI, Docker, and PostgreSQL experience."
        decision_strong = evaluate_with_laya(job, resume_strong)
        self.assertIn(decision_strong.choice, ["apply_now", "consider"])
        self.assertGreaterEqual(decision_strong.score, 60)
        self.assertGreaterEqual(decision_strong.noul, 0.60)
        self.assertIn("kubernetes", decision_strong.missing_skills)
        self.assertIn("python", decision_strong.keywords)

        # Weak match
        resume_weak = "Junior Graphic Designer with Figma and Photoshop skills."
        decision_weak = evaluate_with_laya(job, resume_weak)
        self.assertEqual(decision_weak.choice, "skip")
        self.assertLess(decision_weak.score, 50)
        self.assertLess(decision_weak.noul, 0.50)

    def test_laya_evaluation_without_resume(self):
        job = {
            "title": "Staff DevOps Engineer",
            "company": "Cloud Corp",
            "description": "Looking for a Staff DevOps engineer with AWS, Terraform, and Kubernetes experience.",
            "is_remote": True,
        }
        decision = evaluate_with_laya(job, "")
        self.assertIn(decision.choice, ["apply_now", "consider"])
        self.assertGreaterEqual(decision.score, 50)
        self.assertTrue(len(decision.keywords) > 0)

    def test_calculate_jev(self):
        job = {
            "title": "Senior Software Engineer",
            "company": "Enterprise Tech",
            "salary_min": 140000,
            "salary_max": 180000,
            "is_remote": True,
        }
        from my_actor.scorer import LayaDecision
        decision = LayaDecision(
            choice="apply_now",
            score=90,
            noul=0.85,
            missing_skills=[],
            keywords=["python", "docker"],
        )
        jev_data = calculate_jev(job, decision, hours_old=24)
        self.assertGreater(jev_data["market_value"], 140000)
        self.assertGreaterEqual(jev_data["p_offer"], 0.70)
        self.assertGreater(jev_data["expected_return"], 50000)
        self.assertGreaterEqual(jev_data["jev_score"], 60.0)
        self.assertIn(jev_data["jev_tier"], ["HIGH_EV", "MEDIUM_EV"])

    def test_evaluate_job_master(self):
        job = {
            "title": "Lead Software Engineer",
            "company": "NextGen AI",
            "description": "Python, PyTorch, LLM, AWS, Docker.",
            "is_remote": True,
        }
        res = evaluate_job(job, resume_text="Lead Engineer with Python, PyTorch, and AWS")
        self.assertIn("jev_score", res)
        self.assertIn("jev_tier", res)
        self.assertIn("p_offer", res)
        self.assertIn("market_value", res)
        self.assertIn("expected_return", res)
        self.assertIn("ai_score", res)
        self.assertIn("ai_verdict", res)
        self.assertIn("keywords", res)

    def test_score_job_success(self):
        mock_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps({
            "score": 90,
            "verdict": "apply_now",
            "missing_skills": ["Kubernetes"]
        }))]
        mock_client.messages.create.return_value = mock_msg

        job = {"title": "Backend Dev", "company": "TechCorp", "description": "Need Python and AWS"}
        res = score_job(mock_client, job, "Experienced Python engineer")
        self.assertEqual(res["ai_score"], 90)
        self.assertEqual(res["ai_verdict"], "apply_now")
        self.assertEqual(res["ai_missing_skills"], ["Kubernetes"])

    def test_score_job_failure_fallback(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API rate limit or connection error")

        job = {"title": "Backend Dev", "company": "TechCorp", "description": "Need Python and AWS"}
        res = score_job(mock_client, job, "Experienced Python engineer")
        self.assertEqual(res["ai_score"], 0)
        self.assertEqual(res["ai_verdict"], "error")
        self.assertEqual(res["ai_missing_skills"], [])

    def test_extract_keywords_success(self):
        mock_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps(["Python", "FastAPI", "PostgreSQL"]))]
        mock_client.messages.create.return_value = mock_msg

        job = {"title": "Backend Dev", "company": "TechCorp", "description": "Need Python and AWS"}
        keywords = extract_keywords(mock_client, job)
        self.assertEqual(keywords, ["Python", "FastAPI", "PostgreSQL"])

    def test_extract_keywords_failure(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")

        job = {"title": "Backend Dev", "company": "TechCorp", "description": "Need Python and AWS"}
        keywords = extract_keywords(mock_client, job)
        self.assertEqual(keywords, [])

if __name__ == "__main__":
    unittest.main()
