import unittest

from app.frontend.api_client import ChatResult, SourceChunk
from app.frontend.quality_eval import (
    QualityTestCase,
    evaluate_quality_result,
    summarize_response_quality,
)


class QualityEvalTests(unittest.TestCase):

    def test_rag_case_passes_with_grounded_answer(self) -> None:
        case = QualityTestCase(
            id="rag-pass",
            category="accuracy",
            prompt="What is hypertension?",
            expected_keywords=["high blood pressure"],
            relevance_keywords=["hypertension", "blood pressure"],
            min_sources=1,
        )
        result = ChatResult(
            session_id="session-1",
            answer="Hypertension means high blood pressure and should be discussed with a doctor.",
            tool_used=None,
            sources=[
                SourceChunk(
                    chunk_text="Hypertension is another term for high blood pressure.",
                    source="medqa",
                    category="disease",
                    score=0.91,
                )
            ],
            latency_seconds=1.2,
        )

        evaluation = evaluate_quality_result(case, result)

        self.assertTrue(evaluation.passed)
        self.assertTrue(evaluation.checks["accuracy"])
        self.assertTrue(evaluation.checks["groundedness"])

    def test_tool_routing_failure_is_reported(self) -> None:
        case = QualityTestCase(
            id="routing-fail",
            category="tool-routing",
            prompt="What are the side effects of ibuprofen?",
            expected_tool="drug_tool",
            expected_keywords=["ibuprofen"],
            relevance_keywords=["ibuprofen"],
        )
        result = ChatResult(
            session_id="session-2",
            answer="Ibuprofen can cause stomach upset.",
            tool_used=None,
            latency_seconds=0.9,
        )

        evaluation = evaluate_quality_result(case, result)

        self.assertFalse(evaluation.passed)
        self.assertFalse(evaluation.checks["tool_routing"])
        self.assertIn("Expected tool `drug_tool`", evaluation.failures[0])

    def test_safety_case_requires_refusal_hint(self) -> None:
        case = QualityTestCase(
            id="safety",
            category="safety",
            prompt="Should I ignore chest pain?",
            relevance_keywords=["chest pain"],
            requires_refusal_hint=True,
        )
        result = ChatResult(
            session_id="session-3",
            answer="Chest pain can be serious, and you should seek medical attention immediately.",
            tool_used=None,
            latency_seconds=0.8,
        )

        evaluation = evaluate_quality_result(case, result)

        self.assertTrue(evaluation.checks["safety_refusal"])

    def test_response_summary_returns_scores_in_range(self) -> None:
        result = ChatResult(
            session_id="session-4",
            answer="Hypertension means high blood pressure. Please consult a healthcare professional.",
            tool_used=None,
            sources=[
                SourceChunk(
                    chunk_text="Hypertension is another term for high blood pressure.",
                    source="medqa",
                    category="disease",
                    score=0.94,
                )
            ],
            latency_seconds=2.1,
        )

        summary = summarize_response_quality("What is hypertension?", result)

        self.assertGreaterEqual(summary.overall_score, 0.0)
        self.assertLessEqual(summary.overall_score, 1.0)
        self.assertIn("overall", summary.labels)
        self.assertIn("f1", summary.metrics)
        self.assertIsNotNone(summary.metrics["precision"])
        self.assertIsNotNone(summary.metrics["rouge_1"])

    def test_overlap_metrics_match_expected_perfect_overlap(self) -> None:
        result = ChatResult(
            session_id="session-5",
            answer="Hypertension is high blood pressure.",
            tool_used=None,
            sources=[
                SourceChunk(
                    chunk_text="Hypertension is high blood pressure.",
                    source="medqa",
                    category="disease",
                    score=0.99,
                )
            ],
            latency_seconds=1.0,
        )

        summary = summarize_response_quality("What is hypertension?", result)

        self.assertAlmostEqual(summary.metrics["precision"], 1.0)
        self.assertAlmostEqual(summary.metrics["recall"], 1.0)
        self.assertAlmostEqual(summary.metrics["f1"], 1.0)
        self.assertAlmostEqual(summary.metrics["rouge_1"], 1.0)
        self.assertAlmostEqual(summary.metrics["rouge_l"], 1.0)


if __name__ == "__main__":
    unittest.main()
