import unittest

from lad_mcp_server.model_metadata import ModelMetadataError, parse_models_payload
from lad_mcp_server.token_budget import TokenBudget, TokenBudgetError


class TestModelMetadata(unittest.TestCase):
    def test_parse_minimal_model(self) -> None:
        payload = {
            "data": [
                {
                    "id": "moonshotai/kimi-k2-thinking",
                    "context_length": 100000,
                    "supported_parameters": ["max_tokens", "tools"],
                }
            ]
        }
        models = parse_models_payload(payload)
        self.assertIn("moonshotai/kimi-k2-thinking", models)
        meta = models["moonshotai/kimi-k2-thinking"]
        self.assertEqual(meta.context_length, 100000)
        self.assertTrue(meta.supports_tools())
        self.assertEqual(meta.effective_context_length(), 100000)

    def test_parse_missing_data_raises(self) -> None:
        with self.assertRaises(ModelMetadataError):
            parse_models_payload({"nope": []})


class TestTokenBudget(unittest.TestCase):
    def test_budget_validates(self) -> None:
        budget = TokenBudget(effective_context_length=20000, effective_output_budget=1000, overhead_tokens=2000)
        budget.validate()
        self.assertEqual(budget.input_budget_tokens, 17000)

    def test_budget_too_small(self) -> None:
        budget = TokenBudget(effective_context_length=1000, effective_output_budget=900, overhead_tokens=200)
        with self.assertRaises(TokenBudgetError):
            budget.validate()


if __name__ == "__main__":
    unittest.main()

