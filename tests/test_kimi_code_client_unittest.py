import unittest
from lad_mcp_server.kimi_code_client import is_kimi_model, normalize_kimi_model_name

class TestKimiCodeClientHelpers(unittest.TestCase):
    def test_is_kimi_model_matches_expected(self) -> None:
        self.assertTrue(is_kimi_model("moonshotai/kimi-k2.5"))
        self.assertTrue(is_kimi_model("moonshotai/kimi-k2.6"))
        self.assertTrue(is_kimi_model("MOONSHOTAI/KIMI-K2.6"))
        self.assertTrue(is_kimi_model("moonshotai/other-model"))
        self.assertFalse(is_kimi_model("z-ai/glm-5"))
        self.assertFalse(is_kimi_model("other/model"))

    def test_normalize_model_name_returns_direct_model(self) -> None:
        # R4: moonshotai/kimi-k2.5 -> kimi-for-coding
        self.assertEqual(normalize_kimi_model_name("moonshotai/kimi-k2.5"), "kimi-for-coding")
        self.assertEqual(normalize_kimi_model_name("ANYTHING"), "kimi-for-coding")
        self.assertEqual(normalize_kimi_model_name(None), "kimi-for-coding")

if __name__ == "__main__":
    unittest.main()
