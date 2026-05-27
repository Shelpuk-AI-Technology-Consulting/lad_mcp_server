import unittest

from lad_mcp_server.zai_coding_client import is_zai_model, normalize_zai_model_name


class TestZaiCodingClientHelpers(unittest.TestCase):
    def test_is_zai_model_matches_expected_prefixes(self) -> None:
        self.assertTrue(is_zai_model("z-ai/glm-5"))
        self.assertTrue(is_zai_model("zai/glm-5"))
        self.assertTrue(is_zai_model("Z-AI/glm-5"))
        self.assertFalse(is_zai_model("moonshotai/kimi-k2.5"))
        self.assertFalse(is_zai_model("foo-zai/glm-5"))

    def test_normalize_model_name_strips_prefix(self) -> None:
        self.assertEqual(normalize_zai_model_name("z-ai/glm-5"), "glm-5")
        self.assertEqual(normalize_zai_model_name("zai/glm-5"), "glm-5")
        self.assertEqual(normalize_zai_model_name("Z-AI/glm-4.7"), "glm-4.7")
        self.assertEqual(normalize_zai_model_name("moonshotai/kimi-k2.5"), "moonshotai/kimi-k2.5")


if __name__ == "__main__":
    unittest.main()
