"""Run: python -m unittest tests.test_desktop_client -v"""
import unittest
from unittest.mock import patch, Mock

from desktop.client import compare, health


class ClientTests(unittest.TestCase):
    @patch("desktop.client.requests.get")
    def test_health(self, get):
        reply = Mock()
        reply.json.return_value = {"status": "ok", "loaded_models": 2}
        get.return_value = reply
        self.assertEqual(health()["loaded_models"], 2)
        self.assertTrue(get.call_args.args[0].endswith("/health"))

    @patch("desktop.client.requests.post")
    def test_compare_image_field(self, post):
        pred = {"predicted_class": "unknown", "confidence": 0.6}
        reply = Mock()
        reply.json.return_value = {"model_a": pred, "model_b": pred, "agreement": True}
        post.return_value = reply
        result = compare(b"JPEG_TEST")
        self.assertTrue(result["agreement"])
        self.assertEqual(post.call_args.kwargs["files"]["image"][0], "face.jpg")
        self.assertEqual(post.call_args.kwargs["files"]["image"][1], b"JPEG_TEST")
        self.assertTrue(post.call_args.args[0].endswith("/predict/compare"))

    def test_empty_image(self):
        with self.assertRaises(ValueError):
            compare(b"")


if __name__ == "__main__":
    unittest.main()
