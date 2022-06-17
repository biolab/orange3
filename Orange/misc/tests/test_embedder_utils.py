import os
import unittest

from Orange.misc.utils.embedder_utils import get_proxies


class TestProxies(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_http = os.environ.get("http_proxy")
        self.previous_https = os.environ.get("https_proxy")
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)

    def tearDown(self) -> None:
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        if self.previous_http is not None:
            os.environ["http_proxy"] = self.previous_http
        if self.previous_https is not None:
            os.environ["https_proxy"] = self.previous_https

    def test_add_scheme(self):
        os.environ["http_proxy"] = "test1.com"
        os.environ["https_proxy"] = "test2.com"
        res = get_proxies()
        self.assertEqual("http://test1.com", res.get("http://"))
        self.assertEqual("http://test2.com", res.get("https://"))

        os.environ["http_proxy"] = "test1.com/path"
        os.environ["https_proxy"] = "test2.com/path"
        res = get_proxies()
        self.assertEqual("http://test1.com/path", res.get("http://"))
        self.assertEqual("http://test2.com/path", res.get("https://"))

        os.environ["http_proxy"] = "https://test1.com:123"
        os.environ["https_proxy"] = "https://test2.com:124"
        res = get_proxies()
        self.assertEqual("https://test1.com:123", res.get("http://"))
        self.assertEqual("https://test2.com:124", res.get("https://"))

    def test_both_urls(self):
        os.environ["http_proxy"] = "http://test1.com:123"
        os.environ["https_proxy"] = "https://test2.com:124"
        res = get_proxies()
        self.assertEqual("http://test1.com:123", res.get("http://"))
        self.assertEqual("https://test2.com:124", res.get("https://"))
        self.assertNotIn("all://", res)

    def test_http_only(self):
        os.environ["http_proxy"] = "http://test1.com:123"
        res = get_proxies()
        self.assertEqual("http://test1.com:123", res.get("http://"))
        self.assertNotIn("https://", res)

    def test_https_only(self):
        os.environ["https_proxy"] = "https://test1.com:123"
        res = get_proxies()
        self.assertEqual("https://test1.com:123", res.get("https://"))
        self.assertNotIn("http://", res)

    def test_none(self):
        """ When no variable is set return None """
        self.assertIsNone(get_proxies())


if __name__ == "__main__":
    unittest.main()
