import os
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch

from Orange.misc.utils.embedder_utils import get_proxies, EmbedderCache


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


class TestEmbedderCache(unittest.TestCase):
    # pylint: disable=protected-access
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        patcher = patch(
            "Orange.misc.utils.embedder_utils.cache_dir", return_value=self.temp_dir
        )
        patcher.start()
        self.addCleanup(patch.stopall)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_save_load_cache(self):
        # open when cache file doesn't exist yet
        cache = EmbedderCache("TestModel")
        self.assertDictEqual({}, cache._cache_dict)

        # add values and save to test opening with existing file
        cache.add("abc", [1, 2, 3])
        cache.persist_cache()

        cache = EmbedderCache("TestModel")
        self.assertDictEqual({"abc": [1, 2, 3]}, cache._cache_dict)

    def test_save_cache_no_permission(self):
        # prepare a file
        cache = EmbedderCache("TestModel")
        self.assertDictEqual({}, cache._cache_dict)
        cache.add("abc", [1, 2, 3])
        cache.persist_cache()

        # set file to read-only and try to write
        curr_permission = os.stat(cache._cache_file_path).st_mode
        os.chmod(cache._cache_file_path, stat.S_IRUSR)
        cache.add("abcd", [1, 2, 3])
        # new values should be cached since file is readonly
        cache.persist_cache()
        cache = EmbedderCache("TestModel")
        self.assertDictEqual({"abc": [1, 2, 3]}, cache._cache_dict)
        os.chmod(cache._cache_file_path, curr_permission)

    def test_load_cache_no_permission(self):
        # prepare a file
        cache = EmbedderCache("TestModel")
        self.assertDictEqual({}, cache._cache_dict)
        cache.add("abc", [1, 2, 3])
        cache.persist_cache()

        # no read permission - load no cache
        if os.name == "nt":
            with patch(
                "Orange.misc.utils.embedder_utils.pickle.load",
                side_effect=PermissionError,
            ):
                # it is difficult to change write permission on Windows using
                # patch instead
                cache = EmbedderCache("TestModel")
        else:
            os.chmod(cache._cache_file_path, stat.S_IWUSR)
            cache = EmbedderCache("TestModel")
        self.assertDictEqual({}, cache._cache_dict)

    def test_load_cache_eof_error(self):
        # prepare a file
        cache = EmbedderCache("TestModel")
        self.assertDictEqual({}, cache._cache_dict)
        cache.add("abc", [1, 2, 3])
        cache.persist_cache()

        # eof error
        with patch(
            "Orange.misc.utils.embedder_utils.pickle.load", side_effect=EOFError,
        ):
            cache = EmbedderCache("TestModel")
            self.assertDictEqual({}, cache._cache_dict)


if __name__ == "__main__":
    unittest.main()
