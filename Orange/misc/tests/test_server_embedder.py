import asyncio
import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np
from httpx import ReadTimeout

from Orange.data import Domain, StringVariable, Table
from Orange.misc.tests.example_embedder import ExampleServerEmbedder

_HTTPX_POST_METHOD = "httpx.AsyncClient.post"


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        # pylint: disable=useless-super-delegation
        return super().__call__(*args, **kwargs)


class DummyResponse:
    def __init__(self, content):
        self.content = content


def make_dummy_post(response, sleep=0):
    @staticmethod
    # pylint: disable=unused-argument
    async def dummy_post(url, headers, data):
        await asyncio.sleep(sleep)
        return DummyResponse(content=response)

    return dummy_post


regular_dummy_sr = make_dummy_post(b'{"embedding": [0, 1]}')


class TestServerEmbedder(unittest.TestCase):
    def setUp(self) -> None:
        self.embedder = ExampleServerEmbedder(
            "test", 10, "https://test.com", "image"
        )
        self.embedder.clear_cache()
        self.test_data = Table.from_numpy(
            Domain([], metas=[StringVariable("test_var")]),
            np.empty((3, 0)),
            metas=np.array([["test1"], ["test2"], ["test3"]]),
        )

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_responses(self):
        results = self.embedder.embedd_data(self.test_data)
        np.testing.assert_array_equal(results, [[0, 1]] * 3)
        # pylint: disable=protected-access
        self.assertEqual(3, len(self.embedder._cache._cache_dict))

    @patch(_HTTPX_POST_METHOD, make_dummy_post(b""))
    def test_responses_empty(self):
        results = self.embedder.embedd_data(self.test_data)
        self.assertListEqual([None] * 3, results)
        # pylint: disable=protected-access
        self.assertEqual(0, len(self.embedder._cache._cache_dict))

    @patch(_HTTPX_POST_METHOD, make_dummy_post(b"blabla"))
    def test_on_non_json_response(self):
        results = self.embedder.embedd_data(self.test_data)
        self.assertListEqual([None] * 3, results)
        # pylint: disable=protected-access
        self.assertEqual(0, len(self.embedder._cache._cache_dict))

    @patch(_HTTPX_POST_METHOD, make_dummy_post(b'{"wrong-key": [0, 1]}'))
    def test_on_json_wrong_key_response(self):
        results = self.embedder.embedd_data(self.test_data)
        self.assertListEqual([None] * 3, results)
        # pylint: disable=protected-access
        self.assertEqual(0, len(self.embedder._cache._cache_dict))

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_persistent_caching(self):
        # pylint: disable=protected-access
        self.assertEqual(len(self.embedder._cache._cache_dict), 0)
        self.embedder.embedd_data(self.test_data)
        self.assertEqual(len(self.embedder._cache._cache_dict), 3)

        self.embedder = ExampleServerEmbedder(
            "test", 10, "https://test.com", "image"
        )
        self.assertEqual(len(self.embedder._cache._cache_dict), 3)

        self.embedder.clear_cache()
        self.embedder = ExampleServerEmbedder(
            "test", 10, "https://test.com", "image"
        )
        self.assertEqual(len(self.embedder._cache._cache_dict), 0)

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_different_models_caches(self):
        # pylint: disable=protected-access
        self.embedder.clear_cache()
        self.assertEqual(len(self.embedder._cache._cache_dict), 0)
        self.embedder.embedd_data(self.test_data)
        self.assertEqual(len(self.embedder._cache._cache_dict), 3)

        self.embedder = ExampleServerEmbedder(
            "different_emb", 10, "https://test.com", "image"
        )
        self.assertEqual(len(self.embedder._cache._cache_dict), 0)

        self.embedder = ExampleServerEmbedder(
            "test", 10, "https://test.com", "image"
        )
        self.assertEqual(len(self.embedder._cache._cache_dict), 3)
        self.embedder.clear_cache()

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_too_many_examples_for_one_batch(self):
        test_data = Table.from_numpy(
            Domain([], metas=[StringVariable("test_var")]),
            np.empty((200, 0)),
            metas=np.array([[f"test{i}"] for i in range(200)]),
        )
        results = self.embedder.embedd_data(test_data)
        np.testing.assert_array_equal(results, [[0, 1]] * 200)
        # pylint: disable=protected-access
        self.assertEqual(200, len(self.embedder._cache._cache_dict))

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_embedding_cancelled(self):
        # pylint: disable=protected-access
        # test for the server embedders
        self.assertFalse(self.embedder._cancelled)
        self.embedder.set_cancelled()
        with self.assertRaises(Exception):
            self.embedder.embedd_data(self.test_data)

    @patch(_HTTPX_POST_METHOD, side_effect=OSError)
    def test_connection_error(self, _):
        for num_rows in range(1, 20):
            test_data = Table.from_numpy(
                Domain([], metas=[StringVariable("test_var")]),
                np.empty((num_rows, 0)),
                metas=np.array([[f"test{i}"] for i in range(num_rows)]),
            )
            with self.assertRaises(ConnectionError):
                self.embedder.embedd_data(test_data)
            self.setUp()  # to init new embedder

    @patch(_HTTPX_POST_METHOD, side_effect=ReadTimeout("", request=None))
    def test_read_error(self, _):
        for num_rows in range(1, 20):
            test_data = Table.from_numpy(
                Domain([], metas=[StringVariable("test_var")]),
                np.empty((num_rows, 0)),
                metas=np.array([[f"test{i}"] for i in range(num_rows)]),
            )
            with self.assertRaises(ConnectionError):
                self.embedder.embedd_data(test_data)
            self.setUp()  # to init new embedder

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_encode_data_instance(self):
        mocked_fun = self.embedder._encode_data_instance = AsyncMock(
            return_value=b"abc"
        )
        self.embedder.embedd_data(self.test_data)
        self.assertEqual(3, mocked_fun.call_count)
        mocked_fun.assert_has_calls(
            [call(item) for item in self.test_data], any_order=True
        )
