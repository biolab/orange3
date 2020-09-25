import asyncio
import json
import logging
import random
import uuid
from json import JSONDecodeError
from os import getenv
from typing import Any, Callable, List, Optional

from AnyQt.QtCore import QSettings
from httpx import AsyncClient, NetworkError, ReadTimeout, Response

from Orange.misc.utils.embedder_utils import (EmbedderCache,
                                              EmbeddingCancelledException,
                                              EmbeddingConnectionError,
                                              get_proxies)

log = logging.getLogger(__name__)


class ServerEmbedderCommunicator:
    """
    This class needs to be inherited by the class which re-implements
    _encode_data_instance and defines self.content_type. For sending a table
    with data items use embedd_table function. This one is called with the
    complete Orange data Table. Then _encode_data_instance needs to extract
    data to be embedded from the RowInstance. For images, it takes the image
    path from the table, load image, and transform it into bytes.

    Attributes
    ----------
    model_name
        The name of the model. Name is used in url to define what server model
        gets data to embedd and as a caching keyword.
    max_parallel_requests
        Number of image that can be sent to the server at the same time.
    server_url
        The base url of the server (without any additional subdomains)
    embedder_type
        The type of embedder (e.g. image). It is used as a part of url (e.g.
        when embedder_type is image url is api.garaza.io/image)
    """

    MAX_REPEATS = 3

    count_connection_errors = 0
    count_read_errors = 0
    max_errors = 10

    def __init__(
            self,
            model_name: str,
            max_parallel_requests: int,
            server_url: str,
            embedder_type: str,
    ) -> None:
        self.server_url = getenv("ORANGE_EMBEDDING_API_URL", server_url)
        self._model = model_name
        self.embedder_type = embedder_type

        # attribute that offers support for cancelling the embedding
        # if ran in another thread
        self._cancelled = False

        self.machine_id = None
        try:
            self.machine_id = QSettings().value(
                "error-reporting/machine-id", "", type=str
            ) or str(uuid.getnode())
        except TypeError:
            self.machine_id = str(uuid.getnode())
        self.session_id = str(random.randint(1, 1e10))

        self._cache = EmbedderCache(model_name)

        # default embedding timeouts are too small we need to increase them
        self.timeout = 180
        self.num_parallel_requests = 0
        self.max_parallel = max_parallel_requests
        self.content_type = None  # need to be set in a class inheriting

    def embedd_data(
            self,
            data: List[Any],
            processed_callback: Callable[[bool], None] = None,
    ) -> List[Optional[List[float]]]:
        """
        This function repeats calling embedding function until all items
        are embedded. It prevents skipped items due to network issues.
        The process is repeated for each item maximally MAX_REPEATS times.

        Parameters
        ----------
        data
            List with data that needs to be embedded.
        processed_callback
            A function that is called after each item is embedded
            by either getting a successful response from the server,
            getting the result from cache or skipping the item.

        Returns
        -------
        List of float list (embeddings) for successfully embedded
        items and Nones for skipped items.

        Raises
        ------
        EmbeddingConnectionError
            Error which indicate that the embedding is not possible due to
            connection error.
        EmbeddingCancelledException:
            If cancelled attribute is set to True (default=False).
        """
        # if there is less items than 10 connection error should be raised
        # earlier
        self.max_errors = min(len(data) * self.MAX_REPEATS, 10)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            embeddings = asyncio.get_event_loop().run_until_complete(
                self.embedd_batch(data, processed_callback)
            )
        except Exception:
            loop.close()
            raise

        loop.close()
        return embeddings

    async def embedd_batch(
            self, data: List[Any], proc_callback: Callable[[bool], None] = None
    ) -> List[Optional[List[float]]]:
        """
        Function perform embedding of a batch of data items.

        Parameters
        ----------
        data
            A list of data that must be embedded.
        proc_callback
            A function that is called after each item is fully processed
            by either getting a successful response from the server,
            getting the result from cache or skipping the item.

        Returns
        -------
        List of float list (embeddings) for successfully embedded
        items and Nones for skipped items.

        Raises
        ------
        EmbeddingCancelledException:
            If cancelled attribute is set to True (default=False).
        """
        requests = []
        async with AsyncClient(
                timeout=self.timeout, base_url=self.server_url, proxies=get_proxies()
        ) as client:
            for p in data:
                if self._cancelled:
                    raise EmbeddingCancelledException()
                requests.append(self._send_to_server(p, client, proc_callback))

            embeddings = await asyncio.gather(*requests)
        self._cache.persist_cache()
        assert self.num_parallel_requests == 0

        return embeddings

    async def __wait_until_released(self) -> None:
        while self.num_parallel_requests >= self.max_parallel:
            await asyncio.sleep(0.1)

    def __check_cancelled(self):
        if self._cancelled:
            raise EmbeddingCancelledException()

    async def _encode_data_instance(
            self, data_instance: Any
    ) -> Optional[bytes]:
        """
        The reimplementation of this function must implement the procedure
        to encode the data item in a string format that will be sent to the
        server. For images it is the byte string with an image. The encoding
        must be always equal for same data instance.

        Parameters
        ----------
        data_instance
            The row of an Orange data table.

        Returns
        -------
        Bytes encoding the data instance.
        """
        raise NotImplementedError

    async def _send_to_server(
            self,
            data_instance: Any,
            client: AsyncClient,
            proc_callback: Callable[[bool], None] = None,
    ) -> Optional[List[float]]:
        """
        Function get an data instance. It extract data from it and send them to
        server and retrieve responses.

        Parameters
        ----------
        data_instance
            Single row of the input table.
        client
            HTTPX client that communicates with the server
        proc_callback
            A function that is called after each item is fully processed
            by either getting a successful response from the server,
            getting the result from cache or skipping the item.

        Returns
        -------
        Embedding. For items that are not successfully embedded returns None.
        """
        await self.__wait_until_released()
        self.__check_cancelled()

        self.num_parallel_requests += 1
        # load bytes
        data_bytes = await self._encode_data_instance(data_instance)
        if data_bytes is None:
            self.num_parallel_requests -= 1
            return None

        # if data in cache return it
        cache_key = self._cache.md5_hash(data_bytes)
        emb = self._cache.get_cached_result_or_none(cache_key)

        if emb is None:
            # in case that embedding not sucessfull resend it to the server
            # maximally for MAX_REPEATS time
            for i in range(1, self.MAX_REPEATS + 1):
                self.__check_cancelled()
                url = (
                    f"/{self.embedder_type}/{self._model}?"
                    f"machine={self.machine_id}"
                    f"&session={self.session_id}&retry={i}"
                )
                emb = await self._send_request(client, data_bytes, url)
                if emb is not None:
                    self._cache.add(cache_key, emb)
                    break  # repeat only when embedding None
        if proc_callback:
            proc_callback(emb is not None)

        self.num_parallel_requests -= 1
        return emb

    async def _send_request(
            self, client: AsyncClient, data: bytes, url: str
    ) -> Optional[List[float]]:
        """
        This function sends a single request to the server.

        Parameters
        ----------
        client
            HTTPX client that communicates with the server
        data
            Data packed in the sequence of bytes.
        url
            Rest of the url string.

        Returns
        -------
        embedding
            Embedding. For items that are not successfully embedded returns
            None.
        """
        headers = {
            "Content-Type": self.content_type,
            "Content-Length": str(len(data)),
        }
        try:
            response = await client.post(url, headers=headers, data=data)
        except ReadTimeout as ex:
            log.debug("Read timeout", exc_info=True)
            # it happens when server do not respond in 60 seconds, in
            # this case we return None and items will be resend later

            # if it happens more than in ten consecutive cases it means
            # sth is wrong with embedder we stop embedding
            self.count_read_errors += 1

            if self.count_read_errors >= self.max_errors:
                self.num_parallel_requests = 0  # for safety reasons
                raise EmbeddingConnectionError from ex
            return None
        except (OSError, NetworkError) as ex:
            log.debug("Network error", exc_info=True)
            # it happens when no connection and items cannot be sent to the
            # server
            # we count number of consecutive errors
            # if more than 10 consecutive errors it means there is no
            # connection so we stop embedding with EmbeddingConnectionError
            self.count_connection_errors += 1
            if self.count_connection_errors >= self.max_errors:
                self.num_parallel_requests = 0  # for safety reasons
                raise EmbeddingConnectionError from ex
            return None
        except Exception:
            log.debug("Embedding error", exc_info=True)
            raise
        # we reset the counter at successful embedding
        self.count_connection_errors = 0
        self.count_read_errors = 0
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: Response) -> Optional[List[float]]:
        """
        This function get response and extract embeddings out of them.

        Parameters
        ----------
        response
            Response by the server

        Returns
        -------
        Embedding. For items that are not successfully embedded returns None.
        """
        if response.content:
            try:
                cont = json.loads(response.content.decode("utf-8"))
                return cont.get("embedding", None)
            except JSONDecodeError:
                # in case that embedding was not successful response is not
                # valid JSON
                return None
        else:
            return None

    def clear_cache(self):
        self._cache.clear_cache()

    def set_cancelled(self):
        self._cancelled = True
