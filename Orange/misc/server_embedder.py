import asyncio
import json
import logging
import random
import uuid
from collections import namedtuple
from json import JSONDecodeError
from os import getenv
from typing import Any, Callable, List, Optional

from AnyQt.QtCore import QSettings
from httpx import AsyncClient, NetworkError, ReadTimeout, Response

from Orange.misc.utils.embedder_utils import (
    EmbedderCache,
    EmbeddingCancelledException,
    EmbeddingConnectionError,
    get_proxies,
)

log = logging.getLogger(__name__)
TaskItem = namedtuple("TaskItem", ("id", "item", "no_repeats"))


class ServerEmbedderCommunicator:
    """
    This class needs to be inherited by the class which re-implements
    _encode_data_instance and defines self.content_type. For sending a list
    with data items use embedd_table function.

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
        self.session_id = str(random.randint(1, int(1e10)))

        self._cache = EmbedderCache(model_name)

        # default embedding timeouts are too small we need to increase them
        self.timeout = 180
        self.max_parallel_requests = max_parallel_requests

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
        # if there is less items than 10 connection error should be raised earlier
        self.max_errors = min(len(data) * self.MAX_REPEATS, 10)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            embeddings = asyncio.get_event_loop().run_until_complete(
                self.embedd_batch(data, processed_callback)
            )
        finally:
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
        results = [None] * len(data)
        queue = asyncio.Queue()

        # fill the queue with items to embedd
        for i, item in enumerate(data):
            queue.put_nowait(TaskItem(id=i, item=item, no_repeats=0))

        async with AsyncClient(
            timeout=self.timeout, base_url=self.server_url, proxies=get_proxies()
        ) as client:
            tasks = self._init_workers(client, queue, results, proc_callback)

            # wait for the queue to complete or one of workers to exit
            queue_complete = asyncio.create_task(queue.join())
            await asyncio.wait(
                [queue_complete, *tasks], return_when=asyncio.FIRST_COMPLETED
            )

        # Cancel worker tasks when done
        queue_complete.cancel()
        await self._cancel_workers(tasks)

        self._cache.persist_cache()
        return results

    def _init_workers(self, client, queue, results, callback):
        """Init required number of workers"""
        t = [
            asyncio.create_task(self._send_to_server(client, queue, results, callback))
            for _ in range(self.max_parallel_requests)
        ]
        log.debug("Created %d workers", self.max_parallel_requests)
        return t

    @staticmethod
    async def _cancel_workers(tasks):
        """Cancel worker at the end"""
        log.debug("Canceling workers")
        try:
            # try to catch any potential exceptions
            await asyncio.gather(*tasks)
        except Exception as ex:
            # raise exceptions gathered from an failed worker
            raise ex
        finally:
            # cancel all tasks in both cases
            for task in tasks:
                task.cancel()
            # Wait until all worker tasks are cancelled.
            await asyncio.gather(*tasks, return_exceptions=True)
            log.debug("All workers canceled")

    def __check_cancelled(self):
        if self._cancelled:
            raise EmbeddingCancelledException()

    async def _encode_data_instance(self, data_instance: Any) -> Optional[bytes]:
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
        client: AsyncClient,
        queue: asyncio.Queue,
        results: List,
        proc_callback: Callable[[bool], None] = None,
    ):
        """
        Worker that embedds data. It is pulling items from the until the queue
        is empty. It is canceled by embedd_batch all tasks are finished

        Parameters
        ----------
        client
            HTTPX client that communicates with the server
        queue
            The queue with items of type TaskItem to be embedded
        results
            The list to append results in. The list has length equal to numbers
            of all items to embedd. The result need to be inserted at the index
            defined in queue items.
        proc_callback
            A function that is called after each item is fully processed
            by either getting a successful response from the server,
            getting the result from cache or skipping the item.
        """
        while not queue.empty():
            self.__check_cancelled()

            # get item from the queue
            i, data_instance, num_repeats = await queue.get()
            num_repeats += 1

            # load bytes
            data_bytes = await self._encode_data_instance(data_instance)
            if data_bytes is None:
                continue

            # retrieve embedded item from the local cache
            cache_key = self._cache.md5_hash(data_bytes)
            log.debug("Embedding %s", cache_key)
            emb = self._cache.get_cached_result_or_none(cache_key)

            if emb is None:
                # send the item to the server for embedding if not in the local cache
                log.debug("Sending to the server: %s", cache_key)
                url = (
                    f"/{self.embedder_type}/{self._model}?machine={self.machine_id}"
                    f"&session={self.session_id}&retry={num_repeats}"
                )
                emb = await self._send_request(client, data_bytes, url)
                if emb is not None:
                    self._cache.add(cache_key, emb)

            if emb is not None:
                # store result if embedding is successful
                log.debug("Successfully embedded:  %s", cache_key)
                results[i] = emb
                if proc_callback:
                    proc_callback(emb is not None)
            elif num_repeats < self.MAX_REPEATS:
                log.debug("Embedding unsuccessful - reading to queue:  %s", cache_key)
                # if embedding not successful put the item to queue to be handled at
                # the end - the item is put to the end since it is possible that  server
                # still process the request and the result will be in the cache later
                # repeating the request immediately may result in another fail when
                # processing takes longer
                queue.put_nowait(TaskItem(i, data_instance, no_repeats=num_repeats))
            queue.task_done()

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
            # it happens when server do not respond in time defined by timeout
            # return None and items will be resend later

            # if it happens more than in ten consecutive cases it means
            # sth is wrong with embedder we stop embedding
            self.count_read_errors += 1
            if self.count_read_errors >= self.max_errors:
                raise EmbeddingConnectionError from ex
            return None
        except (OSError, NetworkError) as ex:
            log.debug("Network error", exc_info=True)
            # it happens when no connection and items cannot be sent to server

            # if more than 10 consecutive errors it means there is no
            # connection so we stop embedding with EmbeddingConnectionError
            self.count_connection_errors += 1
            if self.count_connection_errors >= self.max_errors:
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
