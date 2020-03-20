from Orange.data import RowInstance
from Orange.misc.server_embedder import ServerEmbedderCommunicator


class ExampleServerEmbedder(ServerEmbedderCommunicator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content_type = 'image/jpeg'

    async def _encode_data_instance(self, data_instance: RowInstance) -> bytes:
        """
        This is just an implementation for test purposes. We just return
        a sample bytes which is id encoded to bytes.
        """
        number = int(data_instance.id)
        return number.to_bytes((number.bit_length() + 7) // 8, byteorder='big')
