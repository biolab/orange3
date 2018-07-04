import operator
from collections import defaultdict


class Suggestions:
    def __init__(self):
        self.__scheme = None

        self.link_frequencies = defaultdict(int)
        self.source_frequencies = defaultdict(lambda: defaultdict(int))
        self.sink_frequencies = defaultdict(lambda: defaultdict(int))

    def new_link(self, link):
        source_id = link.source_node.title
        sink_id = link.sink_node.title

        link_key = (source_id, sink_id)
        self.link_frequencies[link_key] += 1

        # optimize by making a 2d matrix of id (string) indices
        self.source_frequencies[source_id][sink_id] += 1
        self.sink_frequencies[sink_id][source_id] += 1

    def get_sink_suggestions(self, source_id):
        return self.source_frequencies[source_id]

    def get_source_suggestions(self, sink_id):
        return self.sink_frequencies[sink_id]

    def set_scheme(self, scheme):
        self.__scheme = scheme
        scheme.onNewLink(self.new_link)