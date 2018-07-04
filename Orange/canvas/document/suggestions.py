import operator
from collections import defaultdict


class Suggestions:
    def __init__(self, scheme):
        self.__scheme = scheme

        self.link_frequencies = defaultdict(int)
        self.source_frequencies = defaultdict(lambda: defaultdict(int))
        self.sink_frequencies = defaultdict(lambda: defaultdict(int))

        self.__scheme.onNewLink(self.new_link)

    def new_link(self, link):
        source_id = link.source_node.description.id
        sink_id = link.sink_node.description.id

        link_key = (source_id, sink_id)
        self.link_frequencies[link_key] += 1

        # optimize by making a 2d matrix of id (string) indices
        self.source_frequencies[source_id][sink_id] += 1
        self.sink_frequencies[sink_id][source_id] += 1

        print(self.get_sink_suggestions(source_id))

    def get_sink_suggestions(self, source_id):
        return sorted(self.source_frequencies[source_id].items(), key=operator.itemgetter(1)).reverse()

    def get_source_suggestions(self, sink_id):
        return sorted(self.sink_frequencies[sink_id].items(), key=operator.itemgetter(1)).reverse()
