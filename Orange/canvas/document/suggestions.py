from collections import defaultdict


class Suggestions:
    def __init__(self, scheme):
        self.__scheme = scheme
        self.link_frequencies = defaultdict(int)
        self.__scheme.onNewLink(self.new_link)

    def new_link(self, link):
        link_key = (link.source_node.description.id, link.sink_node.description.id)
        self.link_frequencies[link_key] += 1
