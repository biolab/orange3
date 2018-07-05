import os
import pickle
from collections import defaultdict

from Orange.canvas import config
from .interactions import NewLinkAction


class Suggestions:
    """
    Handles sorting of quick menu items when dragging a link from a widget onto empty canvas.
    """
    def __init__(self):
        self.__frequencies_path = config.data_dir() + "/widget-use-frequency.p"
        self.__import_factor = 0.8  # every time you reopen orange, imported frequencies are reduced

        self.__scheme = None
        self.__direction = None
        self.link_frequencies = defaultdict(int)
        self.source_probability = defaultdict(lambda: defaultdict(float))
        self.sink_probability = defaultdict(lambda: defaultdict(float))

        if not self.load_link_frequency():
            self.default_link_frequency()

    def load_link_frequency(self):
        if not os.path.isfile(self.__frequencies_path):
            return False
        file = open(self.__frequencies_path, "rb")
        imported_freq = pickle.load(file)
        for k, v in imported_freq.items():
            imported_freq[k] = self.__import_factor * v

        self.link_frequencies = imported_freq
        self.overwrite_probabilities_with_frequencies()
        return True

    def default_link_frequency(self):
        self.link_frequencies[("File", "Data Table", NewLinkAction.FROM_SOURCE)] = 3
        self.overwrite_probabilities_with_frequencies()

    def overwrite_probabilities_with_frequencies(self):
        for link, count in self.link_frequencies.items():
            self.increment_probability(link[0], link[1], link[2], count)

    def new_link(self, link):
        # direction is none when a widget was not added+linked via quick menu
        if self.__direction is None:
            return

        source_id = link.source_node.description.name
        sink_id = link.sink_node.description.name

        link_key = (source_id, sink_id, self.__direction)
        self.link_frequencies[link_key] += 1

        self.increment_probability(source_id, sink_id, self.__direction, 1)
        self.write_link_frequency()

        self.__direction = None

    def increment_probability(self, source_id, sink_id, direction, factor):
        if direction == NewLinkAction.FROM_SOURCE:
            self.source_probability[source_id][sink_id] += factor
            self.sink_probability[sink_id][source_id] += factor * 0.5
        else:  # FROM_SINK
            self.source_probability[source_id][sink_id] += factor * 0.5
            self.sink_probability[sink_id][source_id] += factor

    def write_link_frequency(self):
        pickle.dump(self.link_frequencies, open(self.__frequencies_path, "wb"))

    def get_sink_suggestions(self, source_id):
        return self.source_probability[source_id]

    def get_source_suggestions(self, sink_id):
        return self.sink_probability[sink_id]

    def get_default_suggestions(self):
        return self.source_probability

    def set_scheme(self, scheme):
        self.__scheme = scheme
        scheme.onNewLink(self.new_link)

    def set_direction(self, direction):
        """
        When opening quick menu, before the widget is created, set the direction
        of creation (FROM_SINK, FROM_SOURCE).
        """
        self.__direction = direction
