import os
import pickle
from collections import defaultdict
import logging

from Orange.canvas import config
from .interactions import NewLinkAction

log = logging.getLogger(__name__)


class Suggestions:
    """
    Handles sorting of quick menu items when dragging a link from a widget onto empty canvas.
    """
    class __Suggestions:
        def __init__(self):
            self.__frequencies_path = os.path.join(config.data_dir(), "widget-use-frequency.pickle")
            self.__import_factor = 0.8  # upon starting Orange, imported frequencies are reduced

            self.__scheme = None
            self.__direction = None
            self.__link_frequencies = defaultdict(int)
            self.__source_probability = defaultdict(lambda: defaultdict(float))
            self.__sink_probability = defaultdict(lambda: defaultdict(float))

            if not self.__load_link_frequency():
                self.reset_suggestions()

        def __load_link_frequency(self):
            if not os.path.isfile(self.__frequencies_path):
                return False

            try:
                with open(self.__frequencies_path, "rb") as f:
                    imported_freq = pickle.load(f)
            except OSError:
                log.warning("Failed to open widget link frequencies.")
                return False

            for k, v in imported_freq.items():
                imported_freq[k] = self.__import_factor * v

            self.__link_frequencies = imported_freq
            self.__write_frequencies_into_probabilities()
            return True

        def reset_suggestions(self):
            self.__link_frequencies[("File", "Data Table", NewLinkAction.FROM_SOURCE)] = 3
            self.__write_frequencies_into_probabilities()

        def __write_frequencies_into_probabilities(self):
            for link, count in self.__link_frequencies.items():
                self.__increment_probability(link[0], link[1], link[2], count)

        def log_new_link(self, link):
            # direction is none when a widget was not added+linked via quick menu
            if self.__direction is None:
                return

            source_id = link.source_node.description.name
            sink_id = link.sink_node.description.name

            link_key = (source_id, sink_id, self.__direction)
            self.__link_frequencies[link_key] += 1

            self.__increment_probability(source_id, sink_id, self.__direction, 1)
            self.__save_link_frequency()

            self.__direction = None

        def __increment_probability(self, source_id, sink_id, direction, factor):
            if direction == NewLinkAction.FROM_SOURCE:
                self.__source_probability[source_id][sink_id] += factor
                self.__sink_probability[sink_id][source_id] += factor * 0.5
            else:  # FROM_SINK
                self.__source_probability[source_id][sink_id] += factor * 0.5
                self.__sink_probability[sink_id][source_id] += factor

        def __save_link_frequency(self):
            try:
                with open(self.__frequencies_path, "wb") as f:
                    pickle.dump(self.__link_frequencies, f)
            except OSError:
                log.warning("Failed to write widget link frequencies.")
                return

        def set_direction(self, direction):
            """
            When opening quick menu, before the widget is created, set the direction
            of creation (FROM_SINK, FROM_SOURCE).
            """
            self.__direction = direction

        def set_scheme(self, scheme):
            self.__scheme = scheme
            scheme.onNewLink(self.log_new_link)

        def get_sink_suggestions(self, source_id):
            return self.__source_probability[source_id]

        def get_source_suggestions(self, sink_id):
            return self.__sink_probability[sink_id]

        def get_default_suggestions(self):
            return self.__source_probability

    instance = None

    def __init__(self):
        if not Suggestions.instance:
            Suggestions.instance = Suggestions.__Suggestions()

    def __getattr__(self, name):
        return getattr(self.instance, name)
