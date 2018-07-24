import json
import os
from datetime import datetime

from Orange.canvas import config


class UsageStatistics:

    NodeAddClick = 0
    NodeAddDrag = 1
    NodeAddMenu = 2

    last_search_query = None

    def __init__(self):
        self.__statistics_path = os.path.join(config.data_dir(), "usage-statistics.json")
        self.start_time = datetime.now()

        self.toolbox_clicks = []
        self.toolbox_drags = []
        self.quick_menu_actions = []
        self.__node_addition_type = None

    def log_node_added(self, widget_name):
        time = str(datetime.now() - self.start_time)

        if self.__node_addition_type == UsageStatistics.NodeAddMenu:

            self.quick_menu_actions.append({
                "Widget Name": widget_name,
                "Query": UsageStatistics.last_search_query,
                "Time": time
            })

        elif self.__node_addition_type == UsageStatistics.NodeAddClick:

            self.toolbox_clicks.append({
                "Widget Name": widget_name,
                "Time": time
            })

        else:  # NodeAddDrag

            self.toolbox_drags.append({
                "Widget Name": widget_name,
                "Time": time
            })

    def set_node_type(self, addition_type):
        self.__node_addition_type = addition_type

    def write_statistics(self):
        statistics = {
            "Date": str(datetime.now().date()),
            "Session": {
                "Quick Menu Search": self.quick_menu_actions,
                "Toolbox Click": self.toolbox_clicks,
                "Toolbox Drag": self.toolbox_drags
            }
        }

        if os.path.isfile(self.__statistics_path):
            with open(self.__statistics_path) as f:
                data = json.load(f)
        else:
            data = json.loads("[]")

        data.append(statistics)

        with open(self.__statistics_path, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def set_last_search_query(query):
        UsageStatistics.last_search_query = query
