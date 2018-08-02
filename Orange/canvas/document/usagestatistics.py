from datetime import datetime
import json
import logging
import os

from Orange.canvas import config
import requests

log = logging.getLogger(__name__)

statistics_path = os.path.join(config.data_dir(), "usage-statistics.json")
server_url = os.getenv('ORANGE_STATISTICS_API_URL', "https://orange.biolab.si/usage-statistics")


class UsageStatistics:

    NodeAddClick = 0
    NodeAddDrag = 1
    NodeAddMenu = 2

    last_search_query = None

    def __init__(self):
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

        if os.path.isfile(statistics_path):
            with open(statistics_path) as f:
                data = json.load(f)
        else:
            data = []

        data.append(statistics)

        with open(statistics_path, 'w') as f:
            json.dump(data, f)

        send_usage_statistics()

    @staticmethod
    def set_last_search_query(query):
        UsageStatistics.last_search_query = query


def send_usage_statistics():
    try:
        with open(statistics_path) as f:
            requests.post(server_url, files={'file': f}, verify=False)
    except ConnectionError:
        log.warning("Connection error while attempting to send usage statistics.")
