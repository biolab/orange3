from datetime import datetime
import platform
import json
import logging
import os
import requests

from Orange.canvas import config
try:
    from Orange.version import full_version, release
except ImportError:
    full_version = '???'
    release = True


log = logging.getLogger(__name__)


statistics_path = os.path.join(config.data_dir(), "usage-statistics.json")
server_url = os.getenv('ORANGE_STATISTICS_API_URL', "https://orange.biolab.si/usage-statistics")


class UsageStatistics:
    """
    Tracks and sends upon close usage statistics when error-reporting/send-statistics is true.
    This option can be changed in the preferences menu.

    Data is anonymously collected by the Bioinformatics Laboratory, University of Ljubljana,
    Faculty of Computer and Information Science.

    Data tracked per canvas session:
        date,
        Orange version,
        operating system,
        node additions by type:
            widget name
            type of addition:
                quick menu
                toolbox click
                toolbox drag
                drag from other widget
            (if dragged from other widget, other widget name)
    """

    NodeAddClick = 0
    NodeAddDrag = 1
    NodeAddMenu = 2
    NodeAddExtendFromSink = 3
    NodeAddExtendFromSource = 4

    last_search_query = None

    def __init__(self):
        self.toolbox_clicks = []
        self.toolbox_drags = []
        self.quick_menu_actions = []
        self.widget_extensions = []
        self.__node_addition_type = None

    def log_node_added(self, widget_name, extended_widget=None):
        if not config.settings()["error-reporting/send-statistics"]:
            return

        if self.__node_addition_type == UsageStatistics.NodeAddMenu:

            self.quick_menu_actions.append({
                "Widget Name": widget_name,
                "Query": UsageStatistics.last_search_query,
            })

        elif self.__node_addition_type == UsageStatistics.NodeAddClick:

            self.toolbox_clicks.append({
                "Widget Name": widget_name,
            })

        elif self.__node_addition_type == UsageStatistics.NodeAddDrag:

            self.toolbox_drags.append({
                "Widget Name": widget_name,
            })

        elif self.__node_addition_type == UsageStatistics.NodeAddExtendFromSink:

            self.widget_extensions.append({
                "Widget Name": widget_name,
                "Extended Widget": extended_widget,
                "Direction": "FROM_SINK",
                "Query": UsageStatistics.last_search_query,
            })

        elif self.__node_addition_type == UsageStatistics.NodeAddExtendFromSource:

            self.widget_extensions.append({
                "Widget Name": widget_name,
                "Extended Widget": extended_widget,
                "Direction": "FROM_SOURCE",
                "Query": UsageStatistics.last_search_query,
            })

        else:
            log.warning("Invalid usage statistics state; "
                        "attempted to log node before setting node type.")

    def set_node_type(self, addition_type):
        self.__node_addition_type = addition_type

    def send_statistics(self):
        if release and config.settings()["error-reporting/send-statistics"]:
            if os.path.isfile(statistics_path):
                with open(statistics_path) as f:
                    data = json.load(f)
                try:
                    r = requests.post(server_url, files={'file': json.dumps(data)})
                    if r.status_code != 200:
                        log.warning("Error communicating with server while attempting to send "
                                    "usage statistics.")
                        return
                    # success - wipe statistics file
                    log.info("Usage statistics sent.")
                    with open(statistics_path, 'w') as f:
                        json.dump([], f)
                except (ConnectionError, requests.exceptions.RequestException):
                    log.warning("Connection error while attempting to send usage statistics.")

    def write_statistics(self):
        if not release:
            log.info("Not sending usage statistics (non-release version of Orange detected).")
            return

        if not config.settings()["error-reporting/send-statistics"]:
            log.info("Not sending usage statistics (preferences setting).")
            return

        statistics = {
            "Date": str(datetime.now().date()),
            "Orange Version": full_version,
            "Operating System": platform.system() + " " + platform.release(),
            "Session": {
                "Quick Menu Search": self.quick_menu_actions,
                "Toolbox Click": self.toolbox_clicks,
                "Toolbox Drag": self.toolbox_drags,
                "Widget Extension": self.widget_extensions
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


    @staticmethod
    def set_last_search_query(query):
        UsageStatistics.last_search_query = query
