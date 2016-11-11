#!/usr/bin/env python3
"""
Generate widget index for the webpage.
"""

import json
import os
from bs4 import BeautifulSoup


def build_widget_catalog(index_html, outfile, webdocprefix):
    print("building widget catalog...")
    base_dir = os.path.dirname(index_html)

    with open(index_html, 'r') as f:
        index = f.read()

    def get_icon_path(widget_documentation_html):
        with open(os.path.join(base_dir, widget_documentation_html), 'r') as f:
            widget_html = f.read()
        widget_soup = BeautifulSoup(widget_html, "html.parser")
        widget_icon = widget_soup.find("img")
        return os.path.normpath(os.path.join(base_dir, widget_icon.get('src')))

    soup = BeautifulSoup(index, "html.parser")
    div = soup.find("div", {"id": "widgets"})

    ret = []

    for li in div.find_all():
        if li.name == "h3":
            cat = li.text.strip("¶").strip()
            ret.append((cat, []))
        if li.name == "li":
            a = li.find("a")
            imglink = webdocprefix + get_icon_path(a.get("href"))
            doclink = webdocprefix + a.get('href')
            text = a.string
            ret[-1][1].append({"img": imglink, "doc": doclink, "text": text })

    with open(outfile, 'wt') as f:
        json.dump(ret, f, indent=1)
    print("done")


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('--url-prefix', dest="prefix",
                      help="prefix to prepend to all generated urls")
    parser.add_option('--input', dest="input",
                      help="path to documentation index.html file")
    parser.add_option('--output', dest="output",
                      help="path where widgets.json will be created")

    options, args = parser.parse_args()

    build_widget_catalog(options.input, options.output, options.prefix)



