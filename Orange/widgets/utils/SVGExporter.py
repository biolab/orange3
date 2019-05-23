"""
A modified pyqtgraph/exporters/SVGExporter.py from
https://github.com/pyqtgraph/pyqtgraph/ (available under MIT License)

Fixes:
- SVGExporter: fix axes position and scale
  (https://github.com/pyqtgraph/pyqtgraph/pull/641)
- SVG export: handle Qt.NoPen on Qt5
  (https://github.com/pyqtgraph/pyqtgraph/pull/642)

Remove this file when pyqtgraph updates.
"""

from orangewidget.utils.SVGExporter import SVGExporter

__all__ = ["SVGExporter"]
