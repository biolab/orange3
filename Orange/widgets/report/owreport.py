from orangewidget.report.owreport import OWReport, HAVE_REPORT

# back-compatibility for deserialization
from orangewidget.report.owreport import ReportItem  # pylint: disable=unused-import

__all__ = [
    "OWReport", "HAVE_REPORT"
]

if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication
    from Orange.data import Table
    from Orange.widgets.data.owfile import OWFile
    from Orange.widgets.data.owtable import OWDataTable
    from Orange.widgets.data.owdiscretize import OWDiscretize
    from Orange.widgets.model.owrandomforest import OWRandomForest

    iris = Table("iris")
    app = QApplication(sys.argv)

    main = OWReport.get_instance()
    file = OWFile()
    file.create_report_html()
    main.make_report(file)

    table = OWDataTable()
    table.set_dataset(iris)
    table.create_report_html()
    main.make_report(table)

    main = OWReport.get_instance()
    disc = OWDiscretize()
    disc.create_report_html()
    main.make_report(disc)

    learner = OWRandomForest()
    learner.create_report_html()
    main.make_report(learner)

    main.show()
    main.saveSettings()
    assert main.table_model.rowCount() == 4

    sys.exit(app.exec())
