"""Tests for `propertybindings`

"""
from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QCheckBox, QSpinBox, QLineEdit, QTextEdit
)
from AnyQt.QtCore import QObject

from ...gui import test

from ..propertybindings import (
    BindingManager, DynamicPropertyBinding, PropertyBindingExpr,
    PropertyBinding, binding_for
)


class Test(test.QAppTestCase):
    def test_dyn(self):
        obj = QObject()
        changed = []

        binding = DynamicPropertyBinding(obj, "test")
        binding.changed[object].connect(changed.append)

        self.assertIs(binding.get(), None)

        obj.setProperty("test", 1)
        self.assertEqual(binding.get(), 1)
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[-1], 1)

        binding.set(2)
        self.assertEqual(binding.get(), 2)
        self.assertEqual(len(changed), 2)
        self.assertEqual(changed[-1], 2)

        target = QObject()
        binding1 = DynamicPropertyBinding(target, "prop")

        binding1.bindTo(binding)

        self.assertEqual(binding1.get(), binding.get())
        self.assertEqual(target.property("prop"), 2)

        binding.set("a string")
        self.assertEqual(binding1.get(), "a string")
        self.assertEqual(binding1.get(), binding.get())
        self.assertEqual(target.property("prop"), "a string")

        binding1.unbind()
        binding.set(1)
        self.assertEqual(binding1.get(), "a string")
        self.assertEqual(target.property("prop"), "a string")
        self.assertEqual(binding.get(), 1)
        self.assertEqual(obj.property("test"), 1)

    def test_manager(self):
        source = QObject()
        target = QObject()

        manager = BindingManager(submitPolicy=BindingManager.ManualSubmit)

        manager.bind((target, "target"), None).to((source, "source"))

        tbind = DynamicPropertyBinding(target, "target_copy")
        sbind = DynamicPropertyBinding(source, "source")
        schanged = []
        sbind.changed[object].connect(schanged.append)

        manager.bind(tbind, None).to(sbind)

        source.setProperty("source", 1)
        self.assertEqual(len(schanged), 1)

        self.assertEqual(target.property("target"), None)

        manager.commit()

        self.assertEqual(target.property("target"), 1)
        self.assertEqual(target.property("target_copy"), 1)

        source.setProperty("source", 2)

        manager.setSubmitPolicy(BindingManager.AutoSubmit)

        self.assertEqual(target.property("target"), 2)
        self.assertEqual(target.property("target_copy"), 2)

    def test_prop(self):
        w = QWidget()
        layout = QVBoxLayout()
        cb = QCheckBox("Check", w)
        sp = QSpinBox(w)
        le = QLineEdit(w)
        textw = QTextEdit(w, readOnly=True)

        textw.setProperty("checked_", False)
        textw.setProperty("spin_", 0)
        textw.setProperty("line_", "")

        textexpr = PropertyBindingExpr(r"""
("Check box is {0}\n"
 "Spin has value {1}\n"
 "Line contains {2}").format(
    "checked" if checked else "unchecked",
    spin,
    line)
""",
            dict(checked=binding_for(cb, "checked"),
                 spin=binding_for(sp, "value"),
                 line=binding_for(le, "text")),
        )

        layout.addWidget(cb)
        layout.addWidget(sp)
        layout.addWidget(le)
        layout.addWidget(textw)

        manager = BindingManager(submitPolicy=BindingManager.AutoSubmit)

        manager.bind(PropertyBinding(textw, "plainText", "textChanged"),
                     textexpr)

        w.setLayout(layout)
        w.show()

        self.app.exec_()

    def test_expr(self):
        obj1 = QObject()

        obj1.setProperty("value", 1)
        obj1.setProperty("other", 2)

        result = DynamicPropertyBinding(obj1, "result")
        result.bindTo(
            PropertyBindingExpr(
                "value + other",
                locals={"value": binding_for(obj1, "value"),
                        "other": binding_for(obj1, "other")}
                )
            )

        expr = PropertyBindingExpr(
           "True if value < 3 else False",
           dict(value=DynamicPropertyBinding(obj1, "result"))
        )

        result_values = []
        result.changed[object].connect(result_values.append)

        expr_values = []
        expr.changed[object].connect(expr_values.append)

        self.assertEqual(result.get(), 3)
        self.assertEqual(expr.get(), False)

        obj1.setProperty("value", 0)
        self.assertEqual(result_values[-1], 2)
        self.assertEqual(expr_values[-1], True)

        self.assertEqual(result.get(), 2)
        self.assertEqual(expr.get(), True)

#    @test.unittest.skip("Not yet implemented")
#    def test_decl(self):
#
#        class MyObj(QCheckBox):
#            __metaclass__ = declarative
#
#            display_value = property_expr(
#                "'Checked' if checked else 'Unchecked'",
#                type=unicode
#            )
#
#            display_value_changed = display_value.changed
#
#            child_value = property_expr(
#                "'Child enabled' if my_child.enabled else 'Child disabled'",
#                type=unicode
#            )
#
#            child_value_changed = child_value.changed
#
#            parent_area = property_expr("parent.width * parent.height",
#                                        queued_update=True)
#
#            area = property_expr("width * height")
#
#            color = property_expr(
#                lambda: 'red' if area > parent_area else 'blue',
#                type=QColor
#            )
#
#            def _on_color_changed(self, color):
#                self.setStyleSheet("color: {0};".format(color.name()))
#
#            color.changed[QColor]("_on_color_changed()")
#
#            _width_binding = bind("width: parent.width")
#
#            _css_bind = bind(
#                "styleSheet = 'color: {0!s};'.format(color.name())"
#            )
#
#            def __init__(self, *args, **kwargs):
#                QCheckBox.__init__(self, *args, **kwargs)
#
#                child = QWidget(self, objectName="my-child")
#
#                child.setEnabled(False)
#
#                binding(self, "width").bindTo(from_exp("parent.width"))
#                bind(self, "width = parent.width")
