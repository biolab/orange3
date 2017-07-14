# Tests construct several MockWidget classes with different signals
# pylint: disable=function-redefined
# pylint: disable=unused-variable

# It is our constitutional right to use foo and bar as fake handler names
# pylint: disable=blacklisted-name

import unittest
from unittest.mock import patch, MagicMock

from Orange.canvas.registry.description import \
    Single, Multiple, Default, NonDefault, Explicit, Dynamic, InputSignal, \
    OutputSignal
from Orange.widgets.tests.base import GuiTest
from Orange.widgets.utils.signals import _Signal, Input, Output, \
    WidgetSignalsMixin
from Orange.widgets.widget import OWWidget


class SignalTest(unittest.TestCase):
    def test_get_flags(self):
        self.assertEqual(_Signal.get_flags(False, False, False, False),
                         Single | NonDefault)
        self.assertEqual(_Signal.get_flags(True, False, False, False),
                         Multiple | NonDefault)
        self.assertEqual(_Signal.get_flags(False, True, False, False),
                         Single | Default)
        self.assertEqual(_Signal.get_flags(False, False, True, False),
                         Single | NonDefault | Explicit)
        self.assertEqual(_Signal.get_flags(False, False, False, True),
                         Single | NonDefault | Dynamic)


class InputTest(unittest.TestCase):
    def test_init(self):
        with patch("Orange.widgets.utils.signals._Signal.get_flags",
                   return_value=42) as getflags:
            signal = Input("a name", int, "an id", "a doc", ["x"])
            self.assertEqual(signal.name, "a name")
            self.assertEqual(signal.type, int)
            self.assertEqual(signal.id, "an id")
            self.assertEqual(signal.doc, "a doc")
            self.assertEqual(signal.replaces, ["x"])
            self.assertEqual(signal.flags, 42)
            getflags.assert_called_with(False, False, False, False)

            Input("a name", int, "an id", "a doc", ["x"], multiple=True)
            getflags.assert_called_with(True, False, False, False)

            Input("a name", int, "an id", "a doc", ["x"], default=True)
            getflags.assert_called_with(False, True, False, False)

            Input("a name", int, "an id", "a doc", ["x"], explicit=True)
            getflags.assert_called_with(False, False, True, False)

    def test_decorate(self):
        input = Input("a name", int)
        self.assertEqual(input.handler, "")

        @input
        def foo():
            pass
        self.assertEqual(input.handler, "foo")

        with self.assertRaises(ValueError):
            @input
            def bar():
                pass


class OutputTest(unittest.TestCase):
    def test_init(self):
        with patch("Orange.widgets.utils.signals._Signal.get_flags",
                   return_value=42) as getflags:
            signal = Output("a name", int, "an id", "a doc", ["x"])
            self.assertEqual(signal.name, "a name")
            self.assertEqual(signal.type, int)
            self.assertEqual(signal.id, "an id")
            self.assertEqual(signal.doc, "a doc")
            self.assertEqual(signal.replaces, ["x"])
            self.assertEqual(signal.flags, 42)
            getflags.assert_called_with(False, False, False, True)

            Output("a name", int, "an id", "a doc", ["x"], default=True)
            getflags.assert_called_with(False, True, False, True)

            Output("a name", int, "an id", "a doc", ["x"], explicit=True)
            getflags.assert_called_with(False, False, True, True)

            Output("a name", int, "an id", "a doc", ["x"], dynamic=False)
            getflags.assert_called_with(False, False, False, False)

    def test_bind_and_send(self):
        widget = MagicMock()
        output = Output("a name", int, "an id", "a doc", ["x"])
        bound = output.bound_signal(widget)
        value = object()
        id = 42
        bound.send(value, id)
        widget.signalManager.send.assert_called_with(
            widget, "a name", value, id)


class WidgetSignalsMixinTest(GuiTest):
    def test_init_binds_outputs(self):
        class MockWidget(OWWidget):
            name = "foo"
            class Outputs:
                an_output = Output("a name", int)

        widget = MockWidget()
        self.assertEqual(widget.Outputs.an_output.widget, widget)
        self.assertIsNone(MockWidget.Outputs.an_output.widget)

    def test_checking_invalid_inputs(self):
        with self.assertRaises(ValueError):
            class MockWidget(OWWidget):
                name = "foo"

                class Inputs:
                    an_input = Input("a name", int)

        with self.assertRaises(ValueError):
            class MockWidget(OWWidget):
                name = "foo"
                inputs = [("a name", int, "no_such_handler")]

        # Now, don't crash
        class MockWidget(OWWidget):
            name = "foo"
            inputs = [("a name", int, "handler")]

            def handler(self):
                pass

    def test_signal_conversion(self):
        class MockWidget(OWWidget):
            name = "foo"
            inputs = [("name 1", int, "foo"), InputSignal("name 2", int, "foo")]
            outputs = [("name 3", int), OutputSignal("name 4", int)]

            def foo(self):
                pass

        input1, input2 = MockWidget.inputs
        self.assertIsInstance(input1, InputSignal)
        self.assertEqual(input1.name, "name 1")
        self.assertIsInstance(input2, InputSignal)
        self.assertEqual(input2.name, "name 2")

        output1, output2 = MockWidget.outputs
        self.assertIsInstance(output1, OutputSignal)
        self.assertEqual(output1.name, "name 3")
        self.assertIsInstance(output2, OutputSignal)
        self.assertEqual(output2.name, "name 4")

    def test_get_signals(self):
        class MockWidget(OWWidget):
            name = "foo"
            inputs = [("a name", int, "foo")]
            outputs = [("another name", float)]

            def foo(self):
                pass

        self.assertIs(MockWidget.get_signals("inputs"), MockWidget.inputs)
        self.assertIs(MockWidget.get_signals("outputs"), MockWidget.outputs)

        class MockWidget(OWWidget):
            name = "foo"

            class Inputs:
                an_input = Input("a name", int)

            class Outputs:
                an_output = Output("another name", int)

            @Inputs.an_input
            def foo(self):
                pass

        input, = MockWidget.get_signals("inputs")
        self.assertIsInstance(input, InputSignal)
        self.assertEqual(input.name, "a name")

        output, = MockWidget.get_signals("outputs")
        self.assertIsInstance(output, OutputSignal)
        self.assertEqual(output.name, "another name")

    def test_get_signals_order(self):
        class TestWidget(WidgetSignalsMixin):
            class Inputs:
                input_1 = Input("1", int)
                input_2 = Input("2", int)
                input_3 = Input("3", int)
                input_a = Input("a", object)

            class Outputs:
                output_1 = Output("1", int)
                output_2 = Output("2", int)
                output_3 = Output("3", int)
                output_a = Output("a", object)

        inputs = TestWidget.get_signals("inputs")
        self.assertTrue(all(isinstance(s, Input) for s in inputs))
        self.assertSequenceEqual([s.name for s in inputs], list("123a"))
        outputs = TestWidget.get_signals("outputs")
        self.assertTrue(all(isinstance(s, Output) for s in outputs))
        self.assertSequenceEqual([s.name for s in outputs], list("123a"))

if __name__ == "__main__":
    unittest.main()
