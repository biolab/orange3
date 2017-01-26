.. currentmodule:: Orange.widgets.tests.base

Testing Widgets
===============

Writing widget tests may seem a daunting task at first, as there are so many
different functionalities a widget depends on while it operates. But as long
as you keep in mind that widgets are independent of each other, you should
be fine.

Whenever you feel that you need to construct a workflow to test the widget,
stop, and think again. The only way widgets communicate with each other is
through signals. If you need to construct a workflow to prepare the data and
send it to the widget, prepare the data in code and pass it in. Same goes for
outputs, get the value of the widget output and check that the assumptions
about it hold.

In order to make your life easier, Orange provides a base class for unittest
:class:`WidgetTest`, which provides some helper methods.


Class Member Documentation
--------------------------

.. autoclass:: WidgetTest
   :members:
   :member-order: bysource
