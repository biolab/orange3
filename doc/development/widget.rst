.. currentmodule:: Orange.widgets.widget

Orange Widget
#############

OWWidget
--------

.. autoclass:: Orange.widgets.widget.OWWidget
   :members:
   :member-order: bysource


Input signal definitions
------------------------

Input/Output flags:

.. attribute:: Default

   This input is the default for it's type.
   When there are multiple IO signals with the same type the
   one with the default flag takes precedence when adding a new
   link in the canvas.

.. attribute:: Multiple

   Multiple signal (more then one input on the channel). Input with this
   flag receive a second parameter `id`

.. attribute:: Dynamic

   Only applies to output. Specifies that the instances on the output
   will in general be subtypes of the declared type and that the output 
   can be connected to any input signal which can accept a subtype of
   the declared output type.

.. autoclass:: InputSignal

.. autoclass:: OutputSignal
