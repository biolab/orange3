.. currentmodule:: Orange.widgets.utils.concurrent

:mod:`Orange.widgets.utils.concurrent`
--------------------------------------

.. automodule:: Orange.widgets.utils.concurrent

.. autoclass:: ThreadExecutor
    :show-inheritance:
    :members:

.. autoclass:: FutureWatcher
    :show-inheritance:
    :members:
    :exclude-members:
        done, finished, cancelled, resultReady, exceptionReady

    .. autoattribute:: done(future: Future)

    .. autoattribute:: finished(future: Future)

    .. autoattribute:: cancelled(future: Future)

    .. autoattribute:: resultReady(result: Any)

    .. autoattribute:: exceptionReady(exception: BaseException)


.. autoclass:: methodinvoke
    :members:

