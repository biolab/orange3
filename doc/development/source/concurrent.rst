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


.. autoclass:: FutureSetWatcher
    :show-inheritance:
    :members:
    :exclude-members:
        doneAt, finishedAt, cancelledAt, resultReadyAt, exceptionReadyAt,
        progressChanged, doneAll

    .. autoattribute:: doneAt(index: int, future: Future)

    .. autoattribute:: finishedAt(index: int, future: Future)

    .. autoattribute:: cancelledAt(index: int, future: Future)

    .. autoattribute:: resultReadyAt(index: int, result: Any)

    .. autoattribute:: exceptionReadyAt(index: int, exception: BaseException)

    .. autoattribute:: progressChanged(donecount: int, count: int)

    .. autoattribute:: doneAll()


.. autoclass:: methodinvoke
    :members:

