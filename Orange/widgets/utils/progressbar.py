import contextlib
import warnings

from orangewidget.utils.progressbar import (
    ProgressBarMixin as _ProgressBarMixin
)
from Orange.widgets import gui

__all__ = [
    "ProgressBarMixin"
]


def _warn_deprecated_arg():
    warnings.warn(
        "'processEvents' argument is deprecated.\n"
        "It does nothing and will be removed in the future (passing it "
        "will raise a TypeError).",
        FutureWarning, stacklevel=3,
    )


class ProgressBarMixin(_ProgressBarMixin):
    def progressBarInit(self, *args, **kwargs):
        if args or kwargs:
            _warn_deprecated_arg()
        super().progressBarInit()

    def progressBarSet(self, value, *args, **kwargs):
        if args or kwargs:
            _warn_deprecated_arg()
        super().progressBarSet(value)

    def progressBarAdvance(self, value, *args, **kwargs):
        if args or kwargs:
            _warn_deprecated_arg()
        super().progressBarAdvance(value)

    def progressBarFinished(self, *args, **kwargs):
        if args or kwargs:
            _warn_deprecated_arg()
        super().progressBarFinished()

    @contextlib.contextmanager
    def progressBar(self, iterations=0):
        """
        Context manager for progress bar.

        Using it ensures that the progress bar is removed at the end without
        needing the `finally` blocks.

        Usage:

            with self.progressBar(20) as progress:
                ...
                progress.advance()

        or

            with self.progressBar() as progress:
                ...
                progress.advance(0.15)

        or

            with self.progressBar():
                ...
                self.progressBarSet(50)

        :param iterations: the number of iterations (optional)
        :type iterations: int
        """
        progress_bar = gui.ProgressBar(self, iterations)
        try:
            yield progress_bar
        finally:
            progress_bar.finish()
