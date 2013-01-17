import sys
from contextlib import contextmanager


@contextmanager
def redirect_stdout(replacement=None):
    old = sys.stdout
    if replacement is not None:
        sys.stdout = replacement

    try:
        yield
    finally:
        sys.stdout = old


@contextmanager
def redirect_stderr(replacement=None):
    old = sys.stderr
    if replacement is not None:
        sys.stderr = replacement

    try:
        yield
    finally:
        sys.stderr = old


@contextmanager
def redirect_stdin(replacement=None):
    old = sys.stdin
    if replacement is not None:
        sys.stdin = replacement

    try:
        yield
    finally:
        sys.stdin = old
