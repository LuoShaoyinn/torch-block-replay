"""Coordinate one-time CUDA capture with independent background GPU work."""
from contextlib import contextmanager
from functools import wraps
import threading

_condition = threading.Condition()
_active = 0
_capturing = False


@contextmanager
def background_cuda_work():
    """Allow concurrent workers, except while a graph is being recorded."""
    global _active
    with _condition:
        _condition.wait_for(lambda: not _capturing)
        _active += 1
    try:
        yield
    finally:
        with _condition:
            _active -= 1
            _condition.notify_all()


@contextmanager
def exclusive_cuda_capture():
    """Drain workers and keep their CUDA calls outside the capture interval."""
    global _capturing
    with _condition:
        _condition.wait_for(lambda: not _capturing)
        _capturing = True
        _condition.wait_for(lambda: _active == 0)
    try:
        yield
    finally:
        with _condition:
            _capturing = False
            _condition.notify_all()


def guarded_cuda_work(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        with background_cuda_work():
            return function(*args, **kwargs)
    return guarded
