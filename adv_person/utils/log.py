import logging
import sys
from typing import Optional, TextIO


def setup_logger(
    name: Optional[str] = None,
    filename: Optional[str] = None,
    stream: Optional[TextIO] = sys.stderr,
    timestamp=False,
    **kwargs,
):
    """To setup as many loggers as you want.

    Args:
        name: logger name
        filename: log file name, set None to disable
        stream: log to stream, set None to disable
        timestamp: add timestamp to log output
        **kwargs: any other keyword arguments to be passed into `logging.basicConfig()`
    """
    if timestamp:
        # Default format and date format with timestamp
        kwargs.setdefault("format", "[%(asctime)s] - %(message)s")
        kwargs.setdefault("datefmt", "%Y/%m/%d %H:%M:%S")
    else:
        kwargs.setdefault("format", "%(message)s")
    default_handlers: list[logging.Handler] = []
    if stream is not None:
        default_handlers.append(logging.StreamHandler(stream))
    if filename is not None:
        default_handlers.append(logging.FileHandler(filename))
    kwargs.setdefault("handlers", default_handlers)
    assert len(kwargs["handlers"]), (
        "Some handlers should be specified by 'stream', 'filename' or 'handlers'"
    )
    logger = logging.getLogger(name)
    logging.basicConfig(force=True, **kwargs)
    return logger
