from __future__ import annotations

import logging

from ._version import __version__
from .session import SoundSession
from .chat import ChatManager, SignalChatManager

_FORMAT = '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s %(message)s'

logger = logging.getLogger('soundstream')
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt='%H:%M:%S'))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)


def enable_debug() -> None:
    logger.setLevel(logging.DEBUG)
    for h in logger.handlers:
        h.setLevel(logging.DEBUG)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        root_handler = logging.StreamHandler()
        root_handler.setFormatter(logging.Formatter(_FORMAT, datefmt='%H:%M:%S'))
        root_handler.setLevel(logging.DEBUG)
        root.addHandler(root_handler)
    else:
        for h in root.handlers:
            h.setLevel(logging.DEBUG)

    logging.getLogger('aiortc').setLevel(logging.DEBUG)
    logging.getLogger('aioice').setLevel(logging.DEBUG)
    logging.getLogger('aiohttp').setLevel(logging.DEBUG)

__all__ = ['SoundSession', 'ChatManager', 'SignalChatManager', '__version__', 'logger']
