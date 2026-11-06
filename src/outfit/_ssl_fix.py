"""Undo pip-system-certs' global SSL patch, which breaks HTTPS for the openai client.

The shared venv has pip-system-certs, which at interpreter start replaces
``ssl.SSLContext`` with pip's vendored copy of truststore. openai's HTTP stack
(httpx2/httpcore2) uses its own truststore copy, and the two wrapping each other
recurse until RecursionError. httpx2 already verifies against the Windows system
certificates by itself, so removing the global patch loses nothing here.

This must run before openai/httpx2 is imported, hence the call in ``outfit/__init__``.
"""

import logging
import ssl

logger = logging.getLogger(__name__)


def undo_pip_system_certs() -> None:
    """Restore the stdlib ``ssl.SSLContext`` if pip's vendored truststore replaced it."""
    if not ssl.SSLContext.__module__.startswith("pip._vendor.truststore"):
        return
    from pip._vendor import truststore

    truststore.extract_from_ssl()
    logger.debug("Removed pip-system-certs' truststore patch from ssl")
