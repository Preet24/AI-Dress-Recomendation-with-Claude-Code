import ssl

import outfit  # noqa: F401  (importing the package applies the fix)
from outfit._ssl_fix import undo_pip_system_certs


def test_stdlib_ssl_context_is_restored_after_importing_outfit():
    assert ssl.SSLContext.__module__ == "ssl"


def test_fix_is_a_no_op_when_ssl_is_not_patched():
    before = ssl.SSLContext

    undo_pip_system_certs()

    assert ssl.SSLContext is before
