"""Outfit recommender."""

from outfit._ssl_fix import undo_pip_system_certs

undo_pip_system_certs()  # before anything imports openai; see _ssl_fix
