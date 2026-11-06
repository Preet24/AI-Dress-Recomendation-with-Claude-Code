"""Start the whole app from one terminal: the API, then the Streamlit UI pointed at it.

The two stay separate processes (the UI only talks to the API over HTTP); this module
only starts them in order, waits until each answers, and stops both when either one
ends or on Ctrl+C. Used by scripts/run_app.py.
"""

import logging
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
API_START_SECONDS = 180  # opens the wardrobe and loads CLIP before it answers
UI_START_SECONDS = 60
POLL_SECONDS = 0.5
STOP_SECONDS = 10


class LaunchError(RuntimeError):
    """A part of the app couldn't start; the message says what to do."""


def api_command(port: int, record: bool = False) -> list[str]:
    command = [sys.executable, str(ROOT / "scripts" / "serve.py"), "--port", str(port)]
    return [*command, "--record"] if record else command


def ui_command(port: int) -> list[str]:
    # Headless, so Streamlit never stops to ask for an email on its first run; the
    # launcher opens the browser itself once the UI answers.
    app = ROOT / "src" / "outfit" / "ui" / "app.py"
    return [sys.executable, "-m", "streamlit", "run", str(app)] + [
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def is_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except OSError:  # refused, timed out, or an HTTP error status
        return False


def wait_until_up(
    name: str,
    url: str,
    process: subprocess.Popen,
    timeout: float,
    check: Callable[[str], bool] = is_up,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    deadline = clock() + timeout
    while not check(url):
        if process.poll() is not None:
            raise LaunchError(
                f"The {name} stopped while starting (exit code {process.returncode}); "
                "its messages above say why"
            )
        if clock() > deadline:
            raise LaunchError(f"The {name} didn't answer at {url} within {timeout:.0f}s")
        sleep(POLL_SECONDS)


def watch(
    processes: dict[str, subprocess.Popen], sleep: Callable[[float], None] = time.sleep
) -> int:
    """Block until one part exits; returns its exit code."""
    while True:
        for name, process in processes.items():
            if (code := process.poll()) is not None:
                logger.info("The %s stopped (exit code %s); stopping the rest", name, code)
                return code
        sleep(POLL_SECONDS)


def stop(processes: dict[str, subprocess.Popen], grace: float = 0) -> None:
    """UI first, then the API. `grace`: time to finish on their own (they also got Ctrl+C)."""
    for process in reversed(processes.values()):
        try:
            process.wait(grace)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(STOP_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def run(
    api_port: int = 8000,
    ui_port: int = 8501,
    record: bool = False,
    open_browser: bool = True,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    check: Callable[[str], bool] = is_up,
    open_url: Callable[[str], object] = webbrowser.open,
) -> int:
    """Start the API, then the UI; returns when either stops (or on Ctrl+C), both stopped."""
    for port in (api_port, ui_port):
        if port_in_use(port):
            raise LaunchError(
                f"Port {port} is already in use: is the app, serve.py or streamlit already "
                "running? Stop it, or pick other ports with --port / --ui-port"
            )
    api_url = f"http://127.0.0.1:{api_port}"
    ui_url = f"http://localhost:{ui_port}"
    processes: dict[str, subprocess.Popen] = {}
    grace = 0.0
    try:
        logger.info("Starting the API (opening the wardrobe and loading models)...")
        processes["API"] = popen(api_command(api_port, record), cwd=ROOT)
        wait_until_up("API", f"{api_url}/health", processes["API"], API_START_SECONDS, check)

        logger.info("Starting the UI...")
        env = {**os.environ, "OUTFIT_API_URL": api_url}
        processes["UI"] = popen(ui_command(ui_port), cwd=ROOT, env=env)
        wait_until_up("UI", f"{ui_url}/_stcore/health", processes["UI"], UI_START_SECONDS, check)

        logger.info("Ready at %s (Ctrl+C stops everything)", ui_url)
        if open_browser:
            open_url(ui_url)
        return watch(processes)
    except KeyboardInterrupt:
        logger.info("Stopping...")
        grace = STOP_SECONDS
        return 0
    finally:
        stop(processes, grace)
