"""The one-command launcher, with fake processes: no server is started."""

import socket
import subprocess

import pytest

from outfit import launcher
from outfit.launcher import LaunchError, run, stop, wait_until_up


class FakeProcess:
    """Exits with `code` after `polls` calls to poll(); `stubborn` ignores terminate()."""

    def __init__(self, code: int | None = None, polls: int = 0, stubborn: bool = False):
        self.code, self.polls, self.stubborn = code, polls, stubborn
        self.returncode: int | None = None
        self.calls: list[str] = []

    def poll(self) -> int | None:
        if self.returncode is None and self.code is not None:
            if self.polls == 0:
                self.returncode = self.code
            self.polls -= 1
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.poll() is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode

    def terminate(self) -> None:
        self.calls.append("terminate")
        if not self.stubborn:
            self.returncode = 1

    def kill(self) -> None:
        self.calls.append("kill")
        self.returncode = -9


class FakePopen:
    def __init__(self, *processes: FakeProcess):
        self.processes = list(processes)
        self.started: list[tuple[list[str], dict]] = []

    def __call__(self, command: list[str], **kwargs) -> FakeProcess:
        self.started.append((command, kwargs))
        return self.processes.pop(0)


@pytest.fixture
def free_ports(monkeypatch):
    monkeypatch.setattr(launcher, "port_in_use", lambda port: False)


def test_commands_run_the_project_scripts_with_this_python():
    assert launcher.api_command(8000)[1:] == [
        str(launcher.ROOT / "scripts" / "serve.py"),
        "--port",
        "8000",
    ]
    assert launcher.api_command(8080, record=True)[-1] == "--record"
    ui = launcher.ui_command(8502)
    assert ui[1:4] == ["-m", "streamlit", "run"]
    assert ui[-4:] == ["--server.port", "8502", "--server.headless", "true"]


def test_port_in_use_sees_a_listening_socket():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert launcher.port_in_use(port)
    assert not launcher.port_in_use(port)


def test_wait_until_up_returns_once_the_url_answers():
    answers = iter([False, False, True])
    wait_until_up(
        "API", "u", FakeProcess(), 10, check=lambda _: next(answers), sleep=lambda _: None
    )


def test_wait_until_up_reports_a_process_that_died():
    with pytest.raises(LaunchError, match="API stopped while starting \\(exit code 3\\)"):
        wait_until_up("API", "u", FakeProcess(code=3), 10, check=lambda _: False)


def test_wait_until_up_gives_up_after_the_timeout():
    ticks = iter(range(100))
    with pytest.raises(LaunchError, match="didn't answer at u within 5s"):
        wait_until_up(
            "UI",
            "u",
            FakeProcess(),
            5,
            check=lambda _: False,
            sleep=lambda _: None,
            clock=lambda: next(ticks),
        )


def test_stop_terminates_then_kills_a_process_that_ignores_it():
    polite, stubborn = FakeProcess(), FakeProcess(stubborn=True)
    stop({"API": polite, "UI": stubborn})
    assert polite.calls == ["terminate"]
    assert stubborn.calls == ["terminate", "kill"]


def test_stop_leaves_a_process_that_already_exited_alone():
    done = FakeProcess(code=0)
    stop({"API": done})
    assert done.calls == []


def test_run_starts_the_api_then_the_ui_pointed_at_it(free_ports):
    api, ui = FakeProcess(), FakeProcess(code=0)  # the UI exits as soon as it is watched
    popen, opened, checked = FakePopen(api, ui), [], []

    def check(url: str) -> bool:
        checked.append(url)
        return True

    code = run(8001, 8502, popen=popen, check=check, open_url=opened.append)

    assert code == 0
    (api_cmd, _), (ui_cmd, ui_kwargs) = popen.started
    assert "serve.py" in api_cmd[1] and "streamlit" in ui_cmd
    assert ui_kwargs["env"]["OUTFIT_API_URL"] == "http://127.0.0.1:8001"
    assert checked == ["http://127.0.0.1:8001/health", "http://localhost:8502/_stcore/health"]
    assert opened == ["http://localhost:8502"]
    assert api.calls == ["terminate"]  # the UI stopped, so the API is stopped too


def test_run_stops_with_the_apis_exit_code_and_never_starts_the_ui(free_ports):
    popen = FakePopen(FakeProcess(code=1))  # e.g. chat.py still holds the wardrobe
    with pytest.raises(LaunchError, match="exit code 1"):
        run(popen=popen, check=lambda _: False)
    assert len(popen.started) == 1


def test_run_refuses_a_busy_port_before_starting_anything(monkeypatch):
    monkeypatch.setattr(launcher, "port_in_use", lambda port: port == 8501)
    popen = FakePopen()
    with pytest.raises(LaunchError, match="Port 8501 is already in use"):
        run(popen=popen)
    assert popen.started == []


def test_ctrl_c_gives_both_time_to_stop_on_their_own(free_ports, monkeypatch):
    api, ui = FakeProcess(), FakeProcess()
    graces = []
    monkeypatch.setattr(
        launcher, "watch", lambda processes: (_ for _ in ()).throw(KeyboardInterrupt)
    )
    monkeypatch.setattr(launcher, "stop", lambda processes, grace: graces.append(grace))

    assert run(popen=FakePopen(api, ui), check=lambda _: True, open_url=lambda _: None) == 0
    assert graces == [launcher.STOP_SECONDS]
