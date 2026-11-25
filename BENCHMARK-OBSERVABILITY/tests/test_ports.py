"""The ports have to agree with each other, and nothing else enforces that.

`config/gateway.yaml` routes to a fixed backend URL, `scripts/dev.sh` starts
the processes, and the CLI has its own defaults. When they drift, the gateway
still starts and still answers `/health` — it just forwards every request into
a port with nothing listening, which looks exactly like a very slow model.

This is the test that would have caught the drift the first time.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import pytest
from inferbench.cli import build_parser
from infergw.config import load_gateway
from infersched.config import load_scheduler

REPO = Path(__file__).resolve().parents[1]
DEV_SH = REPO / "scripts" / "dev.sh"


def _dev_sh_port(name: str) -> int:
    text = DEV_SH.read_text(encoding="utf-8")
    m = re.search(rf'^{name}="\$\{{{name}:-(\d+)\}}"', text, re.M)
    assert m, f"{name} has no default port in {DEV_SH}"
    return int(m.group(1))


def test_dev_script_engine_port_matches_the_healthy_backend():
    healthy = [b for b in load_gateway().backends if b.healthy]
    assert healthy, "gateway.yaml has no healthy backend to route to"
    ports = {urlparse(b.base_url).port for b in healthy}
    assert len(ports) == 1, f"healthy backends disagree on the port: {ports}"
    assert _dev_sh_port("ENGINE_PORT") == ports.pop()


def test_dev_script_ports_match_the_cli_defaults():
    parser = build_parser()
    assert _dev_sh_port("ENGINE_PORT") == parser.parse_args(["serve", "--config", "x"]).port
    assert _dev_sh_port("GATEWAY_PORT") == parser.parse_args(["gateway"]).port


def test_the_gateway_points_at_the_scheduler_the_dev_script_starts():
    """Two files and a script name the scheduler's port. They have to agree."""
    port = load_scheduler().port
    assert _dev_sh_port("SCHEDULER_PORT") == port
    assert urlparse(load_gateway().scheduler_url).port == port


def test_the_env_can_move_a_port_without_editing_the_configs(monkeypatch):
    monkeypatch.setenv("INFERSCHED_PORT", "9998")
    monkeypatch.setenv("INFERGW_SCHEDULER_URL", "http://localhost:9997")
    assert load_scheduler().port == 9998
    assert load_gateway().scheduler_url == "http://localhost:9997"


def test_an_empty_scheduler_url_means_no_scheduler(monkeypatch):
    """The A/B path: the gateway proxies straight to the backends."""
    monkeypatch.setenv("INFERGW_SCHEDULER_URL", "")
    assert load_gateway().scheduler_url == ""


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash on PATH")
def test_dev_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(DEV_SH)], check=True)
