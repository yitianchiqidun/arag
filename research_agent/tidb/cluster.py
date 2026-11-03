"""A minimal local TiDB cluster with ``tiup playground``: start, stop, status, memory.

One PD, one TiKV and one TiDB of the version the docs describe (v8.5.8, D30), without TiFlash
and without the monitoring components, to stay within the machine's 7 GB of memory. The
playground runs in the background with a tag, so its data survives a restart; its log and
pid go to ``data/tidb_cluster/``.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

COMPONENTS = ("tidb-server", "tikv-server", "pd-server")


class ClusterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "v8.5.8"
    tag: str = "arag"
    tiup: str = "~/.tiup/bin/tiup"
    host: str = "127.0.0.1"
    port: int = 4000  # TiDB SQL port of the playground
    state_dir: str = "data/tidb_cluster"

    @property
    def log_file(self) -> Path:
        return Path(self.state_dir) / "playground.log"

    @property
    def pid_file(self) -> Path:
        return Path(self.state_dir) / "playground.pid"


def playground_command(cfg: ClusterConfig) -> list[str]:
    return [
        os.path.expanduser(cfg.tiup),
        "playground",
        cfg.version,
        "--tag",
        cfg.tag,
        "--db",
        "1",
        "--pd",
        "1",
        "--kv",
        "1",
        "--tiflash",
        "0",
        "--without-monitor",
        "--host",
        cfg.host,
    ]


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _pid(cfg: ClusterConfig) -> int | None:
    try:
        pid = int(cfg.pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    return pid


def start(cfg: ClusterConfig, wait_s: float = 300.0) -> int:
    """Start the playground in the background and wait until TiDB accepts connections;
    returns its pid (the running one if already started)."""
    running = _pid(cfg)
    if running is not None:
        return running
    if not Path(os.path.expanduser(cfg.tiup)).exists():
        raise FileNotFoundError(f"tiup not found at {cfg.tiup}; install it first")
    cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.log_file, "ab") as log:
        proc = subprocess.Popen(
            playground_command(cfg),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # its own process group, stopped as a whole
        )
    cfg.pid_file.write_text(str(proc.pid))
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"tiup playground exited with {proc.returncode}; see {cfg.log_file}")
        if port_open(cfg.host, cfg.port):
            return proc.pid
        time.sleep(2)
    raise TimeoutError(f"TiDB did not accept connections within {wait_s:.0f}s; see {cfg.log_file}")


def stop(cfg: ClusterConfig, wait_s: float = 60.0) -> bool:
    """Stop the playground (its whole process group); False if it was not running."""
    pid = _pid(cfg)
    if pid is None:
        cfg.pid_file.unlink(missing_ok=True)
        return False
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline and _pid(cfg) is not None:
        time.sleep(1)
    if _pid(cfg) is not None:
        os.killpg(pid, signal.SIGKILL)
    cfg.pid_file.unlink(missing_ok=True)
    return True


def component_memory(proc_dir: str = "/proc") -> dict[str, int]:
    """Resident memory (MiB) per cluster component, summed over its processes."""
    usage = dict.fromkeys(COMPONENTS, 0)
    for entry in Path(proc_dir).iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().split(b"\0")[0].decode(errors="ignore")
            status = (entry / "status").read_text()
        except OSError:
            continue
        name = os.path.basename(cmdline)
        if name in usage:
            rss = next((line for line in status.splitlines() if line.startswith("VmRSS:")), "")
            usage[name] += int(rss.split()[1]) // 1024 if rss else 0
    return usage


def status(cfg: ClusterConfig) -> dict[str, object]:
    return {
        "pid": _pid(cfg),
        "sql_port_open": port_open(cfg.host, cfg.port),
        "memory_mib": component_memory(),
        "log": str(cfg.log_file),
    }
