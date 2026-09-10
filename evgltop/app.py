#!/usr/bin/env python3
"""evgltop - Interactive GPU monitoring TUI for SLURM clusters."""

import os
import subprocess
import re
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Button, Select, Input
from textual.timer import Timer
from textual.screen import ModalScreen
from rich.text import Text


# ── Data collection ──────────────────────────────────────────────────────────

CURRENT_USER = os.environ.get("USER", "unknown")


def run_cmd(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def get_hostname() -> str:
    return run_cmd(["hostname"])


def get_gpu_info() -> list[dict]:
    out = run_cmd([
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "mem_used": int(parts[2]),
                "mem_total": int(parts[3]),
                "gpu_util": int(parts[4]),
                "temp": parts[5],
                "power": parts[6],
            })
    return gpus


def get_gpu_uuid_map() -> dict[str, int]:
    out = run_cmd(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"])
    mapping = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            mapping[parts[1]] = int(parts[0])
    return mapping


def get_gpu_processes() -> dict[int, list[dict]]:
    uuid_map = get_gpu_uuid_map()
    out = run_cmd([
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory,name",
        "--format=csv,noheader,nounits",
    ])
    procs: dict[int, list[dict]] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            pid = int(parts[0])
            gpu_uuid = parts[1]
            mem = parts[2]
            cmd_name = parts[3]
            gpu_idx = uuid_map.get(gpu_uuid, -1)
            user = get_pid_user(pid)
            procs.setdefault(gpu_idx, []).append({
                "pid": pid,
                "user": user,
                "mem_mib": int(mem) if mem.isdigit() else 0,
                "command": cmd_name,
            })
    return procs


def get_pid_user(pid: int) -> str:
    out = run_cmd(["ps", "-o", "user=", "-p", str(pid)])
    return out.strip() if out else "unknown"


def parse_slurm_duration(s: str) -> int:
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        h, m, sec = 0, int(parts[0]), int(parts[1])
    else:
        return 0
    return days * 86400 + h * 3600 + m * 60 + sec


def format_duration(seconds: int) -> str:
    if seconds < 0:
        return "expired"
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    mins = (seconds % 3600) // 60
    if days > 0:
        return f"{days}d {hours:02d}h {mins:02d}m"
    if hours > 0:
        return f"{hours}h {mins:02d}m"
    return f"{mins}m"


def format_mem(mib: int) -> str:
    if mib >= 1024:
        return f"{mib / 1024:.1f}G"
    return f"{mib}M"


def get_slurm_gpu_allocations(hostname: str) -> dict[int, dict]:
    allocations: dict[int, dict] = {}
    out = run_cmd(["squeue", "--noheader", "--format=%i %u %T %N", "--states=RUNNING"])
    if not out:
        return allocations

    job_ids = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and hostname in parts[3]:
            job_ids.append(parts[0])

    for job_id in job_ids:
        detail = run_cmd(["scontrol", "show", "job", job_id, "-d"])
        if not detail:
            continue
        user = job_name = runtime_str = timelimit_str = partition = ""
        batch_flag = 0
        num_gpus = 0
        alloc_mem = ""
        gpu_indices = []
        for line in detail.splitlines():
            line = line.strip()
            if "UserId=" in line:
                m = re.search(r"UserId=(\w+)", line)
                if m:
                    user = m.group(1)
            if "JobName=" in line:
                m = re.search(r"JobName=(.+?)(?:\s|$)", line)
                if m:
                    job_name = m.group(1)
            if "RunTime=" in line:
                m = re.search(r"RunTime=(\S+)", line)
                if m:
                    runtime_str = m.group(1)
            if "TimeLimit=" in line:
                m = re.search(r"TimeLimit=(\S+)", line)
                if m:
                    timelimit_str = m.group(1)
            if "BatchFlag=" in line:
                m = re.search(r"BatchFlag=(\d+)", line)
                if m:
                    batch_flag = int(m.group(1))
            if "Partition=" in line:
                m = re.search(r"Partition=(\S+)", line)
                if m:
                    partition = m.group(1)
            if "MinMemoryNode=" in line:
                m = re.search(r"MinMemoryNode=(\S+)", line)
                if m:
                    raw = m.group(1)
                    # Convert MB to human-readable
                    try:
                        mb = int(raw.rstrip("M"))
                        alloc_mem = f"{mb // 1024}G" if mb >= 1024 else f"{mb}M"
                    except ValueError:
                        alloc_mem = raw
            if "TresPerNode=" in line:
                m = re.search(r"gpu:(\d+)", line)
                if m:
                    num_gpus = int(m.group(1))
            if "GRES=" in line and "IDX:" in line:
                m = re.search(r"IDX:([\d,\-]+)", line)
                if m:
                    for part in m.group(1).split(","):
                        if "-" in part:
                            s, e = part.split("-")
                            gpu_indices.extend(range(int(s), int(e) + 1))
                        else:
                            gpu_indices.append(int(part))

        runtime_sec = parse_slurm_duration(runtime_str)
        timelimit_sec = parse_slurm_duration(timelimit_str)
        remaining_sec = timelimit_sec - runtime_sec
        job_type = "sbatch" if batch_flag else "srun"

        for idx in gpu_indices:
            allocations[idx] = {
                "user": user,
                "job_id": job_id,
                "job_name": job_name,
                "job_type": job_type,
                "partition": partition,
                "num_gpus": num_gpus,
                "alloc_mem": alloc_mem,
                "runtime": format_duration(runtime_sec),
                "timelimit": format_duration(timelimit_sec),
                "remaining": format_duration(remaining_sec),
                "remaining_sec": remaining_sec,
            }
    return allocations


def _get_running_end_times() -> list[tuple[datetime, int]]:
    """Get (end_time, num_gpus) for current user's running jobs, sorted by end time."""
    out = run_cmd([
        "squeue", "--noheader", "--user", CURRENT_USER,
        "--format=%e %b", "--states=RUNNING",
    ])
    ends = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                end_dt = datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%S")
                m = re.search(r"gpu:(\d+)", parts[1])
                ngpu = int(m.group(1)) if m else 1
                ends.append((end_dt, ngpu))
            except ValueError:
                continue
    ends.sort(key=lambda x: x[0])
    return ends


def _estimate_qos_start(needed_gpus: int,
                         gpu_free_events: list[tuple[datetime, int]],
                         ) -> tuple[str, str, datetime | None]:
    """Estimate when a pending job can start based on GPU free events.

    gpu_free_events: sorted list of (time, gpus_freed_at_that_time).
    Returns (est_wait, est_display, est_datetime).
    """
    now = datetime.now()
    freed = 0
    est_dt = None
    for free_dt, ngpu in gpu_free_events:
        freed += ngpu
        if freed >= needed_gpus:
            est_dt = free_dt
            break

    if est_dt is None:
        return ("", "", None)

    delta = est_dt - now
    if delta.total_seconds() <= 0:
        return ("soon", "soon", est_dt)

    return (
        format_duration(int(delta.total_seconds())),
        est_dt.strftime("%m/%d %H:%M"),
        est_dt,
    )


def get_pending_jobs() -> list[dict]:
    """Get pending SLURM jobs requesting GPUs with estimated start time."""
    out = run_cmd([
        "squeue", "--noheader",
        "--format=%i|%u|%P|%b|%T|%S|%l|%r",
        "--states=PENDING",
    ])

    # Pre-fetch running end times for QOS estimation
    running_ends = _get_running_end_times()

    # Build a mutable list of GPU free events for chaining pending jobs
    # As each pending job is "scheduled", it occupies GPUs and frees them later
    gpu_free_events = list(running_ends)  # (end_time, gpus_freed)

    jobs = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 6 or "gpu" not in parts[3]:
            continue
        est_start_raw = parts[5].strip() if len(parts) > 5 else ""
        timelimit = parts[6].strip() if len(parts) > 6 else ""
        reason_raw = parts[7].strip() if len(parts) > 7 else ""
        # Shorten long reasons
        reason_map = {
            "QOSMaxGRESPerUser": "QOS limit",
            "Priority": "Priority",
            "Resources": "Resources",
        }
        reason = reason_map.get(reason_raw, reason_raw[:20]) if reason_raw else ""
        if True:
            est_wait = ""
            est_display = ""

            # Parse GPU count from gres
            m_gpu = re.search(r"gpu:(\d+)", parts[3])
            needed_gpus = int(m_gpu.group(1)) if m_gpu else 1

            if est_start_raw and est_start_raw not in ("N/A", "Unknown"):
                # SLURM provided an estimate
                try:
                    start_dt = datetime.strptime(est_start_raw, "%Y-%m-%dT%H:%M:%S")
                    now = datetime.now()
                    delta = start_dt - now
                    if delta.total_seconds() > 0:
                        est_wait = format_duration(int(delta.total_seconds()))
                        est_display = start_dt.strftime("%m/%d %H:%M")
                    else:
                        est_wait = "soon"
                        est_display = "soon"
                except ValueError:
                    pass
            elif reason and est_start_raw in ("N/A", "Unknown", "") and parts[1].strip() == CURRENT_USER:
                # Estimate based on when enough GPUs free up
                est_wait, est_display, est_dt = _estimate_qos_start(
                    needed_gpus, gpu_free_events,
                )
                if est_wait and est_wait != "soon":
                    est_wait = "~" + est_wait

                # Chain: this pending job will occupy GPUs and free them later
                if est_dt:
                    tl_sec = parse_slurm_duration(timelimit) if timelimit else 86400
                    from datetime import timedelta
                    job_end = est_dt + timedelta(seconds=tl_sec)
                    # Remove consumed free events up to est_dt
                    consumed = 0
                    new_events = []
                    for ev_dt, ev_gpu in gpu_free_events:
                        if consumed < needed_gpus and ev_dt <= est_dt:
                            consumed += ev_gpu
                            remaining = consumed - needed_gpus
                            if remaining > 0:
                                new_events.append((ev_dt, remaining))
                        else:
                            new_events.append((ev_dt, ev_gpu))
                    # Add this job's end as a new free event
                    new_events.append((job_end, needed_gpus))
                    new_events.sort(key=lambda x: x[0])
                    gpu_free_events = new_events

            jobs.append({
                "job_id": parts[0].strip(),
                "user": parts[1].strip(),
                "partition": parts[2].strip(),
                "gres": parts[3].strip(),
                "est_wait": est_wait,
                "est_start": est_display,
                "reason": reason,
            })
    return jobs


def get_system_resources() -> dict:
    """Get CPU, RAM, and disk usage."""
    # RAM
    mem = {}
    meminfo = run_cmd(["free", "-b"])
    for line in meminfo.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            mem["ram_total"] = int(parts[1])
            mem["ram_used"] = int(parts[2])
            mem["ram_available"] = int(parts[6])
            break

    # CPU usage (from /proc/stat snapshot delta)
    cpu_pct = 0
    try:
        loadavg = run_cmd(["cat", "/proc/loadavg"])
        if loadavg:
            load1 = float(loadavg.split()[0])
            ncpu = int(run_cmd(["nproc"]) or "1")
            cpu_pct = min(100, load1 / ncpu * 100)
    except (ValueError, IndexError):
        pass

    def parse_df(path: str) -> tuple[int, int, int]:
        """Parse df output, handling wrapped lines."""
        out = run_cmd(["df", "-B1", path])
        # Join all lines after header into one, then split by whitespace
        lines = out.splitlines()[1:]
        if not lines:
            return (0, 0, 0)
        joined = " ".join(lines)
        parts = joined.split()
        # Format: filesystem total used avail use% mount
        for i, p in enumerate(parts):
            if p.endswith("%") and i >= 3:
                return (int(parts[i - 3]), int(parts[i - 2]), int(parts[i - 1]))
        return (0, 0, 0)

    disk = {}
    s_total, s_used, s_avail = parse_df("/local/scratch")
    disk["scratch_total"] = s_total
    disk["scratch_used"] = s_used
    disk["scratch_avail"] = s_avail

    h_total, h_used, h_avail = parse_df(f"/home/{CURRENT_USER}")
    disk["home_total"] = h_total
    disk["home_used"] = h_used
    disk["home_avail"] = h_avail

    return {"cpu_pct": cpu_pct, **mem, **disk}


def get_tmux_sessions() -> list[dict]:
    """Get gpumon-created tmux sessions (gpu* prefix) with SLURM status."""
    out = run_cmd(["tmux", "list-sessions", "-F",
                   "#{session_name} #{session_created} #{session_attached}"])
    # Get running/pending job IDs with GPU index info
    job_states: dict[str, str] = {}
    job_gpu_idx: dict[str, str] = {}
    squeue_out = run_cmd([
        "squeue", "--noheader", "--user", CURRENT_USER,
        "--format=%i %T", "--states=RUNNING,PENDING",
    ])
    for line in squeue_out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            job_states[parts[0]] = parts[1]

    # Get GPU index for running jobs
    hostname = get_hostname()
    for job_id, state in job_states.items():
        if state == "RUNNING":
            detail = run_cmd(["scontrol", "show", "job", job_id, "-d"])
            if detail and hostname in detail:
                m = re.search(r"IDX:([\d,\-]+)", detail)
                if m:
                    job_gpu_idx[job_id] = m.group(1)

    sessions = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].startswith("gpu"):
            from datetime import datetime as dt
            name = parts[0]
            try:
                created = dt.fromtimestamp(int(parts[1])).strftime("%m/%d %H:%M")
            except (ValueError, OSError):
                created = "?"
            attached = int(parts[2]) > 0

            # Determine SLURM state from session name
            # gpu{N}-{JOBID} or gpu{N}-pending-{HHMMSS}
            state = "unknown"
            gpu_idx = ""
            if "-pending-" in name:
                state = "pending"
            else:
                # Extract job ID from name (last part after -)
                job_id = name.rsplit("-", 1)[-1]
                state = job_states.get(job_id, "ended").lower()
                gpu_idx = job_gpu_idx.get(job_id, "")

            sessions.append({
                "name": name,
                "created": created,
                "attached": attached,
                "state": state,
                "gpu_idx": gpu_idx,
            })
    return sessions


def format_bytes(b: int) -> str:
    if b >= 1024**4:
        return f"{b / 1024**4:.1f}T"
    if b >= 1024**3:
        return f"{b / 1024**3:.0f}G"
    if b >= 1024**2:
        return f"{b / 1024**2:.0f}M"
    return f"{b}B"


def cancel_slurm_job(job_id: str) -> str:
    """Cancel a SLURM job."""
    result = subprocess.run(
        ["scancel", job_id], capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        return f"Job {job_id} cancelled."
    return f"Failed: {result.stderr.strip()}"


def find_tmux_for_job(job_id: str) -> str | None:
    """Find tmux session named gpu*-{job_id}."""
    tmux_sessions = run_cmd(["tmux", "list-sessions", "-F", "#{session_name}"])
    if tmux_sessions:
        for sess in tmux_sessions.splitlines():
            if sess.endswith(f"-{job_id}"):
                return sess
    return None


def kill_tmux_session(session_name: str) -> str:
    result = subprocess.run(
        ["tmux", "kill-session", "-t", session_name],
        capture_output=True, text=True, timeout=5,
    )
    if result.returncode == 0:
        return f"tmux '{session_name}' closed."
    return f"Failed to close tmux: {result.stderr.strip()}"


def pick_qos(num_gpus: int) -> str:
    """Pick optimal QOS based on current usage across all tiers.

    Available to yjlab (sorted by priority, highest first):
        normal: up to 1 GPU, priority 40000
        gpu02:  up to 2 GPU, priority 20000
        gpu04:  up to 4 GPU, priority 10000

    Strategy: use the most restrictive (highest priority) QOS
    that has enough remaining capacity for the request.
    """
    # QOS tiers: (name, max_gpus), ordered by priority (highest first)
    tiers = [
        ("normal", 1),
        ("gpu02", 2),
        ("gpu04", 4),
    ]

    # Count current GPU usage per QOS
    out = run_cmd([
        "squeue", "--noheader", "--user", CURRENT_USER,
        "--format=%q %b", "--states=RUNNING,PENDING",
    ])
    usage: dict[str, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            qos = parts[0]
            m = re.search(r"gpu:(\d+)", parts[1])
            if m:
                usage[qos] = usage.get(qos, 0) + int(m.group(1))

    # Find best QOS: most restrictive with enough remaining capacity
    for qos_name, max_gpus in tiers:
        if max_gpus < num_gpus:
            continue
        used = usage.get(qos_name, 0)
        if used + num_gpus <= max_gpus:
            return qos_name

    # Fallback: use least restrictive
    return "gpu04"


# Node evgl1: 64 logical CPUs (2 sockets x 16 cores x 2 threads) shared by 4 GPUs.
# Allocate CPUs proportionally to requested GPUs, leaving headroom for CPU-only jobs.
CPUS_PER_GPU = 12


def cpus_for_gpus(num_gpus: int) -> int:
    """Number of logical CPUs to request for the given GPU count."""
    return max(1, num_gpus) * CPUS_PER_GPU


def launch_srun(num_gpus: int, partition: str, mem: str) -> str:
    """Launch srun in a detached tmux session.

    Naming: gpu{N}-pending-HHMMSS initially, renamed to gpu{N}-{JOBID}
    by rename_pending_tmux() in the refresh loop.
    """
    qos = pick_qos(num_gpus)
    cpus = cpus_for_gpus(num_gpus)
    temp_name = f"gpu{num_gpus}-pending-{datetime.now().strftime('%H%M%S')}"
    srun_cmd = (
        f"srun --gres=gpu:{num_gpus} --partition={partition} "
        f"--qos={qos} --mem={mem} --cpus-per-task={cpus} --pty bash"
    )
    result = subprocess.run(
        ["tmux", "new-session", "-d", "-s", temp_name, srun_cmd],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        return f"Launched tmux '{temp_name}'"
    return f"Failed: {result.stderr.strip()}"


def rename_pending_tmux() -> list[str]:
    """Rename gpu*-pending-* tmux sessions to gpu{N}-{JOBID} once job starts.

    Returns list of new names for sessions that were renamed.
    Matches tmux pane PID to SLURM job via AllocNode:Sid in scontrol.
    """
    renamed = []
    tmux_sessions = run_cmd(["tmux", "list-sessions", "-F", "#{session_name}"])
    if not tmux_sessions:
        return renamed

    pending_sessions = [s for s in tmux_sessions.splitlines() if "-pending-" in s]
    if not pending_sessions:
        return renamed

    # Get running/pending jobs for current user
    job_out = run_cmd([
        "squeue", "--noheader", "--user", CURRENT_USER,
        "--format=%i", "--states=RUNNING,PENDING",
    ])
    if not job_out:
        return renamed

    # Build map: srun PID -> job_id via scontrol AllocNode:Sid
    pid_to_job: dict[str, str] = {}
    for job_id in job_out.splitlines():
        detail = run_cmd(["scontrol", "show", "job", job_id.strip()])
        m = re.search(r"AllocNode:Sid=\S+:(\d+)", detail)
        if m:
            pid_to_job[m.group(1)] = job_id.strip()

    for sess in pending_sessions:
        pane_pid = run_cmd(["tmux", "display-message", "-t", sess, "-p", "#{pane_pid}"])
        if not pane_pid:
            continue
        job_id = pid_to_job.get(pane_pid)
        if job_id:
            gpu_n = sess.split("-")[0]  # e.g. "gpu2"
            new_name = f"{gpu_n}-{job_id}"
            subprocess.run(
                ["tmux", "rename-session", "-t", sess, new_name],
                capture_output=True, timeout=5,
            )
            renamed.append(new_name)
    return renamed


# ── Rich rendering helpers ───────────────────────────────────────────────────

WASTE_THRESHOLD_SEC = 600  # 10 minutes with 0% util = waste


def make_bar(value: float, width: int) -> Text:
    filled = int(value * width)
    empty = width - filled
    if value < 0.5:
        color = "green"
    elif value < 0.8:
        color = "yellow"
    else:
        color = "red"
    bar = Text()
    bar.append("━" * filled, style=f"bold {color}")
    bar.append("─" * empty, style="bright_black")
    return bar


def render_gpu_content(gpu: dict, gpu_procs: list[dict], slurm: dict | None,
                       bar_width: int) -> Text:
    lines = Text()
    short_name = gpu["name"].replace("NVIDIA ", "").replace(" Server Edition", "")

    temp_val = int(gpu["temp"]) if gpu["temp"].isdigit() else 0
    temp_color = "green" if temp_val < 60 else ("yellow" if temp_val < 80 else "red")
    try:
        power_str = f"{float(gpu['power']):.0f}W"
    except ValueError:
        power_str = gpu["power"]

    lines.append(f" {short_name}", style="dim")
    lines.append(f"  {gpu['temp']}°C", style=temp_color)
    lines.append(f"  {power_str}\n", style="dim")

    util = gpu["gpu_util"]
    mem_ratio = gpu["mem_used"] / gpu["mem_total"] if gpu["mem_total"] > 0 else 0

    lines.append(" GPU ", style="dim")
    lines.append_text(make_bar(util / 100, bar_width))
    lines.append(f" {util:>3}%\n", style="bold white")

    lines.append(" MEM ", style="dim")
    lines.append_text(make_bar(mem_ratio, bar_width))
    lines.append(f" {format_mem(gpu['mem_used']):>5}/{format_mem(gpu['mem_total'])}\n", style="white")

    sep_w = bar_width + 16
    lines.append(f" {'· ' * (sep_w // 2)}\n", style="bright_black")

    if slurm:
        lines.append(f" {slurm['user']}", style="bold yellow")
        lines.append(f"  Job {slurm['job_id']}", style="white")
        lines.append(f"  {slurm['job_type']}", style="bold cyan")
        lines.append(f"  {slurm['partition']}", style="dim")
        if slurm.get("num_gpus", 1) > 1:
            lines.append(f"  {slurm['num_gpus']}GPUs", style="white")
        if slurm.get("alloc_mem"):
            lines.append(f"  {slurm['alloc_mem']}", style="dim")
        lines.append("\n")

        remaining_sec = slurm["remaining_sec"]
        r_style = "bold red" if remaining_sec < 3600 else ("yellow" if remaining_sec < 86400 else "green")

        lines.append(f" Run {slurm['runtime']}", style="dim")
        lines.append(f"  Left ", style="dim")
        lines.append(f"{slurm['remaining']}\n", style=r_style)

        # Waste detection: allocated, no processes, running > threshold
        is_waste = (
            not gpu_procs
            and util == 0
            and parse_slurm_duration(slurm.get("runtime_raw", "0")) > WASTE_THRESHOLD_SEC
        )

        if gpu_procs:
            for proc in gpu_procs:
                lines.append(f"  {proc['command']}", style="cyan")
                lines.append(f"  PID {proc['pid']}", style="dim")
                lines.append(f"  {format_mem(proc['mem_mib'])}\n", style="white")
        else:
            lines.append(" No active computation\n", style="italic red")

    elif gpu_procs:
        for proc in gpu_procs:
            lines.append(f" {proc['user']}", style="bold yellow")
            lines.append(f"  {proc['command']}", style="cyan")
            lines.append(f"  PID {proc['pid']}", style="dim")
            lines.append(f"  {format_mem(proc['mem_mib'])}\n", style="white")
    else:
        lines.append(" Available\n", style="dim bright_green")

    return lines


# ── Confirm Cancel dialog ────────────────────────────────────────────────────

class ConfirmCancelScreen(ModalScreen[bool]):

    CSS = """
    ConfirmCancelScreen {
        align: center middle;
    }
    #confirm-dialog {
        width: 40;
        max-height: 7;
        border: round $error;
        background: $surface;
        padding: 0 1;
    }
    #confirm-msg {
        width: 100%;
        height: 1;
        text-align: center;
    }
    #confirm-buttons {
        width: 100%;
        height: 3;
        align: center middle;
    }
    #confirm-buttons Button {
        margin: 0 1;
        min-width: 10;
    }
    """

    def __init__(self, job_id: str, gpu_idx: int) -> None:
        super().__init__()
        self.job_id = job_id
        self.gpu_idx = gpu_idx

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            msg = f"Cancel Job [bold]{self.job_id}[/]"
            if self.gpu_idx >= 0:
                msg += f" on GPU {self.gpu_idx}"
            msg += "?"
            yield Static(msg, id="confirm-msg")
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes", variant="error", id="btn-yes")
                yield Button("No", variant="primary", id="btn-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-yes")

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(False)


# ── Confirm Tmux Kill dialog ─────────────────────────────────────────────────

class ConfirmTmuxScreen(ModalScreen[bool]):

    CSS = """
    ConfirmTmuxScreen {
        align: center middle;
    }
    #tmux-dialog {
        width: 42;
        max-height: 7;
        border: round $warning;
        background: $surface;
        padding: 0 1;
    }
    #tmux-msg {
        width: 100%;
        height: 1;
        text-align: center;
    }
    #tmux-buttons {
        width: 100%;
        height: 3;
        align: center middle;
    }
    #tmux-buttons Button {
        margin: 0 1;
        min-width: 10;
    }
    """

    def __init__(self, session_name: str) -> None:
        super().__init__()
        self.session_name = session_name

    def compose(self) -> ComposeResult:
        with Vertical(id="tmux-dialog"):
            yield Static(
                f"Also close tmux [bold]{self.session_name}[/]?",
                id="tmux-msg",
            )
            with Horizontal(id="tmux-buttons"):
                yield Button("Yes", variant="error", id="btn-yes")
                yield Button("Keep", variant="primary", id="btn-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-yes")

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(False)


# ── Help screen ──────────────────────────────────────────────────────────────

class HelpScreen(ModalScreen[None]):

    CSS = """
    HelpScreen {
        align: center middle;
    }
    #help-dialog {
        width: 55;
        height: auto;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    #help-title {
        width: 100%;
        height: 1;
        text-align: center;
        margin-bottom: 1;
    }
    #help-content {
        width: 100%;
        height: auto;
    }
    #help-close {
        width: 100%;
        height: 3;
        align: center middle;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Static("[bold]evgltop Help[/]", id="help-title")
            help_text = Text()
            keys = [
                ("q", "Quit"),
                ("r", "Refresh now"),
                ("n", "New SLURM session"),
                ("h", "This help screen"),
                ("Esc", "Close dialog"),
            ]
            help_text.append(" Keyboard\n", style="bold cyan")
            for key, desc in keys:
                help_text.append(f"   {key:<6}", style="bold white")
                help_text.append(f" {desc}\n", style="dim")

            help_text.append("\n GPU Status\n", style="bold cyan")
            statuses = [
                ("● ACTIVE", "green", "GPU in use (process running)"),
                ("● ALLOCATED", "yellow", "SLURM allocated, no process"),
                ("● IDLE", "red", "Allocated, idle > 10 min"),
                ("○ FREE", "bright_green", "Available"),
            ]
            for label, color, desc in statuses:
                help_text.append(f"   {label:<14}", style=f"bold {color}")
                help_text.append(f" {desc}\n", style="dim")

            help_text.append("\n Sessions\n", style="bold cyan")
            sess_info = [
                ("● RUN", "green", "Job running"),
                ("◌ WAIT", "yellow", "Job pending"),
                ("○ END", "red", "Job ended, tmux remains"),
            ]
            for label, color, desc in sess_info:
                help_text.append(f"   {label:<14}", style=f"bold {color}")
                help_text.append(f" {desc}\n", style="dim")

            help_text.append("\n QOS (auto-selected)\n", style="bold cyan")
            qos_info = [
                ("normal", "1 GPU", "40000"),
                ("gpu02", "2 GPU", "20000"),
                ("gpu04", "4 GPU", "10000"),
            ]
            for name, limit, pri in qos_info:
                help_text.append(f"   {name:<8}", style="bold white")
                help_text.append(f" {limit:<6}", style="cyan")
                help_text.append(f" priority {pri}\n", style="dim")

            yield Static(help_text, id="help-content")
            with Horizontal(id="help-close"):
                yield Button("Close", variant="primary", id="btn-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    def on_key(self, event) -> None:
        if event.key in ("escape", "h", "q"):
            self.dismiss(None)


# ── New Session dialog ───────────────────────────────────────────────────────

class NewSessionScreen(ModalScreen[dict | None]):

    CSS = """
    NewSessionScreen {
        align: center middle;
    }
    #session-dialog {
        width: 45;
        height: auto;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    #session-dialog .title-label {
        width: 100%;
        height: 1;
        text-align: center;
        margin-bottom: 1;
    }
    .field-row {
        width: 100%;
        height: 3;
        align: left middle;
    }
    .field-row Static {
        width: 12;
        height: 1;
    }
    .field-row Select {
        width: 1fr;
    }
    #qos-info {
        width: 100%;
        height: 1;
        text-align: center;
        margin-top: 1;
    }
    #session-buttons {
        width: 100%;
        height: 3;
        align: center middle;
        margin-top: 1;
    }
    #session-buttons Button {
        margin: 0 1;
        min-width: 12;
    }
    """

    def __init__(self, free_gpus: int) -> None:
        super().__init__()
        self.free_gpus = free_gpus

    def compose(self) -> ComposeResult:
        gpu_options = [(f"{i} GPU{'s' if i > 1 else ''}", i) for i in range(1, 5)]
        partition_options = [
            ("day (1d)", "day"),
            ("week (7d)", "week"),
            ("month (31d)", "month"),
        ]
        mem_options = [
            ("32G", "32G"),
            ("64G", "64G"),
            ("128G", "128G"),
            ("256G", "256G"),
            ("512G", "512G"),
            ("768G", "768G"),
            ("1024G", "1024G"),
        ]

        with Vertical(id="session-dialog"):
            yield Static("[bold]New SLURM Session[/]", classes="title-label")
            with Horizontal(classes="field-row"):
                yield Static("GPUs")
                yield Select(gpu_options, value=1, id="sel-gpus")
            with Horizontal(classes="field-row"):
                yield Static("Partition")
                yield Select(partition_options, value="day", id="sel-partition")
            with Horizontal(classes="field-row"):
                yield Static("Memory")
                yield Select(mem_options, value="32G", id="sel-mem")
            yield Static(id="qos-info")
            with Horizontal(id="session-buttons"):
                yield Button("Launch", variant="success", id="btn-launch")
                yield Button("Cancel", variant="default", id="btn-cancel")

    def on_mount(self) -> None:
        self._update_qos_info()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "sel-gpus":
            self._update_qos_info()

    def _update_qos_info(self) -> None:
        num_gpus = self.query_one("#sel-gpus", Select).value
        if num_gpus is None or num_gpus == Select.BLANK:
            return
        qos = pick_qos(int(num_gpus))
        qos_labels = {"normal": "normal (1 GPU, high priority)",
                      "gpu02": "gpu02 (2 GPU, med priority)",
                      "gpu04": "gpu04 (4 GPU, low priority)"}
        label = qos_labels.get(qos, qos)
        info = self.query_one("#qos-info", Static)
        qos_text = f"QOS: [bold cyan]{qos}[/] [dim]({label.split('(')[1]}[/]" if "(" in label else f"QOS: [bold cyan]{qos}[/]"
        info.update(f"{qos_text}  CPUs: [bold cyan]{cpus_for_gpus(int(num_gpus))}[/]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-launch":
            num_gpus = self.query_one("#sel-gpus", Select).value
            partition = self.query_one("#sel-partition", Select).value
            mem = self.query_one("#sel-mem", Select).value
            self.dismiss({"num_gpus": num_gpus, "partition": partition, "mem": mem})
        else:
            self.dismiss(None)

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)


# ── GPU Card widget ──────────────────────────────────────────────────────────

class GpuCard(Vertical):

    DEFAULT_CSS = """
    GpuCard {
        width: 1fr;
        height: auto;
        margin: 0 0 1 0;
        border: round $secondary;
        padding: 0;
    }
    GpuCard.active {
        border: round green;
    }
    GpuCard.allocated {
        border: round yellow;
    }
    GpuCard.waste {
        border: round red;
    }
    GpuCard.free {
        border: round $secondary;
    }
    GpuCard .card-content {
        width: 100%;
        height: auto;
    }
    GpuCard .cancel-btn {
        width: 100%;
        min-width: 12;
        margin: 0 1 0 1;
    }
    """

    def __init__(self, gpu_idx: int) -> None:
        super().__init__(id=f"gpu-{gpu_idx}")
        self.gpu_idx = gpu_idx
        self.job_id: str | None = None
        self.job_user: str = ""

    def compose(self) -> ComposeResult:
        yield Static(classes="card-content")
        yield Button("Cancel Job", variant="error", classes="cancel-btn")

    def on_mount(self) -> None:
        self.query_one(".cancel-btn", Button).display = False

    def update_card(self, gpu: dict, gpu_procs: list[dict],
                    slurm: dict | None, bar_width: int) -> None:
        has_procs = bool(gpu_procs)
        is_waste = (
            slurm
            and not has_procs
            and gpu["gpu_util"] == 0
            and slurm["remaining_sec"] < (
                parse_slurm_duration(slurm.get("timelimit", "0"))
                - WASTE_THRESHOLD_SEC
            )
        )

        self.remove_class("active", "allocated", "waste", "free")
        if is_waste:
            self.add_class("waste")
            self.border_title = f"● GPU {self.gpu_idx} IDLE"
        elif has_procs:
            self.add_class("active")
            self.border_title = f"● GPU {self.gpu_idx} ACTIVE"
        elif slurm:
            self.add_class("allocated")
            self.border_title = f"● GPU {self.gpu_idx} ALLOCATED"
        else:
            self.add_class("free")
            self.border_title = f"○ GPU {self.gpu_idx} FREE"

        self.job_id = slurm["job_id"] if slurm else None
        self.job_user = slurm["user"] if slurm else ""

        content = render_gpu_content(gpu, gpu_procs, slurm, bar_width)
        self.query_one(".card-content", Static).update(content)

        btn = self.query_one(".cancel-btn", Button)
        if slurm and slurm["user"] == CURRENT_USER:
            btn.label = f"Cancel Job {slurm['job_id']}"
            btn.display = True
        else:
            btn.display = False


# ── Pending Jobs widget ──────────────────────────────────────────────────────

class PendingJobs(Vertical):

    DEFAULT_CSS = """
    PendingJobs {
        width: 100%;
        height: auto;
        padding: 0 1;
    }
    PendingJobs .pending-header {
        width: 100%;
        height: 1;
    }
    PendingJobs .pending-job-row {
        width: 100%;
        height: auto;
        padding: 0 1;
    }
    PendingJobs .pending-job-row Static {
        width: 1fr;
        height: 1;
    }
    PendingJobs .pending-job-row Button {
        min-width: 10;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(classes="pending-header")
        # Job rows are mounted dynamically

    def update_pending(self, pending: list[dict]) -> None:
        header = self.query_one(".pending-header", Static)

        # Remove old job rows
        for row in list(self.query(".pending-job-row")):
            row.remove()

        if not pending:
            header.update("")
            return

        mine = [j for j in pending if j["user"] == CURRENT_USER]
        h = Text()
        h.append(f" Pending: {len(pending)} waiting", style="bold magenta")
        if mine:
            h.append(f" (you: {len(mine)})", style="yellow")
        # Show other users' pending
        others = [j for j in pending if j["user"] != CURRENT_USER]
        for job in others[:3]:
            h.append(f"  {job['user']}", style="dim")
            h.append(f":{job['gres']}", style="dim cyan")
        if len(others) > 3:
            h.append(f" +{len(others) - 3} more", style="dim")
        header.update(h)

        # Add cancel buttons for my pending jobs
        for job in mine:
            info = Text()
            info.append(f" Job {job['job_id']}", style="white")
            info.append(f"  {job['gres']}", style="cyan")
            info.append(f"  {job['partition']}", style="dim")
            if job.get("est_wait"):
                info.append(f"  ~{job['est_wait']}", style="yellow")
            if job.get("est_start") and job["est_start"] != "soon":
                info.append(f" ({job['est_start']})", style="dim")
            if job.get("reason"):
                info.append(f"  {job['reason']}", style="dim red")

            row = Horizontal(classes="pending-job-row")
            self.mount(row)
            row.mount(Static(info))
            row.mount(Button(
                f"Cancel {job['job_id']}",
                variant="error",
                id=f"cancel-pending-{job['job_id']}",
            ))


# ── System Info widget ────────────────────────────────────────────────────────

class SystemInfo(Vertical):

    DEFAULT_CSS = """
    SystemInfo {
        width: 100%;
        height: auto;
        border: round $secondary;
        margin: 1 0;
        padding: 0;
    }
    SystemInfo .sys-content {
        width: 100%;
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        self.border_title = "System"
        yield Static(classes="sys-content")

    def update_info(self, res: dict, term_w: int) -> None:
        wide = term_w >= 80

        cpu = res.get("cpu_pct", 0)
        cpu_color = "green" if cpu < 50 else ("yellow" if cpu < 80 else "red")
        ram_used = res.get("ram_used", 0)
        ram_total = res.get("ram_total", 1)
        ram_ratio = ram_used / ram_total if ram_total else 0
        home_used = res.get("home_used", 0)
        home_total = res.get("home_total", 1)
        home_ratio = home_used / home_total if home_total else 0
        home_color = "green" if home_ratio < 0.7 else ("yellow" if home_ratio < 0.9 else "bold red")
        scratch_used = res.get("scratch_used", 0)
        scratch_total = res.get("scratch_total", 1)
        scratch_ratio = scratch_used / scratch_total if scratch_total else 0

        lines = Text()

        if wide:
            # 2-column layout: left (CPU, /home) | right (RAM, /scratch)
            # inner width = term_w - 4 (border), half_w for each column
            inner_w = term_w - 4
            half_w = inner_w // 2
            # label(10) + bar + value(~15) => bar = half_w - 25
            bw = max(half_w - 25, 6)

            # Row 1: CPU | RAM
            left = f" CPU      "
            right = f"RAM      "
            cpu_val = f" {cpu:.0f}%"
            ram_val = f" {format_bytes(ram_used)}/{format_bytes(ram_total)}"
            left_pad = half_w - len(left) - bw - len(cpu_val)
            right_pad = half_w - len(right) - bw - len(ram_val) - 1

            lines.append(left, style="dim")
            lines.append_text(make_bar(cpu / 100, bw))
            lines.append(cpu_val, style=f"bold {cpu_color}")
            lines.append(" " * max(left_pad, 1))
            lines.append(right, style="dim")
            lines.append_text(make_bar(ram_ratio, bw))
            lines.append(ram_val + "\n", style="white")

            # Row 2: /home | /scratch
            left = f" /home    "
            right = f"/scratch "
            home_val = f" {format_bytes(home_used)}/{format_bytes(home_total)}"
            scratch_val = f" {format_bytes(scratch_used)}/{format_bytes(scratch_total)}"
            left_pad = half_w - len(left) - bw - len(home_val)

            lines.append(left, style="dim")
            lines.append_text(make_bar(home_ratio, bw))
            lines.append(home_val, style=home_color)
            lines.append(" " * max(left_pad, 1))
            lines.append(right, style="dim")
            lines.append_text(make_bar(scratch_ratio, bw))
            lines.append(scratch_val + "\n", style="white")
        else:
            # inner = term_w - 4 (border), label=10, bar, space+value
            # find longest value to size bar correctly
            vals = [
                f" {cpu:.0f}%",
                f" {format_bytes(ram_used)}/{format_bytes(ram_total)}",
                f" {format_bytes(home_used)}/{format_bytes(home_total)}",
                f" {format_bytes(scratch_used)}/{format_bytes(scratch_total)}",
            ]
            max_val = max(len(v) for v in vals)
            bw = max(term_w - 4 - 10 - max_val - 1, 4)

            lines.append(" CPU      ", style="dim")
            lines.append_text(make_bar(cpu / 100, bw))
            lines.append(f"{vals[0]}\n", style=f"bold {cpu_color}")

            lines.append(" RAM      ", style="dim")
            lines.append_text(make_bar(ram_ratio, bw))
            lines.append(f"{vals[1]}\n", style="white")

            lines.append(" /home    ", style="dim")
            lines.append_text(make_bar(home_ratio, bw))
            lines.append(f"{vals[2]}\n", style=home_color)

            lines.append(" /scratch ", style="dim")
            lines.append_text(make_bar(scratch_ratio, bw))
            lines.append(f"{vals[3]}\n", style="white")

        self.query_one(".sys-content", Static).update(lines)


# ── Tmux Sessions widget ─────────────────────────────────────────────────────

class TmuxSessions(Vertical):

    DEFAULT_CSS = """
    TmuxSessions {
        width: 100%;
        height: auto;
        border: round $secondary;
        margin: 0 0 1 0;
        padding: 0;
    }
    TmuxSessions .tmux-content {
        width: 100%;
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        self.border_title = "Sessions"
        yield Static(classes="tmux-content")

    def update_sessions(self, term_w: int = 100) -> None:
        sessions = get_tmux_sessions()
        content = self.query_one(".tmux-content", Static)
        if not sessions:
            content.update("")
            self.display = False
            return

        self.display = True
        wide = term_w >= 80
        name_w = 20 if wide else 12
        lines = Text()
        for sess in sessions:
            state = sess["state"]
            if state == "running":
                icon = "●"
                icon_color = "green"
                tag = "RUN"
                tag_style = "bold green"
            elif state == "pending":
                icon = "◌"
                icon_color = "yellow"
                tag = "WAIT"
                tag_style = "bold yellow"
            else:
                icon = "○"
                icon_color = "dim"
                tag = "END"
                tag_style = "dim red"

            name_style = "bold white" if sess["attached"] else "white"
            lines.append(f" {icon} ", style=icon_color)
            lines.append(f"{sess['name']:<{name_w}}", style=name_style)
            gpu_str = f"GPU {sess['gpu_idx']}" if sess.get("gpu_idx") else ""
            # Fixed-width columns: tag(4) + gpu(8) = 12 chars
            lines.append(f" {tag:<4}", style=tag_style)
            lines.append(f" {gpu_str:<7}", style="cyan")
            lines.append(f" {sess['created']}", style="dim")
            if wide:
                lines.append(f"   tmux attach -t {sess['name']}", style="dim italic")
            lines.append("\n")

        content.update(lines)


# ── Main App ─────────────────────────────────────────────────────────────────

NUM_GPUS = len(get_gpu_info()) or 4


class GpuMonitorApp(App):

    CSS = """
    Screen {
        background: $background;
    }
    #header-bar {
        dock: top;
        height: 1;
        width: 100%;
        background: $primary-background;
        padding: 0 1;
    }
    #gpu-grid {
        width: 100%;
        height: 1fr;
        padding: 0 1;
    }
    .gpu-row {
        width: 100%;
        height: auto;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        width: 100%;
        background: $primary-background;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("n", "new_session", "New Session"),
        ("h", "help", "Help"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.hostname_val = get_hostname()
        self.gpu_cards: list[GpuCard] = []
        self._timer: Timer | None = None
        self._status_msg = ""
        self._free_count = 0

    def compose(self) -> ComposeResult:
        yield Static(id="header-bar")
        with VerticalScroll(id="gpu-grid"):
            for row_start in range(0, NUM_GPUS, 2):
                with Horizontal(classes="gpu-row"):
                    card1 = GpuCard(row_start)
                    self.gpu_cards.append(card1)
                    yield card1
                    if row_start + 1 < NUM_GPUS:
                        card2 = GpuCard(row_start + 1)
                        self.gpu_cards.append(card2)
                        yield card2
            yield SystemInfo(id="sys-info")
            yield TmuxSessions(id="tmux-sessions")
            yield PendingJobs(id="pending-jobs")
        yield Static(id="status-bar")

    def on_mount(self) -> None:
        self._do_refresh()
        self._timer = self.set_interval(1.0, self._do_refresh)

    def _do_refresh(self) -> None:
        renamed = rename_pending_tmux()
        if renamed:
            self._status_msg = f"Session: {', '.join(renamed)}"
            self.set_timer(5.0, self._clear_status)
        gpus = get_gpu_info()
        processes = get_gpu_processes()
        slurm_allocs = get_slurm_gpu_allocations(self.hostname_val)

        term_w = self.size.width
        card_w = max((term_w - 6) // 2, 30) if NUM_GPUS > 1 else max(term_w - 4, 30)
        bar_w = max(card_w - 24, 8)

        active = allocated = 0
        for gpu in gpus:
            idx = gpu["index"]
            gpu_procs = processes.get(idx, [])
            slurm = slurm_allocs.get(idx)
            if gpu_procs:
                active += 1
            elif slurm:
                allocated += 1
            if idx < len(self.gpu_cards):
                self.gpu_cards[idx].update_card(gpu, gpu_procs, slurm, bar_w)

        free = len(gpus) - active - allocated
        self._free_count = free
        now = datetime.now().strftime("%H:%M:%S")

        # Header
        h = Text()
        h.append(f" {self.hostname_val}", style="bold")
        h.append(f"  {now}", style="dim")
        h.append("   ")
        h.append(f"● {active} active", style="green")
        h.append("  ")
        h.append(f"● {allocated} alloc", style="yellow")
        h.append("  ")
        h.append(f"○ {free} free", style="bright_green")

        if term_w >= 90:
            # Right-align lab name
            left_len = len(h.plain)
            lab = "Emory Vision & Graphics Lab"
            pad = max(term_w - left_len - len(lab) - 2, 2)
            h.append(" " * pad)
            h.append(lab, style="dim italic")

        self.query_one("#header-bar", Static).update(h)

        # Pending jobs widget
        pending = get_pending_jobs()
        self.query_one("#pending-jobs", PendingJobs).update_pending(pending)

        # System info
        sys_res = get_system_resources()
        self.query_one("#sys-info", SystemInfo).update_info(sys_res, term_w)

        # Tmux sessions
        self.query_one("#tmux-sessions", TmuxSessions).update_sessions(term_w)

        # Status bar
        s = Text()
        s.append(" q", style="bold")
        s.append(" Quit  ", style="dim")
        s.append("r", style="bold")
        s.append(" Refresh  ", style="dim")
        s.append("n", style="bold")
        s.append(" New  ", style="dim")
        s.append("h", style="bold")
        s.append(" Help", style="dim")
        if self._status_msg:
            s.append(f"   {self._status_msg}", style="bold yellow")
        self.query_one("#status-bar", Static).update(s)

    def _cancel_job_and_ask_tmux(self, job_id: str) -> None:
        """Cancel SLURM job, then ask about tmux if one exists."""
        tmux_sess = find_tmux_for_job(job_id)
        result = cancel_slurm_job(job_id)
        self._status_msg = result
        self._do_refresh()

        if tmux_sess:
            def handle_tmux(kill: bool | None) -> None:
                if kill:
                    msg = kill_tmux_session(tmux_sess)
                    self._status_msg = msg
                else:
                    self._status_msg = f"tmux '{tmux_sess}' kept."
                self.set_timer(5.0, self._clear_status)

            self.push_screen(
                ConfirmTmuxScreen(tmux_sess),
                callback=handle_tmux,
            )
        else:
            self.set_timer(5.0, self._clear_status)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""

        # Cancel pending job button
        if btn_id.startswith("cancel-pending-"):
            job_id = btn_id.replace("cancel-pending-", "")

            def handle_pending_cancel(confirmed: bool | None) -> None:
                if confirmed:
                    self._cancel_job_and_ask_tmux(job_id)

            self.push_screen(
                ConfirmCancelScreen(job_id, -1),
                callback=handle_pending_cancel,
            )
            return

        # Cancel running job button (on GPU card)
        card = event.button.parent
        while card and not isinstance(card, GpuCard):
            card = card.parent
        if not card or not isinstance(card, GpuCard) or not card.job_id:
            return

        target_card = card

        def handle_confirm(confirmed: bool | None) -> None:
            if confirmed and target_card.job_id:
                self._cancel_job_and_ask_tmux(target_card.job_id)

        self.push_screen(
            ConfirmCancelScreen(target_card.job_id, target_card.gpu_idx),
            callback=handle_confirm,
        )

    def action_new_session(self) -> None:
        def handle_launch(result: dict | None) -> None:
            if result:
                msg = launch_srun(result["num_gpus"], result["partition"], result["mem"])
                self._status_msg = msg
                self._do_refresh()
                self.set_timer(8.0, self._clear_status)

        self.push_screen(
            NewSessionScreen(self._free_count),
            callback=handle_launch,
        )

    def _clear_status(self) -> None:
        self._status_msg = ""

    def action_refresh(self) -> None:
        self._do_refresh()

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_quit(self) -> None:
        self.exit()


def main():
    app = GpuMonitorApp()
    app.run()


if __name__ == "__main__":
    main()
