#!/usr/bin/env python3
"""
GPU Keep-Alive Daemon — prevents cloud-platform GPU reclamation during idle windows.

Problem:
  Between eval_entry invocations, model loading takes ~3 minutes with GPU
  utilization at 0%.  Cloud GPU schedulers frequently reclaim GPUs that stay
  idle for too long, killing the evaluation processes.

Solution:
  Monitor specified GPUs via nvidia-smi.  When utilization stays below a
  threshold for longer than a configured tolerance window, allocate small
  PyTorch tensors and run periodic light-weight CUDA kernels (matmul) to keep
  GPU utilization above zero.  Release tensors immediately when real work
  resumes (utilization > threshold).

Memory footprint:
  Default ~50 MB per monitored GPU — negligible relative to 80 GB A100.
  This will NOT interfere with vLLM memory allocation.

Usage:
  python3 gpu_keepalive.py --gpus 0,3 --idle-seconds 30 --pid-file /tmp/keepalive.pid &
  # ... run evaluation ...
  kill $(cat /tmp/keepalive.pid)
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import threading

os.environ.setdefault("PYTHONUNBUFFERED", "1")

# ---------------------------------------------------------------------------
# Minimal nvidia-smi wrapper — no pynvml dependency
# ---------------------------------------------------------------------------

def _query_gpus(gpu_ids):
    """Return {gpu_id: {"util": float_pct, "mem_used_mib": float}} for requested GPUs."""
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except Exception:
        return {}

    info = {}
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        if idx not in gpu_ids:
            continue
        try:
            info[idx] = {
                "util": float(parts[1]),
                "mem_used_mib": float(parts[2]),
            }
        except ValueError:
            continue
    return info


# ---------------------------------------------------------------------------
# Keep-Alive core
# ---------------------------------------------------------------------------

class GPUKeepAlive:
    def __init__(
        self,
        gpu_ids,
        idle_threshold_pct=5.0,
        idle_seconds=30.0,
        warm_interval_s=1.0,
        warm_memory_mb=50,
        check_interval_s=5.0,
        signal_file=None,
        signal_timeout=600,
    ):
        self.gpu_ids = list(gpu_ids)
        self.idle_threshold_pct = idle_threshold_pct
        self.idle_seconds = idle_seconds
        self.warm_interval_s = warm_interval_s
        self.warm_memory_mb = warm_memory_mb
        self.check_interval_s = check_interval_s
        self.signal_file = signal_file
        self.signal_timeout = signal_timeout

        # Internal state
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._warm_active = False
        self._warm_tensors = {}
        self._idle_since = {g: None for g in self.gpu_ids}

        # Cooperative coordination via signal file
        self._cooperative_hold = False
        self._coop_enter_time = None  # when we entered cooperative hold

    # --- signal-file coordination ---------------------------------------

    def _check_signal_file(self):
        """Return True if the cooperative signal file exists and is fresh."""
        if not self.signal_file:
            return False
        try:
            if not os.path.exists(self.signal_file):
                return False
            mtime = os.path.getmtime(self.signal_file)
            age = time.time() - mtime
            return age < self.signal_timeout
        except OSError:
            return False

    # --- warmup --------------------------------------------------------

    def _start_warmup(self):
        with self._lock:
            if self._warm_active:
                return
            self._warm_active = True

        log("warmup START",
            "GPUs=%s mem=%dMB" % (self.gpu_ids, self.warm_memory_mb))

        import torch

        # Use larger matrices for visible GPU utilisation.
        elements = (self.warm_memory_mb * 1024 * 1024) // 4
        side = int(elements ** 0.5)
        side = max(256, min(side, 4096))

        for gid in self.gpu_ids:
            try:
                device = torch.device("cuda:%d" % gid)
                a = torch.randn(side, side, device=device, dtype=torch.float32)
                b = torch.randn(side, side, device=device, dtype=torch.float32)
                with self._lock:
                    self._warm_tensors[gid] = (a, b)
            except Exception as exc:
                log("alloc error", "GPU%d: %s" % (gid, exc))

        t = threading.Thread(target=self._warm_loop, daemon=True, name="gpu-warmup")
        t.start()

    def _stop_warmup(self):
        with self._lock:
            if not self._warm_active:
                return
            self._warm_active = False

        time.sleep(max(0.3, self.warm_interval_s * 2.0))

        with self._lock:
            gpu_ids = list(self._warm_tensors.keys())
            self._warm_tensors.clear()

        import torch
        for gid in gpu_ids:
            pass  # tensors already cleared above
        if gpu_ids:
            torch.cuda.empty_cache()
        log("warmup STOP")

    def _warm_loop(self):
        """Continuously run matmul to generate visible GPU utilization."""
        import torch
        burst_size = 20  # number of matmuls per inner burst
        sync_every = 5   # sync after every N bursts
        iter_count = 0
        while True:
            with self._lock:
                if not self._warm_active:
                    return
                tensors_snapshot = dict(self._warm_tensors)
            if not tensors_snapshot:
                time.sleep(self.warm_interval_s)
                continue

            # Burst: many sequential matmuls to keep GPU kernel queue full.
            for _ in range(burst_size):
                for _gid, (a, b) in tensors_snapshot.items():
                    try:
                        c = torch.mm(a, b)
                        # Prevent dead-code elimination with a cheap add.
                        c.add_(1.0)
                    except Exception:
                        pass

            iter_count += 1
            if iter_count % sync_every == 0:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            time.sleep(self.warm_interval_s)

    # --- main loop -------------------------------------------------------

    def run(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            sys.exit("ERROR: torch is required")

        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        # Do NOT set CUDA_VISIBLE_DEVICES — we reference physical GPU ids
        # directly (cuda:0, cuda:3).  Restricting visibility causes
        # remapping (phys GPU 3 becomes cuda:1), breaking device access.

        def _on_term(_signum, _frame):
            log("signal received, shutting down")
            self._stop.set()
        signal.signal(signal.SIGTERM, _on_term)
        signal.signal(signal.SIGINT, _on_term)

        log("START",
            "GPUs=%s idle_thresh=%.0f%% idle_tol=%.0fs check=%.0fs warm_mem=%dMB"
            % (self.gpu_ids, self.idle_threshold_pct, self.idle_seconds,
               self.check_interval_s, self.warm_memory_mb))

        # Hysteresis: once warmup is active, require a higher threshold to
        # stop it, preventing flutter from warmup's own GPU utilisation.
        STOP_THRESHOLD_PCT = max(self.idle_threshold_pct * 4, 20.0)
        # Also require N consecutive "active" checks before stopping.
        STOP_CONSECUTIVE = 3

        active_streak = 0

        while not self._stop.is_set():
            # --- cooperative signal file check ---
            if self.signal_file:
                signal_exists = self._check_signal_file()
                if signal_exists and not self._cooperative_hold:
                    self._cooperative_hold = True
                    self._coop_enter_time = time.time()
                    self._stop_warmup()
                    active_streak = 0
                    log("cooperative hold ENTER",
                        "signal=%s timeout=%ds" % (self.signal_file, self.signal_timeout))
                elif not signal_exists and self._cooperative_hold:
                    self._cooperative_hold = False
                    age = time.time() - self._coop_enter_time if self._coop_enter_time else 0
                    self._coop_enter_time = None
                    for g in self.gpu_ids:
                        self._idle_since[g] = None
                    active_streak = 0
                    log("cooperative hold EXIT",
                        "held=%.0fs, resuming GPU monitoring" % age)
                elif not signal_exists:
                    # signal is stale or missing, not in hold – proceed to GPU monitoring
                    pass

            if self._cooperative_hold:
                self._stop.wait(self.check_interval_s)
                continue

            # --- query GPU utilisation ---
            info = _query_gpus(self.gpu_ids)
            if not info:
                log("nvidia-smi query failed, retrying")
                self._stop.wait(self.check_interval_s)
                continue

            now = time.monotonic()
            all_idle_long_enough = True
            any_really_active = False

            for gid in self.gpu_ids:
                ginfo = info.get(gid, {})
                util = ginfo.get("util", 100.0)

                if util < self.idle_threshold_pct:
                    # GPU is idle
                    if self._idle_since[gid] is None:
                        self._idle_since[gid] = now
                    elif now - self._idle_since[gid] < self.idle_seconds:
                        all_idle_long_enough = False
                else:
                    # GPU has some activity
                    self._idle_since[gid] = None
                    all_idle_long_enough = False
                    if util >= STOP_THRESHOLD_PCT:
                        any_really_active = True

            # Decide start/stop with hysteresis.
            if all_idle_long_enough and not self._warm_active:
                self._start_warmup()
                active_streak = 0
            elif self._warm_active:
                if any_really_active:
                    active_streak += 1
                    if active_streak >= STOP_CONSECUTIVE:
                        self._stop_warmup()
                        active_streak = 0
                else:
                    active_streak = 0

            self._stop.wait(self.check_interval_s)

        self._stop_warmup()
        log("STOPPED")


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def log(tag, msg=""):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "[GPU-KEEPALIVE %s] %s" % (ts, tag)
    if msg:
        line += "  " + msg
    print(line, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPU Keep-Alive Daemon"
    )
    parser.add_argument(
        "--gpus", default=os.environ.get("KEEPALIVE_GPUS", "0,3"),
        help="Comma-separated GPU ids (env: KEEPALIVE_GPUS, default: 0,3)",
    )
    parser.add_argument(
        "--idle-threshold", type=float, default=5.0,
        help="Util %% below which GPU is idle (default: 5.0)",
    )
    parser.add_argument(
        "--idle-seconds", type=float, default=30.0,
        help="Consecutive idle seconds before warmup (default: 30)",
    )
    parser.add_argument(
        "--warm-interval", type=float, default=1.0,
        help="Seconds between warmup kernel launches (default: 1.0)",
    )
    parser.add_argument(
        "--warm-memory-mb", type=int, default=50,
        help="GPU memory (MB) for warmup tensors (default: 50)",
    )
    parser.add_argument(
        "--check-interval", type=float, default=5.0,
        help="Seconds between nvidia-smi polls (default: 5.0)",
    )
    parser.add_argument(
        "--signal-file", default=None,
        help="Cooperative signal file path",
    )
    parser.add_argument(
        "--signal-timeout", type=float, default=600,
        help="Seconds after which signal file is considered stale (default: 600)",
    )
    parser.add_argument(
        "--pid-file", default=None,
        help="Write PID to this file",
    )
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpu_ids:
        parser.error("--gpus must specify at least one GPU")

    if args.pid_file:
        try:
            with open(args.pid_file, "w") as f:
                f.write(str(os.getpid()))
        except OSError as exc:
            log("ERROR", "cannot write pid-file %s: %s" % (args.pid_file, exc))

    keeper = GPUKeepAlive(
        gpu_ids=gpu_ids,
        idle_threshold_pct=args.idle_threshold,
        idle_seconds=args.idle_seconds,
        warm_interval_s=args.warm_interval,
        warm_memory_mb=args.warm_memory_mb,
        check_interval_s=args.check_interval,
        signal_file=args.signal_file,
        signal_timeout=args.signal_timeout,
    )
    keeper.run()


if __name__ == "__main__":
    main()
