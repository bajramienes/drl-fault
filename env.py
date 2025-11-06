import os
import csv
import psutil
import random
import time
import datetime
import shutil
import types
import sys
import logging

# -----------------------------------------------------------------------------
# Optional GPU metrics
# -----------------------------------------------------------------------------
try:
    import GPUtil
    HAS_GPU = True
except Exception:
    GPUtil = None
    HAS_GPU = False

# Python 3.12+ compatibility
sys.modules['distutils'] = types.ModuleType('distutils')
sys.modules['distutils.spawn'] = shutil

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
LOG_PATH = os.path.join(os.getcwd(), "log.txt")
logger = logging.getLogger("drl-fault")
logger.setLevel(logging.INFO)

if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

# -----------------------------------------------------------------------------
# Results directory
# -----------------------------------------------------------------------------
RESULTS_ROOT = os.path.join(os.getcwd(), "results")
os.makedirs(RESULTS_ROOT, exist_ok=True)

def phase_dir(phase_name: str) -> str:
    p = os.path.join(RESULTS_ROOT, phase_name)
    os.makedirs(p, exist_ok=True)
    return p

# -----------------------------------------------------------------------------
# Metrics collection
# -----------------------------------------------------------------------------
def _gpu_metrics():
    """Collect GPU utilization if available."""
    if not HAS_GPU:
        return (None, None)
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return (None, None)
        util = float(sum(g.load for g in gpus) / len(gpus)) * 100.0
        mem = float(sum((g.memoryUtil or 0.0) for g in gpus) / len(gpus)) * 100.0
        return (round(util, 2), round(mem, 2))
    except Exception:
        return (None, None)

def get_system_metrics(net_last=None):
    """Collect CPU, memory, disk, and network metrics."""
    cpu = psutil.cpu_percent(interval=0.3)
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent if os.name != "nt" else psutil.disk_usage(os.getcwd().split(":")[0] + ":\\").percent

    net = psutil.net_io_counters()
    sent_mb = recv_mb = None
    if net_last is not None:
        sent_mb = round((net.bytes_sent - net_last[0]) / (1024 * 1024), 4)
        recv_mb = round((net.bytes_recv - net_last[1]) / (1024 * 1024), 4)

    gpu_util, gpu_mem = _gpu_metrics()

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "cpu_percent": round(cpu, 2),
        "mem_percent": round(mem, 2),
        "disk_percent": round(disk, 2),
        "net_bytes_sent": net.bytes_sent,
        "net_bytes_recv": net.bytes_recv,
        "delta_sent_mb": sent_mb,
        "delta_recv_mb": recv_mb,
        "gpu_util_percent": gpu_util,
        "gpu_mem_percent": gpu_mem,
    }

# -----------------------------------------------------------------------------
# CSV utilities
# -----------------------------------------------------------------------------
EPISODE_HEADER = [
    "timestamp", "phase", "agent", "episode", "is_train",
    "request_mb", "processing_ms",
    "cpu_percent", "mem_percent", "disk_percent",
    "delta_sent_mb", "delta_recv_mb",
    "gpu_util_percent", "gpu_mem_percent",
    "fault_type", "agent_start", "agent_end", "agent_elapsed_sec"
]

SUMMARY_HEADER = [
    "phase", "agent", "agent_start", "agent_end", "agent_elapsed_sec",
    "episodes_total", "train_episodes", "test_episodes"
]

def init_csv(csv_path, header):
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

def append_row(csv_path, row):
    """Append and flush after every write to avoid data loss."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.flush()

# -----------------------------------------------------------------------------
# Workload and fault simulation
# -----------------------------------------------------------------------------
def process_request(request_mb: int) -> float:
    """Simulate a request processing workload."""
    start = time.perf_counter()

    # CPU work proportional to request size
    work_units = request_mb * 20000
    acc = 0.0
    for _ in range(work_units):
        acc += random.random() * random.random()
    _ = acc  # prevent optimization

    # Light memory footprint (scaled)
    _buf = bytearray(request_mb * 128 * 1024)  # ~128KB per MB

    # Simulate I/O latency
    time.sleep(min(0.004 * request_mb, 0.08))

    end = time.perf_counter()
    return round((end - start) * 1000.0, 3)

def simulate_faults(probability=0.08):
    """Inject random synthetic faults."""
    if random.random() >= probability:
        return None

    fault = random.choice([
        "cpu_spike",
        "memory_spike",
        "network_jitter",
        "io_pause",
        "disk_throttle"
    ])

    try:
        if fault == "cpu_spike":
            t_end = time.perf_counter() + 0.3
            while time.perf_counter() < t_end:
                _ = random.random() * random.random()
        elif fault == "memory_spike":
            junk = [bytearray(2_000_000) for _ in range(8)]
            del junk
        elif fault == "network_jitter":
            time.sleep(random.uniform(0.08, 0.25))
        elif fault == "io_pause":
            time.sleep(random.uniform(0.1, 0.3))
        elif fault == "disk_throttle":
            _ = [os.urandom(1024 * 50) for _ in range(20)]
        logger.info(f"[FAULT] {fault} injected")
    except Exception as e:
        logger.warning(f"[FAULT ERROR] {e}")

    return fault
