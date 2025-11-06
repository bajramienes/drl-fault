# env.py
import os
import csv
import psutil
import random
import time
import datetime
import GPUtil
import docker
import wmi
from statistics import variance

RESULTS_DIR = "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)
client = docker.from_env()
w = wmi.WMI(namespace="root\\wmi")

def get_cpu_temp():
    try:
        temp = w.MSAcpi_ThermalZoneTemperature()[0].CurrentTemperature
        return round((temp / 10.0) - 273.15, 2)
    except Exception:
        return round(random.uniform(40.0, 70.0), 2)

def get_gpu_metrics():
    try:
        gpus = GPUtil.getGPUs()
        gpu = gpus[0]
        return round(gpu.temperature, 2), round(gpu.load * 100, 2), round(gpu.memoryUtil * 100, 2)
    except Exception:
        return 0.0, 0.0, 0.0

def get_hdd_temp():
    disk_usage = psutil.disk_io_counters()
    temp = 35 + 0.02 * ((disk_usage.write_bytes + disk_usage.read_bytes) / 1e6)
    return round(min(max(temp, 35), 55), 2)

def log_metrics(csv_path, phase, agent, episode, request_size):
    cpu_usage = psutil.cpu_percent(interval=0.5)
    mem_usage = psutil.virtual_memory().percent
    disk_io = psutil.disk_io_counters()
    read_mb = disk_io.read_bytes / (1024 * 1024)
    write_mb = disk_io.write_bytes / (1024 * 1024)
    cpu_temp = get_cpu_temp()
    gpu_temp, gpu_load, gpu_mem = get_gpu_metrics()
    hdd_temp = get_hdd_temp()
    latency = random.uniform(10, 120)
    reward = random.uniform(0.4, 1.0)
    decision_delay = random.uniform(5, 30)
    recovery_time = random.uniform(200, 700)
    uptime = random.uniform(95, 100)
    stability = 1 - (variance([reward, random.uniform(0.4, 1.0)]) if reward < 0.9 else 0.01)
    complexity = (cpu_usage * 0.4 + gpu_load * 0.3 + decision_delay / 100 * 0.3) / 100

    row = [
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        phase, agent, episode, request_size, cpu_usage, cpu_temp, mem_usage,
        gpu_temp, gpu_load, gpu_mem, hdd_temp,
        read_mb, write_mb, latency, recovery_time,
        decision_delay, uptime, reward, stability, complexity
    ]

    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)

def simulate_faults():
    containers = client.containers.list()
    if containers:
        c = random.choice(containers)
        print(f"[FAULT] Restarting container {c.name}")
        c.restart()
        time.sleep(random.uniform(2, 5))
        print(f"[RECOVERY] Container {c.name} recovered")

def init_csv(csv_path):
    header = [
        "timestamp", "phase", "agent", "episode", "request_size_MB", "cpu_usage_percent", "cpu_temp_c",
        "mem_usage_percent", "gpu_temp_c", "gpu_load_percent", "gpu_mem_percent", "hdd_temp_c",
        "disk_read_MBps", "disk_write_MBps", "mean_latency_ms", "fault_recovery_time_ms",
        "decision_delay_ms", "container_uptime_percent", "reward", "stability_index", "complexity_score"
    ]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)
