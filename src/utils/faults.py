# src/utils/faults.py
import random

FAULTS = {
    "early": {"none": 0.7, "cpu_spike": 0.15, "mem_pressure": 0.10, "disk_stall": 0.05},
    "mid":   {"none": 0.5, "cpu_spike": 0.20, "mem_pressure": 0.15, "disk_stall": 0.08, "net_jitter": 0.07},
    "final": {"none": 0.3, "cpu_spike": 0.25, "mem_pressure": 0.20, "disk_stall": 0.12, "net_jitter": 0.13},
}

def sample_fault(phase: str, rng: random.Random):
    probs = FAULTS.get(phase, FAULTS["mid"])
    return rng.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]