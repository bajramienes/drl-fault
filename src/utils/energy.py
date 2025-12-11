# src/utils/energy.py
from ..config import CPU_TDP_WATT

def measure_energy(cpu_percent, latency_ms, fault_type, action_level, request_mb):
    runtime_s = latency_ms / 1000.0
    power = CPU_TDP_WATT * (cpu_percent / 100.0) * 1.15  # +15% overhead
    joules = power * runtime_s
    efficiency = 1000.0 / (latency_ms + joules + 1e-6)
    return round(joules, 3), "TDP-calibrated", round(efficiency, 3)