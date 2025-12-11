import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = r"C:\Users\User\Desktop\drl_fault\results\baseline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPISODES = {
    "early": 400,
    "mid": 600,
    "final": 1000
}

STEPS_PER_EPISODE = 20  # you can change


# ============================================================
# BASELINE ACTION POLICY (RULE-BASED)
# ============================================================

def baseline_action(cpu, fault):
    if fault != "none":
        return 0  # migrate immediately
    if cpu > 80:
        return 2  # scale
    if cpu < 25:
        return 0  # migrate
    return 1  # noop


# ============================================================
# SYSTEM TRANSITION MODEL (REALISTIC ENGINEERING SIMULATION)
# ============================================================

def simulate_step(cpu, mem, disk, latency, fault, action):
    # ---- FAULT IMPACT ----
    if fault != "none":
        latency *= np.random.uniform(1.3, 1.7)
        cpu *= np.random.uniform(1.1, 1.2)

    # ---- ACTION EFFECTS ----
    if action == 0:  # migrate
        latency *= np.random.uniform(1.2, 1.4)
        cpu *= np.random.uniform(0.9, 1.0)
        mem *= np.random.uniform(0.95, 1.05)

    elif action == 2:  # scale
        latency *= np.random.uniform(0.6, 0.8)
        cpu *= np.random.uniform(1.05, 1.15)
        mem *= np.random.uniform(1.02, 1.08)

    elif action == 1:  # noop
        latency *= np.random.uniform(0.95, 1.05)
        cpu *= np.random.uniform(0.98, 1.02)
        mem *= np.random.uniform(0.98, 1.02)

    # ---- BOUNDING ----
    cpu = np.clip(cpu, 5, 100)
    mem = np.clip(mem, 5, 100)
    disk = np.clip(disk + np.random.uniform(-0.5, 0.5), 1, 100)
    latency = np.clip(latency, 1, 500)

    energy = latency * cpu * 0.0015  # synthetic but realistic
    efficiency = 1 / (1 + latency)

    return cpu, mem, disk, latency, energy, efficiency


# ============================================================
# FAULT GENERATOR
# ============================================================

FAULT_TYPES = ["none", "cpu_fault", "latency_fault", "disk_fault"]

def sample_fault():
    p = np.random.rand()
    if p < 0.85:
        return "none"
    elif p < 0.90:
        return "cpu_fault"
    elif p < 0.95:
        return "latency_fault"
    else:
        return "disk_fault"


# ============================================================
# REWARD FUNCTION
# ============================================================

def compute_reward(latency, fault, action):
    r = -latency
    if fault != "none":
        r -= 20
    if action == 2:  # scaling costs
        r -= 3
    return r


# ============================================================
# SIMULATE BASELINE EPISODES
# ============================================================

def simulate_phase(phase_name, num_episodes):
    rows = []
    base_time = datetime(2025, 12, 8, 7, 0, 0)

    for ep in range(num_episodes):
        # initial random state
        cpu = np.random.uniform(20, 60)
        mem = np.random.uniform(20, 60)
        disk = np.random.uniform(10, 50)
        latency = np.random.uniform(20, 80)

        for step in range(STEPS_PER_EPISODE):
            timestamp = base_time + timedelta(seconds=len(rows))

            fault = sample_fault()
            action = baseline_action(cpu, fault)

            cpu, mem, disk, latency, energy, efficiency = simulate_step(
                cpu, mem, disk, latency, fault, action
            )

            reward = compute_reward(latency, fault, action)

            rows.append({
                "date_eu": timestamp.strftime("%d-%m-%Y"),
                "time": timestamp.strftime("%H:%M:%S"),
                "timestamp": timestamp.timestamp(),
                "agent": "BASELINE",
                "phase": phase_name,
                "episode_index": ep,
                "step_in_episode": step,
                "action_taken": action,
                "request_mb": np.random.uniform(50, 300),
                "fault_type": fault,
                "latency_ms": latency,
                "cpu_percent": cpu,
                "mem_percent": mem,
                "disk_percent": disk,
                "energy_joule": energy,
                "energy_method": "simulated",
                "energy_efficiency": efficiency,
                "host_cpu_percent": cpu * np.random.uniform(0.8, 1.2),
                "host_mem_percent": mem * np.random.uniform(0.8, 1.2),
                "total_reward": reward
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, f"BASELINE_{phase_name}.csv"), index=False)
    print(f"[OK] Generated baseline for phase: {phase_name}")


# ============================================================
# MAIN
# ============================================================

def main():
    for phase, episodes in EPISODES.items():
        simulate_phase(phase, episodes)
    print("[DONE] Baseline Simulation Complete")


if __name__ == "__main__":
    main()
