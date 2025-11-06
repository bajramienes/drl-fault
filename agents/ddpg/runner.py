# run.py
import os
import time
import random
import datetime
from env import log_metrics, simulate_faults, init_csv, RESULTS_DIR

agents = ["PPO", "SAC", "TD3", "DDPG", "A2C", "DQN", "MADDPG"]
phases = [
    {"name": "early", "train": 500, "test": 2000},
    {"name": "mid", "train": 1000, "test": 3000},
    {"name": "final", "train": 1500, "test": 4000},
]

def print_banner(title):
    print("\n" + "=" * 70)
    print(f"{title} | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

def run_phase(phase):
    csv_path = os.path.join(RESULTS_DIR, f"phase_{phase['name']}.csv")
    init_csv(csv_path)
    print_banner(f"[PHASE: {phase['name'].upper()}] STARTED")

    for agent in agents:
        print_banner(f"Agent: {agent}")
        for episode in range(phase["train"] + phase["test"]):
            request_size = random.randint(50, 500)
            log_metrics(csv_path, phase["name"], agent, episode, request_size)
            if episode % 300 == 0 and episode > 0:
                simulate_faults()
        print(f"[DONE] Agent {agent} finished {phase['name']} phase")

    print_banner(f"[PHASE {phase['name'].upper()}] COMPLETED")

def main():
    start = time.time()
    print_banner("DISTRIBUTED MULTI-AGENT EXPERIMENT INITIATED")
    for phase in phases:
        run_phase(phase)
    print_banner("ALL PHASES COMPLETED")
    duration = round((time.time() - start) / 60, 2)
    print(f"Total Experiment Duration: {duration} minutes")

if __name__ == "__main__":
    main()
