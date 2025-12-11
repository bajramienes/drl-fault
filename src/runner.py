# src/runner.py — FINAL VERSION (DQN WORKS!)
import os
import csv
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import PHASES, AGENTS, RESULTS_ROOT, MAX_TRAIN_STEPS, MAX_TEST_STEPS
from .envs.container_env import HybridContainerEnv   

ALGOS = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "DDPG": DDPG, "A2C": A2C, "DQN": DQN}

COLUMNS = [
    "date_eu","time","timestamp","agent","phase","episode_index","step_in_episode","action_taken",
    "request_mb","fault_type","latency_ms","cpu_percent","mem_percent","disk_percent",
    "energy_joule","energy_method","energy_efficiency","host_cpu_percent","host_mem_percent","total_reward"
]

def log(msg):
    print(f"[{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}] {msg}")

def get_last_episode(csv_path):
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return int(df['episode_index'].max()) + 1 if not df.empty else 0
        except:
            return 0
    return 0

def train_and_test_real(agent: str, phase: dict):
    csv_path = f"{RESULTS_ROOT}/episodes/{agent}_{phase['name']}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    last_ep = get_last_episode(csv_path)
    total_eps = phase["train_episodes"] + phase["test_episodes"]
    episode = last_ep

    with open(csv_path, "a" if last_ep > 0 else "w", newline="", encoding="utf-8") as f:
        if last_ep == 0:
            csv.writer(f).writerow(COLUMNS)

    log(f"PHASE START → {agent.upper()} | {phase['name']} | Ep {last_ep+1}–{total_eps} (REAL DOCKER EVALUATION)")

    phase_start = time.time()

    # ONE ENV CLASS FOR ALL AGENTS (DQN handled inside)
    env = DummyVecEnv([lambda: HybridContainerEnv(agent, phase["name"], seed=42, is_training=True)])
    model = ALGOS[agent]("MlpPolicy", env, verbose=0, device="cpu", seed=42)

    while episode < total_eps:
        model.learn(total_timesteps=MAX_TRAIN_STEPS, reset_num_timesteps=False)

        test_env = HybridContainerEnv(agent, phase["name"], seed=episode + 10000, is_training=False)
        obs, _ = test_env.reset()
        total_r = 0.0

        for step in range(MAX_TEST_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, _, truncated, info = test_env.step(action)
            total_r += r

            now = time.time()
            dt = datetime.fromtimestamp(now)
            info.update({
                "agent": agent, "phase": phase["name"],
                "episode_index": episode, "step_in_episode": step,
                "total_reward": round(total_r, 3), "timestamp": now,
                "date_eu": dt.strftime("%d-%m-%Y"), "time": dt.strftime("%H:%M:%S")
            })

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([info.get(c, "") for c in COLUMNS])

            if truncated: break

        test_env.close()
        episode += 1

        if (episode - last_ep) % 10 == 0 or episode <= last_ep + 5 or episode == total_eps:
            mins = int((time.time() - phase_start) / 60)
            log(f"  Ep {episode:4d}/{total_eps} | Reward: {total_r:+8.2f} | Time: {mins:4d} min")

    env.close()
    log(f"PHASE END → {agent.upper()} | {phase['name']}\n")

def main():
    log("FINAL RUN — DQN FIXED, WILL FINISH TODAY")
    log(f"Start: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    start_all = time.time()

    for agent in AGENTS:
        log(f"\nAGENT START → {agent.upper()}")
        for phase in PHASES:
            train_and_test_real(agent, phase)
        log(f"AGENT END → {agent.upper()}")

    hours = round((time.time() - start_all) / 3600, 1)
    log(f"\nYOU ARE DONE! Total time: {hours} hours")
    log(f"All 18 18 CSV files are in: {os.path.abspath(RESULTS_ROOT + '/episodes')}")

if __name__ == "__main__":
    main()