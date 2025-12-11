# src/config.py — FINAL: FAST TRAINING + REAL EVALUATION (36–48 hours total)
PHASES = [
    {"name": "early", "train_episodes": 200, "test_episodes": 400},   # 600 total
    {"name": "mid",   "train_episodes": 250, "test_episodes": 500},   # 750 total
    {"name": "final", "train_episodes": 300, "test_episodes": 600},   # 900 total
]

AGENTS = ["PPO", "SAC", "TD3", "DDPG", "A2C", "DQN"]
REQUEST_SIZES_MB = [1, 5, 10]
CPU_TDP_WATT = 65
MAX_TRAIN_STEPS = 7
MAX_TEST_STEPS = 7
RESULTS_ROOT = "results"
WORKLOAD_IMAGE = "drl-fault-workload:latest"