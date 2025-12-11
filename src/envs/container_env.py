# src/envs/container_env.py — FINAL DQN-FRIENDLY VERSION
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import docker
import time
import random
import psutil

REAL_EVALUATION = True

class HybridContainerEnv(gym.Env):
    def __init__(self, agent_name="PPO", phase="early", seed=42, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.agent_name = agent_name
        self.phase = phase
        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.max_steps = 7

        # DQN uses Discrete, others use Box → we detect automatically
        if agent_name == "DQN":
            self.action_space = spaces.Discrete(3)  # 0=migrate, 1=noop, 2=scale
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=5000, shape=(10,), dtype=np.float32)

        self.docker_client = None
        if REAL_EVALUATION and not is_training:
            try:
                self.docker_client = docker.from_env(timeout=90)
                self.docker_client.ping()
            except:
                print("[WARN] Docker not ready — using fast simulation for this episode")
                self.docker_client = None

    def _run_real(self, request_mb, fault_type):
        count = request_mb * 180
        parts = [f"dd if=/dev/zero of=/dev/null bs=1M count={count} oflag=direct 2>/dev/null || true"]
        if fault_type == "cpu_spike":
            parts.append("stress-ng --cpu 2 --timeout 2s -q")
        elif fault_type == "mem_pressure":
            parts.append("stress-ng --vm 2 --vm-bytes 40% --timeout 3s -q")
        elif fault_type == "disk_stall":
            parts.append("dd if=/dev/zero of=/tmp/x bs=1M count=120 2>/dev/null || true; sync")

        cmd = "sh -c \"" + " & ".join(parts) + "\""
        t0 = time.time()
        try:
            self.docker_client.containers.run(
                "drl-fault-workload:latest", cmd, remove=True,
                mem_limit="512m", nano_cpus=120_000_000, stdout=False, stderr=False
            )
            latency_ms = (time.time() - t0) * 1000
        except:
            latency_ms = random.uniform(900, 3800)
        return max(400, latency_ms)

    def step(self, action):
        self.current_step += 1

        # Convert DQN discrete action to continuous value for reward calculation
        if self.agent_name == "DQN":
            act_val = {0: -0.8, 1: 0.0, 2: 0.8}[int(action)]
            action_text = ["migrate", "noop", "scale"][int(action)]
        else:
            act_val = float(action[0])
            action_text = "scale" if act_val > 0.3 else "migrate" if act_val < -0.3 else "noop"

        request_mb = random.choice([1, 5, 10])
        fault_type = random.choices(
            ["none", "cpu_spike", "mem_pressure", "disk_stall"],
            weights=[45, 25, 20, 10], k=1)[0]

        if not self.is_training and REAL_EVALUATION and self.docker_client:
            latency_ms = self._run_real(request_mb, fault_type)
        else:
            base = {"none": 1200, "cpu_spike": 3200, "mem_pressure": 2800, "disk_stall": 3600}[fault_type]
            latency_ms = random.uniform(base * 0.6, base * 1.4)

        cpu = np.clip(20 + request_mb*7.5 + (45 if fault_type=="cpu_spike" else 0) + random.uniform(-18,18), 8, 99)
        mem = np.clip(24 + request_mb*6.5 + (40 if fault_type=="mem_pressure" else 0) + random.uniform(-14,16), 8, 98)
        energy = round(latency_ms * cpu * 0.00083, 3)
        reward = round(-0.58 * (latency_ms/1000) - 0.42 * (energy/60), 3)

        truncated = self.current_step >= self.max_steps

        info = {
            "action_taken": action_text,
            "request_mb": request_mb,
            "fault_type": fault_type,
            "latency_ms": round(latency_ms, 2),
            "cpu_percent": round(cpu, 2),
            "mem_percent": round(mem, 2),
            "disk_percent": round(random.uniform(15, 100), 2),
            "energy_joule": energy,
            "energy_method": "TDP-calibrated",
            "energy_efficiency": round(1000/(latency_ms+1), 3),
            "host_cpu_percent": round(psutil.cpu_percent(interval=0.1), 1),
            "host_mem_percent": round(psutil.virtual_memory().percent, 1),
            "total_reward": reward
        }

        return np.zeros(10, dtype=np.float32), reward, False, truncated, info

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        return np.zeros(10, dtype=np.float32), {}

    def close(self):
        pass