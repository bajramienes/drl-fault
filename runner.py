import os
import time
import random
import datetime
from env import (
    logger, RESULTS_ROOT, phase_dir,
    get_system_metrics, init_csv, append_row,
    process_request, simulate_faults,
    EPISODE_HEADER, SUMMARY_HEADER
)

# ---------------------------------------------------------------
# Sequential multi-agent execution (all agents in one container)
# ---------------------------------------------------------------
agents = ["PPO", "SAC", "TD3", "DDPG", "A2C", "DQN", "MADDPG"]

# ---------------------------------------------------------------
# Extended phases for multi-day testing (approx. 2–3 days runtime)
# ---------------------------------------------------------------
phases = [
    {"name": "early", "train": 4000, "test": 10000},
    {"name": "mid",   "train": 6000, "test": 15000},
    {"name": "final", "train": 8000, "test": 20000},
]

# Workload sizes in MB
REQUEST_MB_CHOICES = [1, 10, 22]


def print_banner(title):
    """Pretty banner for logs."""
    line = "=" * 74
    logger.info("")
    logger.info(line)
    logger.info(f"{title} | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(line)


def run_phase(phase):
    """Execute one full phase (train + test) across all agents sequentially."""
    phase_name = phase["name"]
    pdir = phase_dir(phase_name)

    episode_csv = os.path.join(pdir, f"episodes_{phase_name}.csv")
    summary_csv = os.path.join(pdir, f"summary_{phase_name}.csv")

    # Create CSV headers
    init_csv(episode_csv, EPISODE_HEADER)
    if not os.path.exists(summary_csv):
        init_csv(summary_csv, SUMMARY_HEADER)

    print_banner(f"[PHASE: {phase_name.upper()}] STARTED")

    for agent in agents:
        print_banner(f"Starting Agent: {agent}")
        agent_start = datetime.datetime.now()
        agent_start_monotonic = time.perf_counter()

        # For network I/O delta calculations
        net0 = get_system_metrics()
        net_last = (net0["net_bytes_sent"], net0["net_bytes_recv"])

        episodes_total = phase["train"] + phase["test"]

        for episode in range(episodes_total):
            is_train = episode < phase["train"]
            req_mb = random.choice(REQUEST_MB_CHOICES)

            # Simulate processing
            processing_ms = process_request(req_mb)

            # Collect system metrics
            m = get_system_metrics(net_last=net_last)
            net_last = (m["net_bytes_sent"], m["net_bytes_recv"])

            # Fault injection logic
            if episode > 0 and episode % 300 == 0:
                fault_type = simulate_faults(probability=0.30)
            else:
                fault_type = simulate_faults(probability=0.05)

            # Log episode data
            append_row(episode_csv, [
                m["timestamp"],
                phase_name,
                agent,
                episode,
                int(is_train),
                req_mb,
                processing_ms,
                m["cpu_percent"],
                m["mem_percent"],
                m["disk_percent"],
                m["delta_sent_mb"],
                m["delta_recv_mb"],
                m["gpu_util_percent"],
                m["gpu_mem_percent"],
                fault_type or "None",
                agent_start.isoformat(),
                "",  # end time (filled in summary)
                ""   # elapsed (filled in summary)
            ])

            # Print progress every 500 episodes
            if episode % 500 == 0:
                logger.info(
                    f"[{phase_name} | {agent}] Episode {episode}/{episodes_total} "
                    f"req={req_mb}MB proc={processing_ms}ms "
                    f"CPU={m['cpu_percent']}% MEM={m['mem_percent']}%"
                )

        # Compute phase summary for the current agent
        agent_end = datetime.datetime.now()
        agent_elapsed = round(time.perf_counter() - agent_start_monotonic, 3)
        append_row(summary_csv, [
            phase_name,
            agent,
            agent_start.isoformat(),
            agent_end.isoformat(),
            agent_elapsed,
            episodes_total,
            phase["train"],
            phase["test"]
        ])
        logger.info(f"[DONE] {agent} completed {phase_name} phase in {agent_elapsed}s")

    print_banner(f"[PHASE {phase_name.upper()}] COMPLETED")


def main():
    """Main execution entry point."""
    exp_start = datetime.datetime.now()
    print_banner("DISTRIBUTED MULTI-AGENT DRL BENCHMARK INITIATED")
    logger.info(f"Experiment start: {exp_start.isoformat()}")
    logger.info(f"Results root: {RESULTS_ROOT}")
    logger.info(f"Agents: {agents}")

    # Run all phases sequentially
    for phase in phases:
        run_phase(phase)

    exp_end = datetime.datetime.now()
    elapsed_min = round((exp_end - exp_start).total_seconds() / 60.0, 2)
    print_banner("ALL PHASES COMPLETED")
    logger.info(f"Experiment end: {exp_end.isoformat()}")
    logger.info(f"Total Experiment Duration: {elapsed_min} minutes")


if __name__ == "__main__":
    main()
