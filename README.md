# Distributed Multi-Node Benchmark of Deep Reinforcement Learning Controllers for Real-Time Resource and Fault Management in Containerized Systems

## Overview
This repository contains the implementation of a distributed benchmarking framework for evaluating Deep Reinforcement Learning (DRL) controllers in real-time resource and fault management within containerized systems.  
The framework supports multiple algorithms, including PPO, SAC, TD3, DDPG, A2C, DQN, and MADDPG, and enables reproducible testing under controlled workload and fault conditions.

The objective is to provide a transparent and scalable testbed for fault-aware orchestration, emphasizing adaptability, stability, and energy efficiency.  
All experiments are executed in structured phases (early, mid, and final) to analyze algorithmic convergence, resource usage, and recovery performance over time.

### Core Features
- Container-based testbed with realistic workload and system telemetry
- Fault injection for CPU spikes, memory saturation, I/O pauses, and network jitter
- Phase-based experiment control for progressive benchmarking
- Unified metric collection (CPU, memory, disk, network, GPU, latency, recovery)
- Automated CSV logging and reproducible setup for all agents


## Usage
1. **Build the environment**
   ```bash
   docker build -t drl-fault .
   ```

2. **Run the experiment**
   ```bash
   python3 runner.py
   ```

3. **Access results**
   All per-phase CSV logs will appear under the `results/` directory, including episode-level and summary-level statistics for each algorithm.

## Research Context
This repository supports the study titled  
**“Distributed Multi-Node Benchmark of Deep Reinforcement Learning Controllers for Real-Time Resource and Fault Management in Containerized Systems”**,  
submitted to *Engineering Applications of Artificial Intelligence*.  

The work introduces a reproducible and container-based benchmarking framework for analyzing DRL algorithms in realistic, fault-prone environments with real-time telemetry and energy-aware control.

## Citation
If you use this framework in your research, please cite as:
> Enes Bajrami, *Distributed Multi-Node Benchmark of Deep Reinforcement Learning Controllers for Real-Time Resource and Fault Management in Containerized Systems*,  
> Ss. Cyril and Methodius University in Skopje, Faculty of Computer Science and Engineering, Republic of North Macedonia.

## Author
**Enes Bajrami**  
Ss. Cyril and Methodius University in Skopje  
Faculty of Computer Science and Engineering  
Skopje, Republic of North Macedonia  
📧 enes.bajrami@students.finki.ukim.mk

## License
This project is released for academic and research purposes. Redistribution and modification are permitted with appropriate citation and acknowledgment.
