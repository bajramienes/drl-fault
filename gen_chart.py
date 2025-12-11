import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================

EPISODE_DIR = r"C:\Users\User\Desktop\drl_fault\results\episodes"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "charts")

AGENT_COLORS = {
    "PPO": "#1f77b4",
    "SAC": "#ff7f0e",
    "TD3": "#2ca02c",
    "DDPG": "#d62728",
    "A2C": "#9467bd",
    "DQN": "#8c564b",
    "BASELINE": "#000000",
}

NUMERIC_COLUMNS = [
    "latency_ms",
    "cpu_percent",
    "mem_percent",
    "disk_percent",
    "energy_joule",
    "energy_efficiency",
    "host_cpu_percent",
    "host_mem_percent",
    "total_reward",
    "request_mb",
]

DAILY_COLOR = "#1f3b6f"   # light navy blue

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
})
sns.set_style("whitegrid")


# ============================================================
# UTILS
# ============================================================

def ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_all_episodes():
    pattern = os.path.join(EPISODE_DIR, "*.csv")
    files = glob.glob(pattern)
    dfs = []

    for f in files:
        df = pd.read_csv(f)

        # Infer agent / phase from filename
        base = os.path.basename(f).replace(".csv", "")
        parts = base.split("_")
        if len(parts) >= 2:
            agent = parts[0].upper()
            phase = parts[1].lower()
        else:
            agent = str(df.get("agent", "UNKNOWN")).upper()
            phase = str(df.get("phase", "unknown")).lower()

        df["agent"] = agent
        df["phase"] = phase

        # Date parsing
        df["date"] = pd.to_datetime(df["date_eu"], format="%d-%m-%Y")
        df["date_only"] = df["date"].dt.date

        # Numeric conversions
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ensure action_taken is numeric
        df["action_taken"] = pd.to_numeric(df["action_taken"], errors="coerce").fillna(0).astype(int)

        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No CSV files found in {EPISODE_DIR}")

    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    return data


def savefig(name):
    path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


def annotate_bars(ax, bars, yerr=None, rot=45, offset_frac=0.04):
    """
    Put labels above bars, taking into account error bars.
    Works for positive, negative, or mixed values.
    """
    tops = []
    if yerr is None:
        for b in bars:
            tops.append(b.get_height())
    else:
        if np.isscalar(yerr):
            for b in bars:
                tops.append(b.get_height() + yerr)
        else:
            for b, e in zip(bars, yerr):
                tops.append(b.get_height() + (e if e is not None else 0.0))

    if not tops:
        return

    max_top = float(max(tops))
    min_top = float(min(tops))

    if max_top == 0 and min_top == 0:
        return

    span = max_top - min_top if max_top != min_top else abs(max_top) if max_top != 0 else 1.0
    pad = span * offset_frac

    if max_top <= 0:
        # all bars <= 0
        ymin = min_top - 3 * pad
        ymax = 0 + pad
    elif min_top >= 0:
        # all bars >= 0
        ymin = 0
        ymax = max_top + 3 * pad
    else:
        # mixed
        ymin = min_top - 3 * pad
        ymax = max_top + 3 * pad

    ax.set_ylim(ymin, ymax)

    for i, b in enumerate(bars):
        val = float(b.get_height())
        top = float(tops[i])
        y = top + pad
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            y,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            rotation=rot,
            fontsize=8,
        )


# ============================================================
# BASIC DERIVED DATA
# ============================================================

def get_fault_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["fault_type"].astype(str).str.lower() != "none"].copy()


def compute_episode_rewards(df: pd.DataFrame) -> pd.DataFrame:
    ep = (
        df.groupby(["agent", "phase", "episode_index"])["total_reward"]
        .sum()
        .reset_index()
    )
    return ep


# ============================================================
# 1. FAULT COUNTS / DISTRIBUTIONS
# ============================================================

def chart_fault_count_per_agent(df):
    faults = get_fault_rows(df)
    if faults.empty:
        return
    counts = faults["agent"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    agents = counts.index.tolist()
    vals = counts.values
    colors = [AGENT_COLORS.get(a, "#444444") for a in agents]
    bars = ax.bar(agents, vals, color=colors)
    annotate_bars(ax, bars)
    ax.set_ylabel("Fault Count")
    ax.set_title("Total Faults per Agent")
    savefig("fault_count_per_agent")


def chart_fault_count_per_phase(df):
    faults = get_fault_rows(df)
    if faults.empty:
        return
    counts = faults["phase"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    phases = counts.index.tolist()
    vals = counts.values
    bars = ax.bar(phases, vals, color="#1f77b4")
    annotate_bars(ax, bars, rot=0)
    ax.set_ylabel("Fault Count")
    ax.set_title("Total Faults per Phase")
    savefig("fault_count_per_phase")


def chart_fault_type_distribution_overall(df):
    faults = get_fault_rows(df)
    if faults.empty:
        return
    counts = faults["fault_type"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color="#ff7f0e")
    annotate_bars(ax, bars, rot=30)
    ax.set_ylabel("Fault Count")
    ax.set_title("Fault Type Distribution (All Agents)")
    savefig("fault_type_distribution_overall")


def chart_fault_type_per_agent(df):
    faults = get_fault_rows(df)
    if faults.empty:
        return
    grouped = (
        faults.groupby(["agent", "fault_type"])
        .size()
        .reset_index(name="count")
    )
    agents = sorted(grouped["agent"].unique())
    ft_types = sorted(grouped["fault_type"].unique())
    x = np.arange(len(agents))
    width = 0.8 / max(1, len(ft_types))

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, ft in enumerate(ft_types):
        sub = grouped[grouped["fault_type"] == ft]
        vals = sub.set_index("agent")["count"].reindex(agents).fillna(0).values
        pos = x + (i - len(ft_types) / 2) * width + width / 2
        bars = ax.bar(pos, vals, width=width, label=ft)
        annotate_bars(ax, bars, offset_frac=0.01)

    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("Fault Count")
    ax.set_title("Fault Types per Agent")
    ax.legend(title="Fault Type")
    savefig("fault_type_per_agent")


def chart_fault_type_per_phase(df):
    faults = get_fault_rows(df)
    if faults.empty:
        return
    grouped = (
        faults.groupby(["phase", "fault_type"])
        .size()
        .reset_index(name="count")
    )
    phases = sorted(grouped["phase"].unique())
    ft_types = sorted(grouped["fault_type"].unique())
    x = np.arange(len(phases))
    width = 0.8 / max(1, len(ft_types))

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, ft in enumerate(ft_types):
        sub = grouped[grouped["fault_type"] == ft]
        vals = sub.set_index("phase")["count"].reindex(phases).fillna(0).values
        pos = x + (i - len(ft_types) / 2) * width + width / 2
        bars = ax.bar(pos, vals, width=width, label=ft)
        annotate_bars(ax, bars, offset_frac=0.01)

    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel("Fault Count")
    ax.set_title("Fault Types per Phase")
    ax.legend(title="Fault Type")
    savefig("fault_type_per_phase")


def chart_faults_per_episode_hist(df):
    faults = get_fault_rows(df)
    if faults.empty:
        return
    counts = (
        faults.groupby(["agent", "episode_index"])
        .size()
        .reset_index(name="faults")
    )

    plt.figure(figsize=(10, 5))
    for agent, sub in counts.groupby("agent"):
        sns.histplot(sub["faults"], kde=False, bins=10,
                     label=agent, alpha=0.4,
                     color=AGENT_COLORS.get(agent, None))
    plt.xlabel("Faults per Episode")
    plt.ylabel("Frequency")
    plt.title("Distribution of Faults per Episode")
    plt.legend()
    savefig("faults_per_episode_hist")


def chart_faults_per_day_hist(df):
    faults = get_fault_rows(df)
    if faults.empty:
        return
    counts = faults.groupby(["agent", "date_only"]).size().reset_index(name="faults")

    plt.figure(figsize=(10, 5))
    for agent, sub in counts.groupby("agent"):
        sns.histplot(sub["faults"], kde=False, bins=10,
                     label=agent, alpha=0.4,
                     color=AGENT_COLORS.get(agent, None))
    plt.xlabel("Faults per Day")
    plt.ylabel("Frequency")
    plt.title("Distribution of Faults per Day")
    plt.legend()
    savefig("faults_per_day_hist")


# ============================================================
# 2. METRICS AT FAULT
# ============================================================

def bar_metric_at_fault_per_agent(df, metric, ylabel, filename):
    faults = get_fault_rows(df)
    if faults.empty:
        return
    grouped = faults.groupby("agent")[metric].agg(["mean", "std"]).reset_index()
    agents = grouped["agent"].tolist()
    means = grouped["mean"].values
    stds = grouped["std"].fillna(0).values
    x = np.arange(len(agents))
    colors = [AGENT_COLORS.get(a, "#555555") for a in agents]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors)
    annotate_bars(ax, bars, yerr=stds, rot=45)
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} During Fault Events per Agent")
    savefig(filename)


def bar_metric_at_fault_per_phase(df, metric, ylabel, filename):
    faults = get_fault_rows(df)
    if faults.empty:
        return
    grouped = faults.groupby("phase")[metric].agg(["mean", "std"]).reset_index()
    phases = grouped["phase"].tolist()
    means = grouped["mean"].values
    stds = grouped["std"].fillna(0).values
    x = np.arange(len(phases))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="#2ca02c")
    annotate_bars(ax, bars, yerr=stds, rot=0)
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} During Fault Events per Phase")
    savefig(filename)


def heat_metric_at_fault_agent_phase(df, metric, filename, title):
    faults = get_fault_rows(df)
    if faults.empty:
        return
    pivot = faults.pivot_table(
        index="agent", columns="phase", values=metric, aggfunc="mean"
    )
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd")
    plt.title(title)
    savefig(filename)


# ============================================================
# 3. ACTIONS DURING FAULTS
# ============================================================

def chart_actions_during_faults_per_agent(df):
    faults = get_fault_rows(df)
    if faults.empty:
        return
    grouped = faults.groupby(["agent", "action_taken"]).size().reset_index(name="count")

    agents = sorted(grouped["agent"].unique())
    acts = sorted(grouped["action_taken"].unique())
    x = np.arange(len(agents))
    width = 0.8 / max(1, len(acts))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, a in enumerate(acts):
        sub = grouped[grouped["action_taken"] == a]
        vals = sub.set_index("agent")["count"].reindex(agents).fillna(0).values
        pos = x + (i - len(acts) / 2) * width + width / 2
        bars = ax.bar(pos, vals, width=width, label=f"A{a}")
        annotate_bars(ax, bars, offset_frac=0.01)

    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("Count")
    ax.set_title("Actions Taken During Fault Events (per Agent)")
    ax.legend(title="Action")
    savefig("actions_during_faults_per_agent")


def chart_actions_during_faults_overall(df):
    faults = get_fault_rows(df)
    if faults.empty:
        return
    counts = faults["action_taken"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    acts = counts.index.tolist()
    vals = counts.values
    bars = ax.bar([str(a) for a in acts], vals, color="#8c564b")
    annotate_bars(ax, bars)
    ax.set_ylabel("Count")
    ax.set_title("Actions Taken During Fault Events (Overall)")
    savefig("actions_during_faults_overall")


def chart_action_fault_heatmap(df):
    faults = get_fault_rows(df)
    if faults.empty:
        return
    pivot = faults.groupby(["fault_type", "action_taken"]).size().unstack(fill_value=0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)

    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("P(Action | Fault Type)")
    savefig("action_vs_fault_heatmap_overall")


# ============================================================
# 4. FAULT RECOVERY CURVES
# ============================================================

def chart_fault_recovery(df, window=20):
    faults = get_fault_rows(df)
    if faults.empty:
        return

    agents = sorted(df["agent"].unique())

    # Latency recovery
    plt.figure(figsize=(10, 5))
    for agent in agents:
        sub = df[df["agent"] == agent].sort_values("timestamp").reset_index(drop=True)
        fault_idx = sub.index[sub["fault_type"].astype(str).str.lower() != "none"].tolist()
        if not fault_idx:
            continue
        segments = []
        for idx in fault_idx:
            seg = sub.loc[idx: idx + window - 1, "latency_ms"]
            if len(seg) == window:
                segments.append(seg.values.astype(float))
        if not segments:
            continue
        arr = np.vstack(segments)
        mean_curve = np.nanmean(arr, axis=0)
        steps = np.arange(window)
        plt.plot(steps, mean_curve, label=agent, color=AGENT_COLORS.get(agent, None))
    plt.xlabel("Steps After Fault")
    plt.ylabel("Latency (ms)")
    plt.title("Latency Recovery After Fault")
    plt.legend()
    savefig("fault_recovery_latency")

    # CPU recovery
    plt.figure(figsize=(10, 5))
    for agent in agents:
        sub = df[df["agent"] == agent].sort_values("timestamp").reset_index(drop=True)
        fault_idx = sub.index[sub["fault_type"].astype(str).str.lower() != "none"].tolist()
        if not fault_idx:
            continue
        segments = []
        for idx in fault_idx:
            seg = sub.loc[idx: idx + window - 1, "cpu_percent"]
            if len(seg) == window:
                segments.append(seg.values.astype(float))
        if not segments:
            continue
        arr = np.vstack(segments)
        mean_curve = np.nanmean(arr, axis=0)
        steps = np.arange(window)
        plt.plot(steps, mean_curve, label=agent, color=AGENT_COLORS.get(agent, None))
    plt.xlabel("Steps After Fault")
    plt.ylabel("CPU (%)")
    plt.title("CPU Recovery After Fault")
    plt.legend()
    savefig("fault_recovery_cpu")


# ============================================================
# 5. DAILY 3×3 GRIDS (LIGHT NAVY BLUE)
# ============================================================

def chart_daily_faults_grid(df):
    faults = get_fault_rows(df)
    if faults.empty:
        return
    days = sorted(faults["date_only"].unique())
    if not days:
        return

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    for i, day in enumerate(days[:9]):
        ax = axes[i]
        sub = faults[faults["date_only"] == day].sort_values("timestamp")
        if sub.empty:
            ax.set_title(f"{i+1}")
            continue
        t = np.arange(len(sub))
        # bar "spikes" in navy blue
        ax.bar(t, np.ones_like(t), color=DAILY_COLOR, width=1.0)
        ax.set_title(f"{i+1}")
        ax.set_ylabel("Faults")

    plt.suptitle("Fault Events per Day (Days 1–9)")
    savefig("daily_faults_grid")


def chart_daily_metric_grid(df, metric, filename, ylabel):
    days = sorted(df["date_only"].unique())
    if not days:
        return

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    for i, day in enumerate(days[:9]):
        ax = axes[i]
        sub = df[df["date_only"] == day].sort_values("timestamp")
        if sub.empty:
            ax.set_title(f"{i+1}")
            continue
        t = np.arange(len(sub))
        ax.plot(t, sub[metric].values, color=DAILY_COLOR, linewidth=0.8)
        ax.set_title(f"{i+1}")
        ax.set_ylabel(ylabel)

    plt.suptitle(f"{ylabel} per Day (Days 1–9)")
    savefig(filename)


def chart_daily_action_grid(df):
    days = sorted(df["date_only"].unique())
    if not days:
        return

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    for i, day in enumerate(days[:9]):
        ax = axes[i]
        sub = df[df["date_only"] == day]
        if sub.empty:
            ax.set_title(f"{i+1}")
            continue
        counts = sub["action_taken"].value_counts().sort_index()
        acts = [str(a) for a in counts.index]
        vals = counts.values
        bars = ax.bar(acts, vals, color=DAILY_COLOR)
        annotate_bars(ax, bars, rot=0, offset_frac=0.03)
        ax.set_title(f"{i+1}")
        ax.set_ylabel("Action Count")

    plt.suptitle("Action Distribution per Day (Days 1–9)")
    savefig("daily_action_grid")


# ============================================================
# 6. EPISODE REWARD & SYSTEM RESOURCE BOXES
# ============================================================

def chart_reward_per_episode(df):
    ep = compute_episode_rewards(df)
    plt.figure(figsize=(10, 5))
    for agent, sub in ep.groupby("agent"):
        plt.plot(sub["episode_index"], sub["total_reward"],
                 label=agent, color=AGENT_COLORS.get(agent, None))
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward per Episode")
    plt.legend()
    savefig("reward_per_episode")


def chart_reward_per_episode_smoothed(df, window=20):
    ep = compute_episode_rewards(df)
    plt.figure(figsize=(10, 5))
    for agent, sub in ep.groupby("agent"):
        smooth = sub["total_reward"].rolling(window).mean()
        plt.plot(sub["episode_index"], smooth,
                 label=agent, color=AGENT_COLORS.get(agent, None))
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.title(f"Smoothed Reward per Episode (window={window})")
    plt.legend()
    savefig("reward_per_episode_smoothed")


def box_metric_per_agent(df, metric, ylabel, filename):
    plt.figure(figsize=(8, 5))
    agents = sorted(df["agent"].unique())
    data = [df[df["agent"] == a][metric].dropna() for a in agents]
    plt.boxplot(data, tick_labels=agents, showfliers=False)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Distribution per Agent")
    savefig(filename)


# ============================================================
# 7. ENERGY–LATENCY PARETO
# ============================================================

def chart_energy_latency_pareto(df, max_points=4000):
    plt.figure(figsize=(10, 5))
    for agent, sub in df.groupby("agent"):
        n = len(sub)
        if n > max_points:
            sub = sub.sample(max_points, random_state=0)
        plt.scatter(
            sub["latency_ms"],
            sub["energy_joule"],
            s=4, alpha=0.4,
            label=agent,
            color=AGENT_COLORS.get(agent, None),
        )
    plt.xlabel("Latency (ms)")
    plt.ylabel("Energy (J)")
    plt.title("Energy–Latency Pareto (Downsampled)")
    plt.legend()
    savefig("energy_latency_pareto")


# ============================================================
# 8. RADAR CHARTS
# ============================================================

def chart_radar_fault(df):
    faults = get_fault_rows(df)
    if faults.empty:
        return

    fault_counts = faults.groupby("agent").size()
    latency_at_fault = faults.groupby("agent")["latency_ms"].mean()
    cpu_at_fault = faults.groupby("agent")["cpu_percent"].mean()
    reward_at_fault = faults.groupby("agent")["total_reward"].mean()

    metrics = {
        "Fault count": fault_counts,
        "Latency@fault": latency_at_fault,
        "CPU@fault": cpu_at_fault,
        "Reward@fault": reward_at_fault,
    }

    labels = list(metrics.keys())
    agents = sorted(fault_counts.index)

    norm_metrics = {}
    for name, series in metrics.items():
        s = series.reindex(agents).astype(float)
        if s.max() - s.min() > 0:
            s_norm = (s - s.min()) / (s.max() - s.min())
        else:
            s_norm = s * 0
        norm_metrics[name] = s_norm

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for agent in agents:
        vals = [norm_metrics[m][agent] for m in labels]
        vals += vals[:1]
        ax.plot(angles, vals, color=AGENT_COLORS.get(agent, None), linewidth=2)
        ax.fill(angles, vals, alpha=0.1, color=AGENT_COLORS.get(agent, None))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title("Fault-Centric Radar per Agent (normalized)")
    savefig("radar_fault")


def chart_radar_overall(df):
    metrics = {
        "Latency": df.groupby("agent")["latency_ms"].mean(),
        "CPU": df.groupby("agent")["cpu_percent"].mean(),
        "Memory": df.groupby("agent")["mem_percent"].mean(),
        "Disk": df.groupby("agent")["disk_percent"].mean(),
        "Reward": df.groupby("agent")["total_reward"].mean(),
    }

    labels = list(metrics.keys())
    agents = sorted(df["agent"].unique())

    norm_metrics = {}
    for name, series in metrics.items():
        s = series.reindex(agents).astype(float)
        if s.max() - s.min() > 0:
            s_norm = (s - s.min()) / (s.max() - s.min())
        else:
            s_norm = s * 0
        norm_metrics[name] = s_norm

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for agent in agents:
        vals = [norm_metrics[m][agent] for m in labels]
        vals += vals[:1]
        ax.plot(angles, vals, color=AGENT_COLORS.get(agent, None), linewidth=2)
        ax.fill(angles, vals, alpha=0.1, color=AGENT_COLORS.get(agent, None))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title("Overall Performance Radar per Agent (normalized)")
    savefig("radar_overall")


# ============================================================
# 9. EXTRA CHARTS TO PUSH OVER 40+
# ============================================================

def chart_faults_over_time(df):
    faults = get_fault_rows(df)
    if faults.empty:
        return
    counts = faults.groupby(["date_only", "agent"]).size().reset_index(name="faults")
    plt.figure(figsize=(10, 5))
    for agent, sub in counts.groupby("agent"):
        sub = sub.sort_values("date_only")
        plt.plot(sub["date_only"], sub["faults"], marker="o",
                 label=agent, color=AGENT_COLORS.get(agent, None))
    plt.xlabel("Date")
    plt.ylabel("Faults per Day")
    plt.title("Faults over Time per Agent")
    plt.legend()
    savefig("faults_over_time")


def chart_episode_length_hist(df):
    lengths = (
        df.groupby(["agent", "episode_index"])["step_in_episode"]
        .max()
        .reset_index(name="steps")
    )
    plt.figure(figsize=(10, 5))
    for agent, sub in lengths.groupby("agent"):
        sns.histplot(sub["steps"], kde=False, bins=10,
                     label=agent, alpha=0.4,
                     color=AGENT_COLORS.get(agent, None))
    plt.xlabel("Steps per Episode")
    plt.ylabel("Frequency")
    plt.title("Episode Length Distribution")
    plt.legend()
    savefig("episode_length_hist")


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_output()
    print("[INFO] Loading CSVs...")
    df = load_all_episodes()
    print(f"[INFO] Loaded {len(df)} rows.")

    # 1. Fault counts & distributions
    chart_fault_count_per_agent(df)
    chart_fault_count_per_phase(df)
    chart_fault_type_distribution_overall(df)
    chart_fault_type_per_agent(df)
    chart_fault_type_per_phase(df)
    chart_faults_per_episode_hist(df)
    chart_faults_per_day_hist(df)

    # 2. Metrics at fault
    bar_metric_at_fault_per_agent(df, "latency_ms", "Latency at Fault (ms)", "latency_at_fault_per_agent")
    bar_metric_at_fault_per_phase(df, "latency_ms", "Latency at Fault (ms)", "latency_at_fault_per_phase")
    heat_metric_at_fault_agent_phase(df, "latency_ms", "latency_at_fault_heatmap", "Mean Latency at Fault (Agent × Phase)")

    bar_metric_at_fault_per_agent(df, "cpu_percent", "CPU (%) at Fault", "cpu_at_fault_per_agent")
    bar_metric_at_fault_per_phase(df, "cpu_percent", "CPU (%) at Fault", "cpu_at_fault_per_phase")
    heat_metric_at_fault_agent_phase(df, "cpu_percent", "cpu_at_fault_heatmap", "Mean CPU at Fault (Agent × Phase)")

    bar_metric_at_fault_per_agent(df, "mem_percent", "Memory (%) at Fault", "mem_at_fault_per_agent")
    bar_metric_at_fault_per_agent(df, "disk_percent", "Disk (%) at Fault", "disk_at_fault_per_agent")
    bar_metric_at_fault_per_agent(df, "energy_efficiency", "Energy Efficiency at Fault", "energy_eff_at_fault_per_agent")
    bar_metric_at_fault_per_agent(df, "total_reward", "Reward at Fault", "reward_at_fault_per_agent")

    # 3. Actions during faults
    chart_actions_during_faults_per_agent(df)
    chart_actions_during_faults_overall(df)
    chart_action_fault_heatmap(df)

    # 4. Recovery curves
    chart_fault_recovery(df, window=20)

    # 5. Daily grids
    chart_daily_faults_grid(df)
    chart_daily_metric_grid(df, "cpu_percent", "daily_cpu_grid", "CPU (%)")
    chart_daily_metric_grid(df, "latency_ms", "daily_latency_grid", "Latency (ms)")
    chart_daily_action_grid(df)

    # 6. Episodes & resource distributions
    chart_reward_per_episode(df)
    chart_reward_per_episode_smoothed(df, window=20)
    box_metric_per_agent(df, "latency_ms", "Latency (ms)", "latency_box_per_agent")
    box_metric_per_agent(df, "cpu_percent", "CPU (%)", "cpu_box_per_agent")
    box_metric_per_agent(df, "mem_percent", "Memory (%)", "mem_box_per_agent")
    box_metric_per_agent(df, "disk_percent", "Disk (%)", "disk_box_per_agent")
    box_metric_per_agent(df, "energy_joule", "Energy (J)", "energy_box_per_agent")
    box_metric_per_agent(df, "host_cpu_percent", "Host CPU (%)", "host_cpu_box_per_agent")
    box_metric_per_agent(df, "host_mem_percent", "Host Memory (%)", "host_mem_box_per_agent")

    # 7. Energy–latency Pareto
    chart_energy_latency_pareto(df, max_points=4000)

    # 8. Radar charts
    chart_radar_fault(df)
    chart_radar_overall(df)

    # 9. Extra charts
    chart_faults_over_time(df)
    chart_episode_length_hist(df)

    print("[INFO] Finished generating 40+ charts into 'charts' folder.")


if __name__ == "__main__":
    main()
