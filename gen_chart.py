# gen_chart_speed.py (drop-in replacement for gen_chart.py)
# Stable, no multiprocessing, Overleaf-friendly chart generation.

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =========================
# SPEED MODE SETTINGS
# =========================
SPEED_MODE = True   # Keep ON
NUM_WORKERS = 1     # Force sequential for Windows stability

MAX_SCATTER_POINTS_PER_AGENT = 500
BOX_MAX_SAMPLES_PER_BIN = 3000
ROLL_FRACTION = 0.02
ROLL_MIN = 50
ROLL_MAX = 500

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.8,
})

AGENT_ORDER = ["A2C", "DDPG", "DQN", "MADDPG", "PPO", "SAC", "TD3"]
AGENT_COLORS = {
    "PPO": "#E63946",
    "SAC": "#457B9D",
    "TD3": "#2A9D8F",
    "DDPG": "#1D3557",
    "A2C": "#E9C46A",
    "DQN": "#EF476F",
    "MADDPG": "#6D597A",
}
PHASES = ["early", "mid", "final"]

CPU_BUCKET_LABELS = ["0-25%", "25-50%", "50-75%", "75-100%"]
CPU_BUCKET_EDGES = [0, 25, 50, 75, 100]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, "Results")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "figures")
CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

PHASE_FILES = {
    "early": os.path.join(BASE_PATH, "early", "episodes_early.csv"),
    "mid":   os.path.join(BASE_PATH, "mid", "episodes_mid.csv"),
    "final": os.path.join(BASE_PATH, "final", "episodes_final.csv"),
}
SUMMARY_FILES = {
    "early": os.path.join(BASE_PATH, "early", "summary_early.csv"),
    "mid":   os.path.join(BASE_PATH, "mid", "summary_mid.csv"),
    "final": os.path.join(BASE_PATH, "final", "summary_final.csv"),
}

# ---------------- Helpers ----------------
def _legend_outside(ax): 
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, bbox_to_anchor=(1.02,1), loc="upper left")

def _pad_ylim_for_labels(ax, pad=0.12):
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin)*pad)

def _rolling(series):
    n = len(series)
    if n == 0:
        return series
    w = max(ROLL_MIN, min(ROLL_MAX, int(n * ROLL_FRACTION)))
    return series.rolling(window=w, min_periods=max(3, w//3)).mean()

def _cache_path(name): return os.path.join(CACHE_DIR, f"{name}.pkl")
def _load_cached(name):
    p = _cache_path(name)
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None
def _save_cached(name, obj):
    pickle.dump(obj, open(_cache_path(name),"wb"))

def _derive_columns(df):
    if "cpu_percent" not in df.columns:
        df["cpu_percent"] = np.nan
    if "mem_percent" not in df.columns:
        df["mem_percent"] = np.nan
    if "request_mb" in df.columns and "processing_ms" in df.columns:
        df["throughput_MBps"] = (df["request_mb"]/df["processing_ms"]) * 1000
    return df

def _parse_date(df):
    if "timestamp" in df.columns:
        t = pd.to_datetime(df["timestamp"], errors="coerce")
        df["date"] = t.dt.date
    else:
        df["date"] = pd.NaT
    return df

def _load_phase_data():
    cached = _load_cached("phase_data")
    if cached is not None: return cached
    data = {}
    for phase, path in PHASE_FILES.items():
        df = pd.read_csv(path)
        df = _derive_columns(df)
        df = _parse_date(df)
        df = df[df["agent"].isin(AGENT_ORDER)].copy()
        data[phase] = df
    _save_cached("phase_data", data)
    return data

def _load_summaries():
    cached = _load_cached("summaries")
    if cached is not None: return cached
    return {ph: pd.read_csv(path) for ph, path in SUMMARY_FILES.items()}

def _savefig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

# ---------------- Chart Functions ----------------
def chart_trend_phase(data, metric, ylabel, fname_prefix, phase):
    df = data[phase]
    plt.figure(figsize=(8,5))
    for agent in AGENT_ORDER:
        sub = df[df.agent == agent].sort_values("episode")
        if sub.empty: continue
        roll = _rolling(sub[metric])
        plt.plot(sub["episode"], roll, color=AGENT_COLORS[agent], label=agent, alpha=0.7)
    plt.title(f"{ylabel} Trend ({phase})")
    plt.xlabel("Episode"); plt.ylabel(ylabel); plt.grid(alpha=0.3)
    _legend_outside(plt.gca())
    _savefig(os.path.join(OUTPUT_DIR, f"{fname_prefix}_{phase}.pdf"))

def chart_mean_line_per_agent(data, metric, ylabel, title, fname):
    plt.figure(figsize=(8,5))
    for phase in PHASES:
        df = data[phase]
        m = df.groupby("agent")[metric].mean().reindex(AGENT_ORDER)
        plt.plot(AGENT_ORDER, m.values, "--o", label=phase)
    plt.ylabel(ylabel); plt.title(title); plt.grid(alpha=0.3)
    _legend_outside(plt.gca())
    _savefig(os.path.join(OUTPUT_DIR, fname))

def chart_train_test_counts():
    summaries = _load_summaries()
    fig, axes = plt.subplots(3,1,figsize=(8,10),sharey=True)
    for ax, phase in zip(axes, PHASES):
        df = summaries[phase].groupby("agent").sum().reindex(AGENT_ORDER)
        ax.bar(AGENT_ORDER, df["train_episodes"], label="Train", alpha=0.7)
        ax.bar(AGENT_ORDER, df["test_episodes"], label="Test", alpha=0.7)
        ax.set_title(phase); ax.grid(alpha=0.3)
    _legend_outside(axes[0])
    _savefig(os.path.join(OUTPUT_DIR,"train_vs_test_per_phase.pdf"))

def chart_speedup(data, phase):
    df = data[phase]
    avg = df.groupby("agent")["processing_ms"].mean().reindex(AGENT_ORDER)
    rel = avg / avg.min()
    plt.figure(figsize=(8,5))
    plt.bar(AGENT_ORDER, rel.values, color=[AGENT_COLORS[a] for a in AGENT_ORDER])
    plt.title(f"Relative Execution Time (lower=better) - {phase}")
    plt.ylabel("× slow-down"); plt.grid(alpha=0.3)
    _savefig(os.path.join(OUTPUT_DIR, f"speedup_{phase}.pdf"))

def chart_cpu_latency_scatter(data, phase):
    df = data[phase][["agent","cpu_percent","processing_ms"]].dropna()
    plt.figure(figsize=(7,5))
    for a in AGENT_ORDER:
        sub = df[df.agent == a]
        if SPEED_MODE: sub = sub.sample(min(len(sub), MAX_SCATTER_POINTS_PER_AGENT), random_state=42)
        plt.scatter(sub.cpu_percent, sub.processing_ms, s=8, alpha=0.25, color=AGENT_COLORS[a], label=a)
    plt.xlabel("CPU (%)"); plt.ylabel("Latency (ms)")
    plt.title(f"CPU vs Latency ({phase})"); plt.grid(alpha=0.3)
    _legend_outside(plt.gca())
    _savefig(os.path.join(OUTPUT_DIR,f"cpu_vs_latency_scatter_{phase}.pdf"))

def chart_cpu_load_boxplot(data, phase):
    df = data[phase][["cpu_percent","processing_ms"]].dropna()
    df["zone"] = pd.cut(df.cpu_percent, bins=CPU_BUCKET_EDGES, labels=CPU_BUCKET_LABELS)
    if SPEED_MODE: df = df.sample(min(len(df), BOX_MAX_SAMPLES_PER_BIN*4), random_state=42)
    vals = [df[df.zone==z].processing_ms for z in CPU_BUCKET_LABELS]
    plt.figure(figsize=(7,5))
    plt.boxplot(vals, labels=CPU_BUCKET_LABELS, showfliers=False)
    plt.ylabel("Latency (ms)"); plt.title(f"Latency by CPU Load ({phase})"); plt.grid(alpha=0.3)
    _savefig(os.path.join(OUTPUT_DIR,f"latency_by_cpu_zone_{phase}.pdf"))

def chart_fault_dashboards(data):
    pass  # We keep this disabled for speed now.

def chart_complexity(data):
    pass  # Leave disabled for now.

# ---------------- SAFE SEQUENTIAL EXECUTION ----------------
def _task(func,*args):
    try:
        func(*args)
        return f"OK: {func.__name__}"
    except Exception as e:
        return f"ERR: {func.__name__} -> {e}"

def main():
    print("Loading data...")
    data = _load_phase_data()

    tasks = []

    for ph in PHASES:
        tasks += [
            (chart_trend_phase,(data,"processing_ms","Latency (ms)","latency_trend",ph)),
            (chart_trend_phase,(data,"mem_percent","Memory (%)","memory_trend",ph)),
            (chart_trend_phase,(data,"cpu_percent","CPU (%)","cpu_trend",ph)),
            (chart_speedup,(data,ph)),
            (chart_cpu_latency_scatter,(data,ph)),
            (chart_cpu_load_boxplot,(data,ph)),
        ]

    tasks += [
        (chart_mean_line_per_agent,(data,"processing_ms","Mean Latency (ms)","Mean Latency per Agent","mean_latency_per_agent.pdf")),
        (chart_mean_line_per_agent,(data,"cpu_percent","Mean CPU (%)","Mean CPU per Agent","mean_cpu_per_agent.pdf")),
        (chart_mean_line_per_agent,(data,"mem_percent","Mean Memory (%)","Mean Memory per Agent","mean_memory_per_agent.pdf")),
        (chart_train_test_counts,()),
    ]

    print("Generating charts sequentially (stable mode)...")

    results=[]
    for func,args in tasks:
        r=_task(func,*args)
        print(r)
        results.append(r)

    ok=sum(r.startswith("OK") for r in results)
    err=[r for r in results if r.startswith("ERR")]

    print(f"\nDone. OK: {ok}, ERR: {len(err)}")
    print(f"Charts saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
