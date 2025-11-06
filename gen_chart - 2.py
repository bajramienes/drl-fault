import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ----------------- Paths & Styling -----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHASES = ["early","mid","final"]
AGENT_ORDER = ["A2C","DDPG","DQN","MADDPG","PPO","SAC","TD3"]
AGENT_COLORS = {
    "PPO": "#E63946","SAC": "#457B9D","TD3": "#2A9D8F",
    "DDPG": "#1D3557","A2C": "#E9C46A","DQN": "#EF476F","MADDPG": "#6D597A",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.35,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ----------------- Helpers -----------------
def _rotate_xticks(ax, deg=35):
    ax.tick_params(axis="x", labelrotation=deg)

def add_bar_labels(ax, rotation=35):
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.05
    new_ymax = ymax
    for bar in ax.patches:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + pad,
                f"{h:.0f}", ha="center", va="bottom", rotation=rotation)
        new_ymax = max(new_ymax, h + pad)
    ax.set_ylim(ymin, new_ymax * 1.12)
    _rotate_xticks(ax, rotation)

def smooth_series(s: pd.Series, w=80):
    w = max(5, int(w))
    return s.rolling(window=w, min_periods=max(3, w//3)).mean()

def sample_df(d, n=1500, seed=42):
    return d.sample(n, random_state=seed) if len(d) > n else d

# ----------------- Load Data -----------------
def load_merged():
    frames=[]
    for p in PHASES:
        path = os.path.join(SCRIPT_DIR,"Results",p,f"episodes_{p}.csv")
        d = pd.read_csv(path)
        d["phase"]=p
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    df["date"] = pd.to_datetime(df.get("timestamp"), errors="coerce").dt.date
    df["fault_type"] = df.get("fault_type", "none").fillna("none").astype(str)

    if "cpu_percent" not in df.columns:
        df["cpu_percent"] = np.nan
    if "mem_percent" not in df.columns:
        df["mem_percent"] = np.nan

    if {"request_mb","processing_ms"} <= set(df.columns):
        df["throughput"] = (df["request_mb"] / df["processing_ms"]) * 1000
    else:
        df["throughput"] = np.nan

    return df

def day_slices(df):
    days = sorted([d for d in df["date"].unique() if pd.notna(d)])
    if len(days)>=3:
        return [df[df["date"]==days[i]].copy() for i in range(3)]
    return list(np.array_split(df.sort_values("episode"),3))

# ----------------- FAULT DASHBOARDS -----------------
def build_fault_dashboards(merged):
    fault_mask_all = merged["fault_type"].str.lower()!="none"
    days = day_slices(merged)

    for i,day in enumerate(days,1):
        faults = day[fault_mask_all.loc[day.index]]

        # -------- MAIN DAILY DASHBOARD --------
        with PdfPages(os.path.join(OUTPUT_DIR, f"faults_day{i}.pdf")) as pdf:
            fig, axes = plt.subplots(3,3,figsize=(11,10))
            axes=axes.flatten()
            fig.suptitle(f"Daily {i}", fontsize=20)

            # 1 Fault Count
            c = faults.groupby("agent").size().reindex(AGENT_ORDER).fillna(0)
            axes[0].bar(AGENT_ORDER,c,color=[AGENT_COLORS[a] for a in AGENT_ORDER])
            axes[0].set_title("Fault Count per Agent")
            add_bar_labels(axes[0])

            # 2 Fault Types
            ft = faults["fault_type"].value_counts()
            if not ft.empty:
                axes[1].barh(ft.index,ft.values,color="#457B9D")
            axes[1].set_title("Fault Type Frequency")

            # 3 CPU During Faults
            m = faults.groupby("agent")["cpu_percent"].mean().reindex(AGENT_ORDER).fillna(0)
            axes[2].bar(AGENT_ORDER,m,color=[AGENT_COLORS[a] for a in AGENT_ORDER])
            axes[2].set_title("CPU During Faults")
            add_bar_labels(axes[2])

            # 4 Latency During Faults
            m = faults.groupby("agent")["processing_ms"].mean().reindex(AGENT_ORDER).fillna(0)
            axes[3].bar(AGENT_ORDER,m,color=[AGENT_COLORS[a] for a in AGENT_ORDER])
            axes[3].set_title("Latency During Faults")
            add_bar_labels(axes[3])

            # 5 Memory During Faults
            m = faults.groupby("agent")["mem_percent"].mean().reindex(AGENT_ORDER).fillna(0)
            axes[4].bar(AGENT_ORDER,m,color=[AGENT_COLORS[a] for a in AGENT_ORDER])
            axes[4].set_title("Memory During Faults")
            add_bar_labels(axes[4])

            # 6 Post-Fault Stability
            vals=[day.loc[day.agent==a,"processing_ms"].dropna() for a in AGENT_ORDER]
            bp = axes[5].boxplot(vals,labels=AGENT_ORDER,showfliers=False,patch_artist=True)
            for patch,a in zip(bp["boxes"],AGENT_ORDER):
                patch.set_facecolor(AGENT_COLORS[a]); patch.set_alpha(0.45)
            _rotate_xticks(axes[5])
            axes[5].set_title("Post-Fault Latency Stability")

            # 7 Recovery Trend
            for a in AGENT_ORDER:
                sub = day[day.agent==a].sort_values("episode")
                if not sub.empty:
                    axes[6].plot(sub["episode"], smooth_series(sub["processing_ms"],120),
                                 color=AGENT_COLORS[a],alpha=0.85,label=a)
            axes[6].set_title("Recovery Trend (Smoothed)")
            axes[6].legend(frameon=False,fontsize=8)

            # 8 CPU vs Latency
            for a in AGENT_ORDER:
                sub = sample_df(faults[faults.agent==a][["cpu_percent","processing_ms"]].dropna(),1200)
                axes[7].scatter(sub["cpu_percent"],sub["processing_ms"],s=8,alpha=0.28,color=AGENT_COLORS[a])
            axes[7].set_title("Fault Impact: CPU vs Latency")

            # 9 Heatmap
            if not faults.empty:
                pivot = faults.pivot_table(index="agent",columns="episode",aggfunc="size",fill_value=0).reindex(AGENT_ORDER)
                axes[8].imshow(pivot,aspect="auto",cmap="Reds")
                axes[8].set_yticks(range(len(AGENT_ORDER)))
                axes[8].set_yticklabels(AGENT_ORDER)
            axes[8].set_title("Fault Occurrence Distribution")

            plt.tight_layout()
            pdf.savefig(fig,bbox_inches="tight")
            plt.close(fig)

        # -------- EXTRA FAULT ANALYTICS (NO TITLE) --------
        with PdfPages(os.path.join(OUTPUT_DIR, f"faults_day{i}_extra.pdf")) as pdf:
            fig, axes = plt.subplots(3,3,figsize=(11,10))
            axes=axes.flatten()

            # 1 p95 Latency
            p95 = faults.groupby("agent")["processing_ms"].quantile(0.95).reindex(AGENT_ORDER).fillna(0)
            axes[0].bar(AGENT_ORDER,p95,color=[AGENT_COLORS[a] for a in AGENT_ORDER])
            axes[0].set_title("p95 Latency During Faults")
            add_bar_labels(axes[0])

            # 2 p95 CPU
            cpu95 = faults.groupby("agent")["cpu_percent"].quantile(0.95).reindex(AGENT_ORDER).fillna(0)
            axes[1].bar(AGENT_ORDER,cpu95,color=[AGENT_COLORS[a] for a in AGENT_ORDER])
            axes[1].set_title("p95 CPU During Faults")
            add_bar_labels(axes[1])

            # 3 Throughput
            thr = faults.groupby("agent")["throughput"].mean().reindex(AGENT_ORDER).fillna(0)
            axes[2].bar(AGENT_ORDER,thr,color=[AGENT_COLORS[a] for a in AGENT_ORDER])
            axes[2].set_title("Throughput During Faults (MB/s)")
            add_bar_labels(axes[2])

            # 4 Efficiency
            cpu_mean = faults.groupby("agent")["cpu_percent"].mean().reindex(AGENT_ORDER).replace(0,np.nan)
            eff = (thr / cpu_mean).fillna(0)
            axes[3].bar(AGENT_ORDER,eff,color=[AGENT_COLORS[a] for a in AGENT_ORDER])
            axes[3].set_title("Efficiency (Thr/CPU)")
            add_bar_labels(axes[3])

            # 5 Fault Type Composition
            stack = (faults.groupby(["agent","fault_type"]).size()
                     .unstack(fill_value=0).reindex(AGENT_ORDER).sort_index(axis=1))
            bottom = np.zeros(len(stack))
            for col in stack.columns:
                axes[4].bar(stack.index,stack[col],bottom=bottom,label=col)
                bottom+=stack[col]
            axes[4].legend(frameon=False,fontsize=7)
            axes[4].set_title("Fault Type Composition per Agent")
            _rotate_xticks(axes[4])

            # 6 Time to First Fault
            ttf = (faults.groupby("agent")["episode"].min().reindex(AGENT_ORDER)
                   - day.groupby("agent")["episode"].min().reindex(AGENT_ORDER)).fillna(0)
            axes[5].bar(AGENT_ORDER,ttf,color=[AGENT_COLORS[a] for a in AGENT_ORDER])
            axes[5].set_title("Time to First Fault (episodes)")
            add_bar_labels(axes[5])

            # 7 Mem vs Lat
            sub = sample_df(faults[["mem_percent","processing_ms","agent"]].dropna(),2000)
            for a in AGENT_ORDER:
                s = sub[sub.agent==a]
                axes[6].scatter(s["mem_percent"],s["processing_ms"],s=7,alpha=0.25,color=AGENT_COLORS[a])
            axes[6].set_title("Memory vs Latency During Faults")

            # 8 Fault Rate Normalized
            if "episode" in faults.columns:
                norm=[]
                for a in AGENT_ORDER:
                    d = day[day.agent==a]
                    if d.empty: norm.append(0); continue
                    span = max(1, d["episode"].max() - d["episode"].min())
                    norm.append((faults[faults.agent==a].shape[0] / span) * 1000)
                axes[7].bar(AGENT_ORDER,norm,color=[AGENT_COLORS[a] for a in AGENT_ORDER])
                axes[7].set_title("Faults per 1000 Episodes")
                add_bar_labels(axes[7])

            # 9 empty panel removed
            axes[8].axis("off")

            plt.tight_layout()
            pdf.savefig(fig,bbox_inches="tight")
            plt.close(fig)

# ----------------- Complexity Dashboard -----------------
def build_complexity_dashboard(merged):
    with PdfPages(os.path.join(OUTPUT_DIR,"complexity_dashboard.pdf")) as pdf:
        fig,axes=plt.subplots(3,3,figsize=(11,9.5))
        axes=axes.flatten()
        fig.suptitle("Computational Complexity (3×3)", fontsize=18)

        # 1 Mean Latency
        m = merged.groupby("agent")["processing_ms"].mean().reindex(AGENT_ORDER).fillna(0)
        axes[0].bar(AGENT_ORDER,m,color=[AGENT_COLORS[a] for a in AGENT_ORDER])
        axes[0].set_title("Mean Latency (ms)")
        add_bar_labels(axes[0])

        # 2 Latency Std
        s = merged.groupby("agent")["processing_ms"].std().reindex(AGENT_ORDER).fillna(0)
        axes[1].bar(AGENT_ORDER,s,color=[AGENT_COLORS[a] for a in AGENT_ORDER])
        axes[1].set_title("Latency Variability (Std)")
        add_bar_labels(axes[1])

        # 3 Mean CPU
        cpu = merged.groupby("agent")["cpu_percent"].mean().reindex(AGENT_ORDER).fillna(0)
        axes[2].bar(AGENT_ORDER,cpu,color=[AGENT_COLORS[a] for a in AGENT_ORDER])
        axes[2].set_title("Mean CPU (%)")
        add_bar_labels(axes[2])

        # 4 Mean Memory
        mem = merged.groupby("agent")["mem_percent"].mean().reindex(AGENT_ORDER).fillna(0)
        axes[3].bar(AGENT_ORDER,mem,color=[AGENT_COLORS[a] for a in AGENT_ORDER])
        axes[3].set_title("Mean Memory (%)")
        add_bar_labels(axes[3])

        # 5 Throughput
        thr = merged.groupby("agent")["throughput"].mean().reindex(AGENT_ORDER).fillna(0)
        axes[4].bar(AGENT_ORDER,thr,color=[AGENT_COLORS[a] for a in AGENT_ORDER])
        axes[4].set_title("Mean Throughput (MB/s)")
        add_bar_labels(axes[4])

        # 6 Efficiency
        eff = (thr / cpu.replace(0,np.nan)).fillna(0)
        axes[5].bar(AGENT_ORDER,eff,color=[AGENT_COLORS[a] for a in AGENT_ORDER])
        axes[5].set_title("Efficiency (Thr/CPU)")
        add_bar_labels(axes[5])

        # 7 Scaling Trend
        for p in PHASES:
            sub=merged[merged.phase==p]
            axes[6].plot(AGENT_ORDER, sub.groupby("agent")["processing_ms"].mean().reindex(AGENT_ORDER), "-o", label=p)
        axes[6].set_title("Scaling Trend Across Phases")
        axes[6].legend(frameon=False,fontsize=8)
        _rotate_xticks(axes[6])

        # 8 Boxplot Latency
        vals=[merged[merged.agent==a]["processing_ms"].dropna() for a in AGENT_ORDER]
        axes[7].boxplot(vals,labels=AGENT_ORDER,showfliers=False)
        _rotate_xticks(axes[7],35)
        axes[7].set_title("Latency Distribution")

        # 9 CPU vs Latency
        sub_all=sample_df(merged[["cpu_percent","processing_ms","agent"]].dropna(),6000)
        for a in AGENT_ORDER:
            s=sub_all[sub_all.agent==a]
            axes[8].scatter(s["cpu_percent"],s["processing_ms"],s=6,alpha=0.22,color=AGENT_COLORS[a])
        axes[8].set_title("CPU vs Latency")

        plt.tight_layout()
        pdf.savefig(fig,bbox_inches="tight")
        plt.close(fig)

# ----------------- Main -----------------
def main():
    print("Loading data...")
    merged=load_merged()
    print("Building FAULT dashboards...")
    build_fault_dashboards(merged)
    print("Building Complexity dashboard...")
    build_complexity_dashboard(merged)
    print("Done. Check figures/")

if __name__ == "__main__":
    main()
