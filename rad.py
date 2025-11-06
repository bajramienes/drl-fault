import numpy as np
import matplotlib.pyplot as plt

labels = [
    "Container\nScheduling",
    "Energy\nOptimization",
    "Fault\nManagement",
    "Multi-Node /\nDistributed",
    "Real-Time\nBenchmark",
    "Reproducible\nSetup"
]

data = {
    "Xiao et al. (2024)":      [1,0,0,1,0,0],
    "Safavifar et al. (2024)": [1,1,0,1,0,0],
    "Nagarajan et al. (2025)": [1,1,1,0,0,0],
    "Verma et al. (2025)":     [0,0,1,1,1,1],
    "Our Work":                [1,1,1,1,1,1]
}

# Softer, eventone palette
colors = [
    "#4C72B0",  # blue
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#E17C05"   # orange (Our Work)
]

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111, polar=True)

for (name, values), color in zip(data.items(), colors):
    values = values + values[:1]
    ax.plot(angles, values, linewidth=2.4, color=color, label=name)
    ax.fill(angles, values, color=color, alpha=0.18)

# Move labels outward so they do not overlap lines
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)
for label, angle in zip(ax.get_xticklabels(), angles):
    label.set_horizontalalignment('center')
    label.set_verticalalignment('center')

# Remove radial grid text, reduce clutter
ax.set_yticklabels([])
ax.set_ylim(0, 1)

ax.grid(color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

# Title + legend formatting
ax.set_title("Comparative Capability Coverage Across Representative Approaches",
             fontsize=15, pad=25)

ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.12), fontsize=11, frameon=False)

plt.tight_layout()
plt.savefig("figures/radar_literature_comparison.pdf", dpi=300, bbox_inches="tight")
plt.show()
