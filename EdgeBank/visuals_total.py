import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load summary
df = pd.read_csv("edgebank_summary.csv")

# Add legend labels with asterisks and aliases
df['mem_mode_legend'] = df['mem_mode'].replace({
    'freq_weight': 'freq_weight*',
    'window_freq_weight': 'window_freq_weight*',
    'unlim_mem': 'unlim_mem (EdgeBank_inf)',
    'time_window': 'time_window (EdgeBank_tw)'
})

# Aggregate mean and std for each memory mode
grouped = df.groupby('mem_mode_legend').agg(
    mean_auc=('au_roc_score_mean', 'mean'),
    std_auc=('au_roc_score_mean', 'std')
).reset_index()

# Sort by mean for display
grouped = grouped.sort_values(by='mean_auc', ascending=False)

# Colors: warm to cool
color_palette = [
    '#e41a1c', '#ff7f00', '#ffbf00', '#a6cee3',
    '#1f78b4', '#33a02c', '#6a3d9a', '#b2df8a',
    '#fb9a99', '#cab2d6', '#ffff99', '#fdbf6f'
]

# Plot
plt.figure(figsize=(12, 6))
x = np.arange(len(grouped))
bars = plt.bar(
    x,
    grouped['mean_auc'],
    yerr=grouped['std_auc'],
    capsize=5,
    color=color_palette[:len(grouped)],
    edgecolor='black'
)

# Axis and labels
plt.xticks(ticks=x, labels=grouped['mem_mode_legend'], rotation=30, ha='right')
plt.title("Average AUC-ROC by Memory Strategy Across All Datasets")
plt.xlabel("Memory Strategy (* = new)")
plt.ylabel("Mean AUC-ROC")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Final layout
plt.tight_layout()
plt.savefig("edgebank_aucroc_memory_averages.png", dpi=800)
