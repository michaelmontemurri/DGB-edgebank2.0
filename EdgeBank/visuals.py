import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV
df = pd.read_csv("edgebank_summary.csv")

# Normalize dataset names for display
df['dataset'] = df['dataset'].replace({
    'can parl': 'CanParl',
    'us legis.': 'USLegis',
    'un trade': 'UNtrade',
    'un vote': 'UNvote',
    'contact': 'Contacts',
    'socialevo': 'SocialEvo',
    'mooc': 'MOOC',
    'wikipedia': 'Wikipedia',
    'lastfm': 'LastFM',
    'enron': 'enron',
    'uci': 'uci'
})

# Desired x-axis dataset order
dataset_order = [
    "Wikipedia", "MOOC", "LastFM", "enron", "SocialEvo", "uci",
    "CanParl", "USLegis", "UNtrade", "UNvote", "Contacts"
]

# Adjust memory mode legend labels
df['mem_mode_legend'] = df['mem_mode'].replace({
    'freq_weight': 'freq_weight*',
    'window_freq_weight': 'window_freq_weight*',
    'unlim_mem': 'unlim_mem (EdgeBank_inf)',
    'time_window': 'time_window (EdgeBank_tw)'
})

# Ensure x-axis follows desired dataset order
df['dataset'] = pd.Categorical(df['dataset'], categories=dataset_order, ordered=True)

# Sort for visual clarity
df = df.sort_values(by=['dataset', 'au_roc_score_mean'], ascending=[True, False])

# Color palette (warm to cool)
color_palette = ['#e41a1c', '#ff7f00', '#ffbf00', '#a6cee3',
                 '#1f78b4', '#33a02c', '#6a3d9a', '#b2df8a',
                 '#fb9a99', '#cab2d6', '#ffff99', '#fdbf6f']

# Plot
plt.figure(figsize=(14, 7))
sns.barplot(
    data=df,
    x='dataset',
    y='au_roc_score_mean',
    hue='mem_mode_legend',
    palette=color_palette
)

# Axis and legend tweaks
plt.title("AUC-ROC of EdgeBank Memory Strategies by Dataset")
plt.xlabel("Dataset")
plt.ylabel("Mean AUC-ROC")
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title='Memory Strategy (* = new)', bbox_to_anchor=(1.02, 1), loc='upper left')

# Save figure
plt.tight_layout()
plt.savefig("edgebank_aucroc_custom_grouped.png", dpi=800)
