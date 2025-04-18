import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("edgebank_summary.csv")

df['method'] = df['mem_mode'] + "_" + df['w_mode'].replace({'': 'none'})
df = df.sort_values(by='au_roc_score_mean', ascending=False)


plt.figure(figsize=(12, 6))
sns.barplot(
    data=df,
    x='method',
    y='au_roc_score_mean',
    palette='Set2'
)

# Add plot details
plt.title("AUC-ROC of EdgeBank Strategies on Wikipedia (Standard NS)")
plt.xlabel("EdgeBank Strategy")
plt.ylabel("Mean AUC-ROC")
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# Show plot
plt.show()