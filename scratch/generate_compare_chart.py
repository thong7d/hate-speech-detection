import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Set standard style for clean & premium graphics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

# Colors matching elegant modern dark/light themes
colors = {
    'Baseline (TF-IDF + LR)': '#94a3b8',        # Slate gray
    'PhoBERT Base': '#3b82f6',                 # Royal blue
    'XLM-R Base (Core Proposed)': '#10b981',   # Emerald green (Primary focus)
}

# Define models and their metrics extracted from real JSON reports
models = list(colors.keys())
metrics = ['Accuracy', 'Macro F1', 'CLEAN F1', 'OFFENSIVE F1', 'HATE F1']

# Standardized data values from the JSON files:
# Baseline: 80.53%, 62.50%, 89.65%, 41.23%, 56.61%
# PhoBERT: 85.11%, 63.36%, 92.41%, 41.06%, 56.60%
# XLM-R Base: 86.74%, 64.61%, 93.62%, 41.13%, 59.09%
# XLM-R + 4 Techs: 34.25%, 22.09%, 49.80%, 9.94%, 6.54%
data = {
    'Baseline (TF-IDF + LR)': [80.53, 62.50, 89.65, 41.23, 56.61],
    'PhoBERT Base': [85.11, 63.36, 92.41, 41.06, 56.60],
    'XLM-R Base (Core Proposed)': [86.74, 64.61, 93.62, 41.13, 59.09],
}

x = np.arange(len(metrics))
width = 0.18

# Plot bar series
for idx, model in enumerate(models):
    offset = (idx - 1.5) * width
    rects = ax.bar(x + offset, data[model], width, label=model, color=colors[model], edgecolor='none', alpha=0.9)
    
    # Add values on top of bars
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#334155')

# Title & labels styling (premium look)
ax.set_title('Bản đồ so sánh hiệu răng của các mô hình kiểm duyệt ViHSD', fontsize=16, fontweight='bold', pad=25, color='#1e293b')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12, fontweight='bold', color='#475569')
ax.set_ylabel('Ty le (%)', fontsize=12, fontweight='bold', color='#475569')
ax.set_ylim(0, 110)

# Grid and spines configuration
ax.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cbd5e1')
ax.spines['bottom'].set_color('#cbd5e1')

# Legend setup
ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', shadow=False, fontsize=10)

plt.tight_layout()

# Save final graphic to target directories
output_paths = [
    'd:/000MINHTHONG/Junior - Semester II/ANLPB/FinalProject/hate-speech-detection/results/compare_chart.png'
]

# Ensure directory exists
os.makedirs(os.path.dirname(output_paths[0]), exist_ok=True)

# Save
plt.savefig(output_paths[0], dpi=300, bbox_inches='tight')
print(f"Success: Comparison chart successfully generated and saved to: {output_paths[0]}")
