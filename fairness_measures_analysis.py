import os
import sys
import random
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tabulate import tabulate

# Matplotlib config
matplotlib.use('Agg')

# Set up paths
project_root = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(project_root)

output_dir = os.path.join(project_root, 'outputs')
os.makedirs(output_dir, exist_ok=True)

# Constants for confusion matrix analysis
N = 30
K = 8
CM_FILE_PATH = os.path.join(output_dir, "confusion_matrix.parquet")

# Constants for charts
BIN_SIZE = 0.01
CUSTOM_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
METRIC_LABELS = {
    'acc': 'Accuracy',
    'pr': 'Precision',
    'rc': 'Recall'
}


def set_seed(seed=123):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_parquet_confusion_matrix(file_path):
    """Load confusion matrix combinations from a Parquet file."""
    return pd.read_parquet(file_path, engine='pyarrow')


def calculate_metrics(df):
    """Calculate classification and fairness metrics from the confusion matrix columns."""
    df['acc'] = abs((df[0] + df[3] + df[4] + df[7]) /
                    (df[0] + df[1] + df[2] + df[3] + df[4] + df[5] + df[6] + df[7]))

    df['pr'] = abs((df[0] + df[4]) / (df[0] + df[2] + df[4] + df[6]))

    df['rc'] = abs((df[0] + df[4]) / (df[0] + df[1] + df[4] + df[5]))

    df['ae'] = abs(((df[0] + df[3]) / (df[0] + df[1] + df[2] + df[3])) -
                   ((df[4] + df[7]) / (df[4] + df[5] + df[6] + df[7])))

    df['sp'] = abs(((df[0] + df[2]) / (df[0] + df[1] + df[2] + df[3])) -
                   ((df[4] + df[6]) / (df[4] + df[5] + df[6] + df[7])))

    df['eop'] = abs((df[0] / (df[0] + df[1])) - (df[4] / (df[4] + df[5])))

    df['pe'] = abs((df[2] / (df[2] + df[3])) - (df[6] / (df[6] + df[7])))

    return df


def bin_metrics(df, metrics):
    """Calculate classification and fairness metrics from the confusion matrix columns."""
    for m in metrics:
        df[f'{m}_bin'] = (df[m] / BIN_SIZE).round(0) * BIN_SIZE
    return df


def generate_heatmaps(df, main_metric, save_path):
    """Generate heatmaps and histograms for predictive performance metrics against fairness metrics."""
    # Create 2D distributions between the predictive performance metric and fairness metrics
    heatmaps = {
        'Accuracy Equality': df.groupby([f'{main_metric}_bin', 'ae_bin']).size().unstack(fill_value=0) / len(df),
        'Statistical Parity': df.groupby([f'{main_metric}_bin', 'sp_bin']).size().unstack(fill_value=0) / len(df),
        'Equal Opportunity': df.groupby([f'{main_metric}_bin', 'eop_bin']).size().unstack(fill_value=0) / len(df),
        'Predictive Equality': df.groupby([f'{main_metric}_bin', 'pe_bin']).size().unstack(fill_value=0) / len(df),
    }

    fig = plt.figure(figsize=(28, 8))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 5], hspace=0.05, wspace=0.2)
    axes_hist = [plt.subplot(gs[0, i]) for i in range(4)]  # Histograms
    axes_heat = [plt.subplot(gs[1, i]) for i in range(4)]  # Heatmaps
    cbar_ax = fig.add_axes((0.92, 0.1, 0.015, 0.6))  # Colorbar axis

    for i, (ax_hist, ax_heat, title) in enumerate(zip(axes_hist, axes_heat, heatmaps.keys())):
        data = heatmaps[title]

        sns.heatmap(
            data,
            cmap='mako',
            ax=ax_heat,
            cbar=ax_heat is axes_heat[-1],
            cbar_ax=cbar_ax if ax_heat is axes_heat[-1] else None,
            vmax=0.003 if main_metric == 'acc' else 0.001
        )

        ax_heat.set_xlabel(METRIC_LABELS.get(main_metric, main_metric), fontsize=14)
        ax_heat.set_ylabel(title, fontsize=14)

        # Format tick labels
        xticks = [j for j, val in enumerate(data.columns) if val in CUSTOM_TICKS]
        yticks = [j for j, val in enumerate(data.index) if val in CUSTOM_TICKS]
        xlabels = [f"{val:.2f}" for val in data.columns if val in CUSTOM_TICKS]
        ylabels = [f"{val:.2f}" for val in data.index if val in CUSTOM_TICKS]

        ax_heat.set_xticks(xticks)
        ax_heat.set_xticklabels(xlabels, rotation=45, fontsize=14)
        ax_heat.set_yticks(yticks)
        ax_heat.set_yticklabels(ylabels, fontsize=14)
        ax_heat.invert_yaxis()

        # Histogram above heatmap
        ax_hist.bar(data.columns, data.sum(axis=0), width=BIN_SIZE, color='gray')
        ax_hist.set_xlim(0, 1)
        ax_hist.set_xticks([])
        ax_hist.set_yticks([])
        for spine in ax_hist.spines.values():
            spine.set_visible(False)

    # Colorbar and layout
    cbar_ax.set_ylabel('Proportion of all results', fontsize=14)
    cbar_ax.tick_params(labelsize=14)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.1, wspace=0.2, hspace=0.05)

    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved heatmap '{os.path.basename(save_path)}' to: {save_path}\n")

    # Generate summary table with tabulate
    table = []
    for title, data in heatmaps.items():
        total = data.values.sum()
        high_total = data.loc[:, data.columns > 0.5].values.sum()
        percentage = high_total / total * 100
        table.append([title, f"{percentage:.4f}%"])

    print(tabulate(table,
                   headers=["Fairness Measure", f"% Results with {METRIC_LABELS.get(main_metric, main_metric)} > 0.5"]))
    print()  # newline after table


def main():
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")

    df = load_parquet_confusion_matrix(CM_FILE_PATH)

    df = calculate_metrics(df)
    df = bin_metrics(df, ['acc', 'pr', 'rc', 'ae', 'sp', 'eop', 'pe'])

    generate_heatmaps(df, 'acc', os.path.join(output_dir, "heatmaps_accuracy.png"))
    generate_heatmaps(df, 'pr', os.path.join(output_dir, "heatmaps_precision.png"))
    generate_heatmaps(df, 'rc', os.path.join(output_dir, "heatmaps_recall.png"))


if __name__ == "__main__":
    main()
