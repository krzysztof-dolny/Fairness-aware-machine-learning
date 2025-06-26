import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Set up paths
project_root = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(project_root)
output_dir = os.path.join(project_root, "outputs")
os.makedirs(output_dir, exist_ok=True)

# Constants
RESULTS_CSV_PATH = os.path.join(output_dir, 'results.csv')
COLUMNS_ORDER = ['ml_algorithm', 'fairness_mode', 'alpha_mode', 'alpha_value',
                 'test_accuracy', 'var_accuracy', 'fairness_score', 'var_fairness']


def load_and_prepare_data(path):
    """Load CSV data and prepare the DataFrame."""
    df = pd.read_csv(path)

    df['fairness_score'] = 1 - df['fairness_score']

    df = df.groupby(
        ['ml_algorithm', 'fairness_mode', 'alpha_mode', 'alpha_value']
    ).agg(
        test_accuracy=('test_accuracy', 'mean'),
        fairness_score=('fairness_score', 'mean'),
        var_accuracy=('test_accuracy', 'var'),
        var_fairness=('fairness_score', 'var')
    ).reset_index()

    return df


def display_grouped_results(df):
    """Group and display results by ML algorithm, fairness mode, and alpha mode."""
    for ml in df['ml_algorithm'].unique():
        for fairness in df['fairness_mode'].unique():
            for alpha in df['alpha_mode'].unique():
                subset = df[
                    (df['ml_algorithm'] == ml) &
                    (df['fairness_mode'] == fairness) &
                    (df['alpha_mode'] == alpha)
                ][COLUMNS_ORDER].copy()

                if not subset.empty:
                    subset[['test_accuracy', 'fairness_score']] = subset[['test_accuracy', 'fairness_score']].round(4)
                    print(f"\n=== ML Algorithm: {ml} | Fairness Mode: {fairness} | Alpha Mode: {alpha} ===")
                    print(tabulate(subset.sort_values(by='alpha_value'), headers='keys', tablefmt='psql', showindex=False))


def print_aggregated_stats(df):
    """Print statistical summary grouped by ML algorithm, fairness mode, and alpha mode."""
    for group in ['ml_algorithm', 'fairness_mode', 'alpha_mode']:
        print(f"\n=== Statistics grouped by {group} ===")
        stats = df.groupby(group)[['test_accuracy', 'fairness_score']].agg(['min', 'median', 'mean', 'var', 'max']).round(4)
        print(tabulate(stats, headers='keys', tablefmt='psql'))


def plot_line_charts(df, model_name):
    """Plot line charts for accuracy and fairness per alpha_value."""
    df_model = df[df['ml_algorithm'] == model_name]
    unique_alpha_modes = df_model['alpha_mode'].unique()
    unique_fairness_modes = ['AE', 'SP', 'EO', 'PE']

    for alpha_mode in unique_alpha_modes:
        # Create subplot for current alpha_mode
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

        # Filtering for certain alpha_mode
        df_alpha = df_model[df_model['alpha_mode'] == alpha_mode]

        # --- Accuracy ---
        for mode in unique_fairness_modes:
            subset = df_alpha[df_alpha['fairness_mode'] == mode]
            # axes[0].plot(subset['alpha_value'], subset['test_accuracy'], marker='o', label=mode)
            axes[0].errorbar(subset['alpha_value'], subset['test_accuracy'], yerr=subset['var_accuracy'],
                             fmt='o-', markersize=3, capsize=3, label=mode)

        axes[0].set_title(f"Accuracy vs Alpha for \"{alpha_mode}\" scheduling strategy", fontsize=14)
        axes[0].set_xlabel(r"$\alpha_0$", fontsize=14)
        axes[0].set_ylabel("AC", fontsize=14)
        axes[0].legend(title="Fairness criterion", loc='lower right', fontsize=12, title_fontsize=12)
        axes[0].grid(True)
        axes[0].set_xlim(-0.02, 1.02)
        axes[0].set_ylim(0.6, 1.02)
        axes[0].tick_params(axis='both', labelsize=12)

        # --- Fairness ---
        for mode in unique_fairness_modes:
            subset = df_alpha[df_alpha['fairness_mode'] == mode]
            # axes[1].plot(subset['alpha_value'], subset['fairness_score'], marker='o', label=mode)
            axes[1].errorbar(subset['alpha_value'], subset['fairness_score'], yerr=subset['var_fairness'],
                             fmt='o-', markersize=3, capsize=3, label=mode)

        axes[1].set_title(f"Fairness vs Alpha for \"{alpha_mode}\" scheduling strategy", fontsize=14)
        axes[1].set_xlabel(r"$\alpha_0$", fontsize=14)
        axes[1].set_ylabel("Fairness", fontsize=14)
        axes[1].legend(title="Fairness criterion", loc='lower right', fontsize=12, title_fontsize=12)
        axes[1].grid(True)
        axes[1].set_xlim(-0.02, 1.02)
        axes[1].set_ylim(0.6, 1.02)
        axes[1].tick_params(axis='both', labelsize=12)

        # --- Figure ---
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        save_path = os.path.join(output_dir, f"{model_name}_{alpha_mode}_line_charts.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved line chart for {model_name} model and {alpha_mode} scheduling strategy in {save_path}")


def is_pareto_efficient(df):
    """Determine Pareto-efficient points from accuracy and fairness score."""
    data = -np.column_stack((df['test_accuracy'], df['fairness_score']))
    is_efficient = np.ones(data.shape[0], dtype=bool)

    for i, c in enumerate(data):
        if is_efficient[i]:
            is_efficient[is_efficient] = (
                np.any(data[is_efficient] < c, axis=1) |
                np.all(data[is_efficient] == c, axis=1)
            )
            is_efficient[i] = True

    return is_efficient


def plot_scatter_charts(df, model_name):
    """Plot scatter chart for each fairness mode of a given model."""
    df_model = df[df['ml_algorithm'] == model_name].copy()
    df_model['score'] = df_model['test_accuracy'] + df_model['fairness_score']
    unique_fairness_modes = ['AE', 'SP', 'EO', 'PE']

    # Subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Color palette
    alpha_modes = df['alpha_mode'].unique()
    palette = sns.color_palette('tab10', len(alpha_modes))
    color_dict = dict(zip(alpha_modes, palette))

    for idx, mode in enumerate(unique_fairness_modes):
        ax = axes[idx]
        df_mode = df_model[df_model['fairness_mode'] == mode]

        # Drop outliers
        df_mode = df_mode[df_mode['score'] >= 1.5]

        # Pareto front calculation
        pareto_mask = is_pareto_efficient(df_mode)
        pareto_df = df_mode[pareto_mask].sort_values(by="test_accuracy")

        # Plot Pareto front
        ax.plot(pareto_df['test_accuracy'], pareto_df['fairness_score'],
                color='red', linewidth=2, zorder=2, label="_nolegend_")

        # Plot points
        for alpha_mode in alpha_modes:
            df_subset = df_mode[df_mode['alpha_mode'] == alpha_mode]
            ax.scatter(df_subset['test_accuracy'], df_subset['fairness_score'],
                       alpha=0.6, zorder=3, label=f"{alpha_mode}", color=color_dict[alpha_mode], s=25)

        ax.set_title(f"Accuracy vs Fairness for {mode}", fontsize=14)
        ax.set_xlabel("AC", fontsize=12)
        ax.set_ylabel(mode, fontsize=12)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True)
        ax.set_xlim(0.54, 0.86)
        ax.set_ylim(0.7, 1.02)
        ax.legend(title="$\\alpha$ scheduling strategy", loc='lower left', fontsize=11, title_fontsize=11)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    save_path = os.path.join(output_dir, f"{model_name}_scatter_charts.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved scatter chart for {model_name} model in {save_path}")


def main():
    df = load_and_prepare_data(RESULTS_CSV_PATH)

    display_grouped_results(df)
    print_aggregated_stats(df)

    for model_name in df['ml_algorithm'].unique():
        plot_line_charts(df, model_name)
        plot_scatter_charts(df, model_name)


if __name__ == "__main__":
    main()
