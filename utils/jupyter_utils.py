import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from typing import Optional, List


def show_alpha_vs_beta(high_beta_positive_alpha: pd.DataFrame, df: pd.DataFrame):
    """
    Creates a scatter plot showing the relationship between Alpha and Beta for all tokens,
    highlighting the top 5 tokens with the highest Beta values.

    Args:
        high_beta_positive_alpha (pd.DataFrame): DataFrame containing tokens with high Beta and positive Alpha values.
        df (pd.DataFrame): DataFrame containing all tokens' Alpha and Beta values.

    Returns:
        plt.Figure: The matplotlib plot object with the scatter plot.
    """
    plt.figure(figsize=(12, 5))
    top_5_tokens = high_beta_positive_alpha.nlargest(5, "Beta")

    # Scatter plot for tokens not in top 5
    plt.scatter(
        df[~df.index.isin(top_5_tokens.index)]["Alpha"],
        df[~df.index.isin(top_5_tokens.index)]["Beta"],
        label="Other Tokens",
        color="blue",
    )

    plt.scatter(
        top_5_tokens["Alpha"],
        top_5_tokens["Beta"],
        color="red",
        label="Top 5 Tokens",
        marker="o",
    )

    for _, token in top_5_tokens.iterrows():
        plt.annotate(
            token.name,
            (token["Alpha"], token["Beta"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.xlabel("Alpha")
    plt.ylabel("Beta")
    plt.title("Alpha vs Beta for all tokens")
    plt.legend()

    return plt


def show_beta_distribution_box_plot(df: pd.DataFrame):
    """
    Creates a box plot and a violin plot to visualize the distribution of Beta values across all tokens.

    Args:
        df (pd.DataFrame): DataFrame containing Beta values for all tokens.

    Returns:
        plt.Figure: The matplotlib plot object with the box and violin plots.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(x="Beta", data=df)
    plt.title("Beta Values for all tokens")

    plt.subplot(1, 2, 2)
    sns.violinplot(x="Beta", data=df)
    plt.title("Beta Values for all tokens")

    plt.tight_layout()

    return plt


def show_kde(df: pd.DataFrame, hue: Optional[str] = None):
    """
    Creates KDE (Kernel Density Estimate) plots for Beta and Alpha values, optionally segmented by a hue variable.

    Args:
        df (pd.DataFrame): DataFrame containing Beta and Alpha values for all tokens.
        hue (Optional[str]): Optional column name in the DataFrame to use for segmentation in the KDE plot.

    Returns:
        plt.Figure: The matplotlib plot object with the KDE plots.
    """
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metric1 = "Beta"
    metric2 = "Alpha"

    # Plot for the first metric (left side)
    (
        sns.kdeplot(data=df, x=metric1, hue=hue, fill=True, ax=axes[0])
        if hue
        else sns.kdeplot(df[metric1], fill=True, ax=axes[0])
    )
    axes[0].set_xlabel(metric1)
    axes[0].set_ylabel("Probability Density")
    axes[0].set_title(f"Distribution of {metric1} Values")

    # Plot for the second metric (right side)
    (
        sns.kdeplot(data=df, x=metric2, hue=hue, fill=True, ax=axes[1])
        if hue
        else sns.kdeplot(df[metric2], fill=True, ax=axes[1])
    )
    axes[1].set_xlabel(metric2)
    axes[1].set_ylabel("Probability Density")
    axes[1].set_title(f"Distribution of {metric2} Values")

    plt.tight_layout()

    return plt


def show_timeseries_plot(df: pd.DataFrame, tokens: List[str], metric: str):
    """
    Creates a timeseries plot of the specified metric (Alpha or Beta) values for the given tokens over a 60-day rolling average.

    Args:
        df (pd.DataFrame): DataFrame containing the timeseries of Alpha or Beta values with a 60-day rolling average.
        tokens (List[str]): List of token symbols to be plotted.
        metric (str): The metric to be plotted, either "Alpha" or "Beta".

    Returns:
        plt.Figure: The matplotlib figure object with the timeseries plot.
    """
    plt.figure(figsize=(12, 5))

    for _, token in enumerate(tokens):
        plt.plot(df.index, df[token], label=token)

    plt.xlabel("Date")
    plt.ylabel(f"{metric}, 60 day Rolling Avg")
    plt.title(f"{metric} vs. Ether Timeseries Plot")
    plt.legend()
    plt.grid(True)

    return plt


def show_pct_distribution(df: pd.DataFrame):
    """
    Displays a 3x3 grid where some cells are combinations of others and shows percentage distributions.

    Args:
        df (pd.DataFrame): DataFrame containing the beta and alpha values.
    """
    # Define conditions for each cell
    conditions = {
        "1_0": (df["Beta"] < 1) & (df["Alpha"] < 0),
        "1_1": (df["Beta"] >= 1) & (df["Alpha"] < 0),
        "2_0": (df["Beta"] < 1) & (df["Alpha"] >= 0),
        "2_1": (df["Beta"] >= 1) & (df["Alpha"] >= 0),
    }

    # Calculate percentage distribution for each cell
    total_count = len(df)
    percentages = {
        key: 100 * len(df[condition]) / total_count
        for key, condition in conditions.items()
    }

    percentages.update(
        {
            "0_0": percentages["1_0"] + percentages["2_0"],
            "0_1": percentages["1_1"] + percentages["2_1"],
            "1_2": percentages["1_0"] + percentages["1_1"],
            "2_2": percentages["2_0"] + percentages["2_1"],
        }
    )

    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(3, 3, width_ratios=[4, 4, 2.5], height_ratios=[2.5, 4, 4])

    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    ax5 = plt.subplot(gs[1, 2])
    ax6 = plt.subplot(gs[2, 0])
    ax7 = plt.subplot(gs[2, 1])
    ax8 = plt.subplot(gs[2, 2])

    axes = [ax0, ax1, None, ax3, ax4, ax5, ax6, ax7, ax8]

    labels = {
        (0, 0): f"Beta < 1\n {percentages['0_0']:.1f}%",
        (0, 1): f"Beta > 1\n{percentages['0_1']:.1f}%",
        (1, 0): f"{percentages['1_0']:.1f}%",
        (1, 1): f"{percentages['1_1']:.1f}%",
        (1, 2): f"Alpha < 0 \n{percentages['1_2']:.1f}%",
        (2, 0): f"{percentages['2_0']:.1f}%",
        (2, 1): f"{percentages['2_1']:.1f}%",
        (2, 2): f"Alpha > 0 \n{percentages['2_2']:.1f}%",
    }

    for i in range(3):
        for j in range(3):
            if i == 0 and j == 2:
                break
            ax = axes[i * 3 + j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if (i, j) in labels:
                ax.text(0.5, 0.5, labels[(i, j)], ha="center", va="center")

    axes[3].set_ylabel("< 0", labelpad=15, rotation=0)
    axes[6].set_ylabel("> 0", labelpad=15, rotation=0)

    axes[6].set_xlabel(" < 1", labelpad=15)
    axes[7].set_xlabel(" > 1", labelpad=15)

    fig.text(0.42, 0.0, "Beta", ha="center", va="center", fontsize=14)
    fig.text(0.0, 0.4, "Alpha", ha="center", va="center", fontsize=14)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return plt
