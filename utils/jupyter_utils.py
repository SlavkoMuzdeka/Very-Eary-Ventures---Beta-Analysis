import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Optional, List


def show_alpha_vs_beta(high_beta_positive_alpha: pd.DataFrame, df: pd.DataFrame):
    """
    Creates a scatter plot showing the relationship between Alpha and Beta for all tokens,
    highlighting the top 5 tokens with the highest Beta values.

    Args:
        high_beta_positive_alpha (pd.DataFrame): DataFrame containing tokens with high Beta and positive Alpha values.
        df (pd.DataFrame): DataFrame containing all tokens' Alpha and Beta values.

    Returns:
        plt: The matplotlib plot object with the scatter plot.
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
        plt: The matplotlib plot object with the box and violin plots.
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
        plt: The matplotlib plot object with the KDE plots.
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


def show_timeseries_plot(beta_df: pd.DataFrame, tokens: List[str]):
    """
    Creates a timeseries plot of the Beta values for the top 5 tokens with the highest Beta over a 60-day rolling average.

    Args:
        df (pd.DataFrame): DataFrame containing the Beta values of all tokens.
        beta_df (pd.DataFrame): DataFrame containing the timeseries of Beta values with a 60-day rolling average.

    Returns:
        plt: The matplotlib plot object with the timeseries plot.
    """
    plt.figure(figsize=(12, 5))

    for _, token in enumerate(tokens):
        plt.plot(beta_df.index, beta_df[token], label=token)

    plt.xlabel("Date")
    plt.ylabel("Beta, 60 day Rolling Avg")
    plt.title("Beta vs. Ether Timeseries Plot")
    plt.legend()
    plt.grid(True)

    return plt
