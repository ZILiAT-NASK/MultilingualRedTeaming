import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Moved import to top
import pandas as pd
import seaborn as sns
import streamlit as st

from src.utils import count_by_quarter, language_families, preprocess_data

# --- Module-level Constants ---

# Load taxonomy data once
try:
    TAXONOMY_DF = pd.read_csv("data/taxonomy.csv")
    # Create a lookup dictionary for faster processing
    TAXONOMY_LOOKUP = TAXONOMY_DF.set_index("language")["class"].to_dict()
except FileNotFoundError:
    st.error("Error: 'data/taxonomy.csv' not found. Language class plot will fail.")
    TAXONOMY_DF = pd.DataFrame(columns=["language", "class"])
    TAXONOMY_LOOKUP = {}

# Languages to skip in taxonomy processing
HARDCODED_SKIP_LANGS = {"kyrgyz", "nyanja", "filipino", "myanmar", "bcms"}


# --- Plotting Setup ---


def setup_plot_style():
    """Sets the global matplotlib and seaborn style for all plots."""
    sns.set(style="whitegrid")
    sns.set_palette("rocket")  # Set default palette
    plt.rcParams.update(
        {
            "figure.max_open_warning": 0,
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "font.size": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "lines.linewidth": 2,
            "text.usetex": False,
            "pgf.rcfonts": False,
        }
    )


# Apply the style settings immediately when the module is imported
setup_plot_style()


# --- Internal Helper Functions ---


def _plot_simple_value_counts(
    data_series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str = "Number of Papers",
    figsize: tuple = (6, 4),
    rotation: int = 45,
    ha: str = "right",
):
    """
    Internal helper to plot value_counts for a given series,
    filtering out '-' placeholders.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Process counts and filter out the placeholder
    counts = data_series.value_counts()
    counts = counts[counts.index != "-"]

    sns.barplot(x=counts.index, y=counts.values, palette="rocket", ax=ax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=14, fontweight="bold")

    if rotation > 0:
        plt.xticks(rotation=rotation, ha=ha)
    else:
        plt.xticks(rotation=rotation, ha="center")

    plt.tight_layout()
    return fig


def _get_new_dataset_papers(df: pd.DataFrame) -> pd.DataFrame:
    """Filters DataFrame for papers with new or modified datasets."""
    # Drop rows where 'New or Modified Dataset(s)' is NaN before filtering
    df_filtered = df.dropna(subset=["New or Modified Dataset(s)"])
    # Return a copy to avoid SettingWithCopyWarning
    return df_filtered[df_filtered["New or Modified Dataset(s)"] == "YES"].copy()


def _get_families_for_lang_list(lang_list: list) -> set:
    """Helper for .apply() to find language families from a list of languages."""
    if not isinstance(lang_list, list):
        return set()  # Handle potential NaNs or non-list data

    families = set()
    for lang in lang_list:
        lang = lang.strip()
        # Find the family for the language
        family = next((k for k, v in language_families.items() if lang in v), None)
        if family:  # Only add if a family is found
            families.add(family)
    return families


def _get_classes_from_lookup(lang_list: list) -> set:
    """Helper for .apply() to find language classes using the global lookup."""
    if not isinstance(lang_list, list):
        return set()

    classes = set()
    for lang in lang_list:
        lang = lang.strip().lower()
        if lang not in HARDCODED_SKIP_LANGS:
            lang_class = TAXONOMY_LOOKUP.get(lang)  # Use fast dictionary lookup
            if lang_class:
                classes.add(lang_class)
    return classes


def _clean_measure(measure: str) -> list:
    """Internal helper to clean and normalize evaluation measure strings."""
    parts = re.split(r"[\n;]| - ", str(measure))
    cleaned = []
    for part in parts:
        part = part.strip().lstrip("-").strip()
        part = re.sub(r"\s*\([A-Za-z0-9]+\)", "", part)  # Remove acronyms in parens

        if not part:
            continue

        # Standardize capitalization
        if part.isupper() or len(part) <= 4:
            part = part.upper()
        else:
            part = part.title()

        cleaned.append(part)
    return cleaned


# --- Plotting Functions ---


def plot_human_vs_auto_eval(data: pd.DataFrame) -> plt.Figure:
    """Generates the Human vs. Automated Evaluation plot."""
    return _plot_simple_value_counts(
        data["Human vs. Automated Evaluation"],
        title="Human vs. Automated Evaluation",
        xlabel="Evaluation Type",
    )


def plot_classification_scope(data: pd.DataFrame) -> plt.Figure:
    """Generates the Safety Classification Scope plot."""
    return _plot_simple_value_counts(
        data["Safety Classification Scope"],
        title="Safety Classification Scope",
        xlabel="Classification Scope",
    )


def plot_benchmark_vs_finetuning(data: pd.DataFrame) -> plt.Figure:
    """Generates the Benchmark vs. Fine-tuning plot."""
    return _plot_simple_value_counts(
        data["Benchmark vs. Fine-tuning"],
        title="Benchmark vs. Fine-tuning",
        xlabel="Approach Type",
    )


def plot_zero_vs_finetuned(data: pd.DataFrame) -> plt.Figure:
    """Generates the Zero-shot vs. Fine-tuned Performance plot."""
    return _plot_simple_value_counts(
        data["Zero-shot vs. Fine-tuned Performance"],
        title="Zero-shot vs. Fine-tuned Performance",
        xlabel="Performance Type",
    )


def plot_llm_as_judge(data: pd.DataFrame) -> plt.Figure:
    """Generates the LLM as a Judge? plot."""
    return _plot_simple_value_counts(
        data["LLM as a Judge?"].str.upper(),
        title="LLM as a Judge?",
        xlabel="LLM as Judge?",
        rotation=0,
        ha="center",
    )


def plot_safety_granularity(data: pd.DataFrame) -> plt.Figure:
    """Generates the Granularity of Safety Evaluation plot."""
    fig, ax = plt.subplots(figsize=(6, 4))

    granularity_counts = (
        data["Granularity of Safety Evaluation"]
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .str.lower()
        .value_counts()
    )
    granularity_counts = granularity_counts[granularity_counts.index != "-"]

    sns.barplot(
        x=granularity_counts.index, y=granularity_counts.values, palette="rocket", ax=ax
    )
    ax.set_ylabel("Number of Papers")
    ax.set_xlabel("Granularity Type")
    ax.set_title("Granularity of Safety Evaluation", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_common_evaluation_measures(data: pd.DataFrame) -> plt.Figure:
    """Generates the Common Evaluation Measures plot."""
    all_measures = data["Evaluation measure"].dropna().apply(_clean_measure).explode()
    measure_counts = all_measures.value_counts().head(20)  # Top 20 measures

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x=measure_counts.index, y=measure_counts.values, palette="rocket", ax=ax
    )
    ax.set_xlabel("Evaluation Measure")
    ax.set_ylabel("Number of Papers")
    ax.set_title("Common Evaluation Measures (Top 20)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=60, ha="right", fontsize=13)
    plt.tight_layout()
    return fig


def plot_publication_trend(data: pd.DataFrame) -> plt.Figure:
    """
    Parses MM/YYYY dates, converts them to Q# - YYYY format,
    and generates a trend plot sorted from oldest to newest.
    """
    df_temp = data.copy()
    df_q = count_by_quarter(df_temp, "Publication Date\nMM/YYYY")

    palette = sns.color_palette("rocket", len(df_q))
    palette.reverse()  # Specific to this plot

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x="quarter", y="count", data=df_q, palette=palette, ax=ax)

    ax.set_xlabel("Publication Quarter", fontsize=14)
    ax.set_ylabel("Number of Papers", fontsize=14)
    ax.set_title("Publication Trend by Quarter", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=75, labelsize=10)

    plt.tight_layout()
    return fig


def plot_model_counts(data: pd.DataFrame) -> plt.Figure:
    """Plot and save bar chart of model counts."""
    processed_data = preprocess_data(data)  # Assumes preprocess_data is necessary
    model_counts = processed_data["Model"].value_counts().reset_index()
    model_counts.columns = ["Model", "Count"]

    colors = sns.color_palette("rocket", n_colors=len(model_counts))

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x="Model", y="Count", data=model_counts, palette=colors, ax=ax)
    ax.set_title("Model Counts", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="center", fontsize=10)
    plt.tight_layout()
    return fig


def plot_number_of_languages_in_dataset(df: pd.DataFrame) -> plt.Figure:
    """Plots the distribution of the number of languages evaluated in new datasets."""
    df_langs = _get_new_dataset_papers(df)

    # Filter for relevant data
    df_langs = df_langs.dropna(subset=["# Langs", "Languages Covered"])

    # Count occurrences
    counts = df_langs["# Langs"].value_counts().reset_index()
    counts.columns = ["# Langs", "Count"]
    counts["# Langs"] = counts["# Langs"].astype(int)
    counts = counts.sort_values(by="# Langs", ascending=True)

    colors = sns.color_palette("rocket", n_colors=len(counts))

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x="# Langs", y="Count", data=counts, palette=colors, ax=ax)

    ax.set_title(
        "Distribution of number of languages evaluated", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Number of languages in dataset")
    ax.set_ylabel("Count")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    return fig


def frequency_distribution_top_ten_languages(df: pd.DataFrame) -> plt.Figure:
    """Plots the top 10 most frequent non-English languages in new datasets."""
    df_langs = _get_new_dataset_papers(df)

    # Process languages
    df_langs["Languages Covered"] = df_langs["Languages Covered"].str.split(",")
    exploded = df_langs.explode("Languages Covered")
    exploded["Languages Covered"] = exploded["Languages Covered"].str.strip()

    # Get value counts, take top 11, and skip the first one (assumed to be English)
    counts = exploded["Languages Covered"].value_counts().reset_index().head(11)[1:]
    counts.columns = ["Languages Covered", "Count"]

    colors = sns.color_palette("rocket", n_colors=len(counts))

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x="Languages Covered", y="Count", data=counts, palette=colors, ax=ax)

    ax.set_xlabel("Language")
    ax.set_ylabel("Number of datasets")
    ax.set_title(
        "Ten most frequent non-English languages", fontsize=14, fontweight="bold"
    )

    plt.xticks(fontsize=13, rotation=30, ha="right")
    plt.yticks(fontsize=13)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))  # Use imported ticker

    plt.tight_layout()
    return fig


def distribution_of_dataset_accross_langs_families(df: pd.DataFrame) -> plt.Figure:
    """Plots the distribution of new datasets across language families."""
    df_langs = _get_new_dataset_papers(df)
    df_langs["Languages Covered"] = df_langs["Languages Covered"].str.split(",")

    # Use fast .apply() method instead of iterrows()
    df_langs["families_set"] = df_langs["Languages Covered"].apply(
        _get_families_for_lang_list
    )

    # Count all occurrences of each family
    all_families = [
        family for families_set in df_langs["families_set"] for family in families_set
    ]
    family_counts = Counter(all_families)
    family_counts_df = pd.DataFrame(
        family_counts.items(), columns=["Language Family", "Count"]
    )

    # Calculate percentage
    total_rows = len(df_langs)
    family_counts_df["Percentage"] = (family_counts_df["Count"] / total_rows) * 100
    family_counts_df = family_counts_df.sort_values(by="Percentage", ascending=False)

    colors = sns.color_palette("rocket", n_colors=len(family_counts_df))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=family_counts_df,
        x="Language Family",
        y="Percentage",
        palette=colors,
        ax=ax,
    )

    plt.xticks(fontsize=10, rotation=45, ha="right")
    ax.set_ylabel("Percentage of datasets")
    ax.set_xlabel("Language family")
    ax.set_title(
        "Distribution of datasets across language families",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def percentage_of_datasets_in_each_class(df: pd.DataFrame) -> plt.Figure:
    """Plots the percentage of new datasets that evaluate languages from each class."""
    df_langs = _get_new_dataset_papers(df)
    df_langs["Languages Covered"] = df_langs["Languages Covered"].str.split(",")

    # Use fast .apply() method with the pre-computed lookup dictionary
    df_langs["Article class set"] = df_langs["Languages Covered"].apply(
        _get_classes_from_lookup
    )

    # Count all occurrences of each class
    all_classes = [
        cls for class_set in df_langs["Article class set"] for cls in class_set
    ]
    class_counts = Counter(all_classes)
    class_counts_df = pd.DataFrame(
        class_counts.items(), columns=["Article class set", "Count"]
    )

    # Calculate percentage
    total_rows = len(df_langs)
    class_counts_df["Percentage"] = (class_counts_df["Count"] / total_rows) * 100
    class_counts_df = class_counts_df.sort_values(by="Percentage", ascending=False)

    colors = sns.color_palette("rocket", n_colors=len(class_counts_df))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=class_counts_df,
        x="Article class set",
        y="Percentage",
        palette=colors,
        ax=ax,
    )

    plt.xticks(fontsize=13, ha="right")
    ax.set_ylabel("Percentage of datasets")
    ax.set_xlabel("Class of language")
    ax.set_title(
        "Percentage of datasets evaluating languages", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    return fig
