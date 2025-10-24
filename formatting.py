import pandas as pd

yes_no_colums = [
    "New or Modified Dataset(s)",
    "Includes Adversarial Examples?",
    "LLM as a Judge?",
]

categorical_columns = [
    "Publication Type",
    "Dataset size cat.",
    "Safety Label type",
    "Safety Classification Scope",
    "Benchmark vs. Fine-tuning",
    "Zero-shot vs. Fine-tuned Performance",
    "Human vs. Automated Evaluation",
    "Granularity of Safety Evaluation",
    "Dataset construction process",
]

palette = [
    "#DF4C4C",
    "#8F2746",
    "#5F1E40",
    "#3F1939",
    "#BF374A",
    "#AF3049",
    "#03051A",
    "#6F2142",
    "#0B0920",
    "#7F2444",
    "#160E27",
    "#22122D",
    "#301634",
    "#4F1B3D",
    "#9F2B47",
    "#CF404B",
]


def style_yesno(value):
    if pd.isna(value):
        color = "#f57c00"
    elif "no" in str(value).lower():
        color = "#d32f2f"
    elif "yes" in str(value).lower():
        color = "#388e3c"
    else:
        color = "#f57c00"
    return f'<span style="background:{color};color:white;padding:4px 8px;border-radius:12px;display:inline-block;">{value}</span>'


import itertools


def get_unique_parts(series):
    unique_parts = set()
    for val in series.dropna():
        for part in str(val).split(","):
            part_clean = part.strip()
            if part_clean:
                unique_parts.add(part_clean)
    return unique_parts


def build_color_map(series, palette=palette):
    """Build a dict: value -> color, based on unique comma-separated tokens in a Series."""
    # get unique parts
    unique_parts = get_unique_parts(series)
    # assign colors (cycle if needed)
    color_map = {}
    for part, color in zip(unique_parts, itertools.cycle(palette)):
        color_map[part] = color
    return color_map


def style_multi_general(value, color_map):
    """Return HTML for comma-separated items with colors from color_map."""
    if pd.isna(value):
        return ""
    parts = [p.strip() for p in str(value).split(",")]
    styled_parts = []
    for part in parts:
        color = color_map.get(part, "#9e9e9e")  # fallback gray
        styled_parts.append(
            f"<span style='background:{color};color:white;padding:4px 8px;border-radius:12px;margin-right:4px;display:inline-block;'>{part}</span>"
        )
    return " ".join(styled_parts)
