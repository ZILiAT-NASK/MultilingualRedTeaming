import re

import pandas as pd


def count_by_quarter(df, date_col):

    dates = pd.to_datetime(df[date_col])
    quarters = ((dates.dt.month - 1) // 3 + 1).astype(str)
    years = dates.dt.year.astype(str)
    quarter_year = "Q" + quarters + "-" + years

    counts = (
        quarter_year.value_counts().rename_axis("quarter").reset_index(name="count")
    )

    start = dates.min().to_period("Q")
    end = dates.max().to_period("Q")
    all_quarters = pd.period_range(start, end, freq="Q")

    all_labels = ["Q" + str(p.quarter) + "-" + str(p.year) for p in all_quarters]
    df_q = pd.DataFrame({"quarter": all_labels})

    df_q = df_q.merge(counts, on="quarter", how="left").fillna(0)
    df_q["count"] = df_q["count"].astype(int)

    return df_q


def preprocess_data(data, min_count: int = 3) -> pd.DataFrame:
    """Load, clean, and standardize model names from Excel data."""

    items = data["LLM(s) Evaluated"].tolist()
    items_full = []
    for item in items:
        items_full.extend(re.findall(r"- ([^\n\r]+)", item))

    df = pd.DataFrame({"Model": items_full})

    def clean_model_name(name):
        name = re.sub(r"^.*?/", "", name)  # Remove repo prefixes
        name = re.sub(r"\s*\(.*?\)", "", name)  # Remove parentheses
        return name.strip().lower()

    df["Model"] = df["Model"].apply(clean_model_name)

    def standardize_name(name):
        mapping = {
            "llama": "llama",
            "gemini": "gemini",
            "mistral": "mistral",
            "mixtral": "mixtral",
            "qwen": "qwen",
            "palm": "palm",
            "gemma": "gemma",
            "phi": "phi",
            "claude": "claude",
            "aya": "aya",
            "command r": "command r",
            "command-r": "command r",
            "vicuna": "vicuna",
            "bloomz": "bloomz",
            "pythia": "pythia",
            "tulu": "tulu",
            "gpt-": "chatgpt",
            "chatgpt": "chatgpt",
            "yi-6b-chat": "yi",
            "swallow": "swallow",
            "baichuan": "baichuan",
            "wizardlm": "wizardlm",
            "minichat": "minichat",
        }
        for key, val in mapping.items():
            if key in name:
                return val
        return name

    df["Model"] = df["Model"].apply(standardize_name)

    model_counts = df["Model"].value_counts()
    top_models = model_counts[model_counts > min_count].index
    df = df[df["Model"].isin(top_models)]

    return df


language_families = {
    "Indo-European": [
        "Afrikaans",
        "Albanian",
        "Armenian",
        "BCMS",
        "Belarusian",
        "Bengali",
        "Bosnian",
        "Bulgarian",
        "Catalan",
        "Corsican",
        "Croatian",
        "Czech",
        "Danish",
        "Dutch",
        "English",
        "Estonian",
        "French",
        "Frisian",
        "Galician",
        "German",
        "Greek",
        "Gujarati",
        "Hindi",
        "Icelandic",
        "Irish",
        "Italian",
        "Kurdish",
        "Latin",
        "Latvian",
        "Lithuanian",
        "Luxembourgish",
        "Macedonian",
        "Marathi",
        "Nepali",
        "Norwegian",
        "Odia",
        "Pashto",
        "Persian",
        "Polish",
        "Portuguese",
        "Punjabi",
        "Romanian",
        "Russian",
        "Scots Gaelic",
        "Serbian",
        "Sindhi",
        "Sinhala",
        "Slovak",
        "Slovenian",
        "Spanish",
        "Swedish",
        "Tajik",
        "Ukrainian",
        "Urdu",
        "Welsh",
        "Yiddish",
    ],
    "Uralic": ["Finnish", "Hungarian"],
    "Afro-Asiatic": ["Amharic", "Arabic", "Hausa", "Hebrew", "Maltese", "Somali"],
    "Turkic": ["Azerbaijani", "Kazakh", "Kyrgyz", "Turkish", "Uyghur", "Uzbek"],
    "Language Isolate": [
        "Basque",
        "Korean",
    ],
    "Sino-Tibetan": ["Burmese", "Chinese", "Hmong"],
    "Austronesian": [
        "Cebuano",
        "Filipino",
        "Hawaiian",
        "Indonesian",
        "Javanese",
        "Malagasy",
        "Malay",
        "Maori",
        "Samoan",
        "Sundanese",
        "Tagalog",
    ],
    "Niger-Congo": [
        "Chichewa",
        "Igbo",
        "Nyanja",
        "Sesotho",
        "Shona",
        "Swahili",
        "Xhosa",
        "Yoruba",
        "Zulu",
    ],
    "Dravidian": ["Kannada", "Malayalam", "Tamil", "Telugu"],
    "Kra-Dai (Tai-Kadai)": ["Lao", "Thai"],
    "Mongolic": ["Mongolian"],
    "Austroasiatic": ["Khmer", "Vietnamese"],
    # "Unclassified": [
    #     "Esperanto",  # Constructed language, Indo-European-based
    #     "Haitian Creole"  # French-based creole
    # ],
    "Other": [
        "Guarani",  # Tupi-Guarani # Often treated as a language isolate, sometimes linked to Altaic
        "Esperanto",
        "Haitian Creole",
    ],
}
