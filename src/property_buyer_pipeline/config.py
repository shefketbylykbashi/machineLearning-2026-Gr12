"""
Configuration and constants for the Property Buyer Pipeline ETL.
"""
from pathlib import Path

# =========================================================
# Pipeline Version
# =========================================================
PIPELINE_VERSION = "v1.0"
DESCRIPTION = """
v1.0 FINAL
- Fixes constant area issue
- Improves mixed-asset detection
- Avoids artificial price_per_* fills
- Keeps full preparation pipeline
"""

# =========================================================
# Data Paths
# =========================================================
INPUT_FILE = "./data/raw/sales_with_converted_rents_enriched_with_arbk.xlsx"
DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "processed" / "property_buyer"
MODELS_DIR = DATA_DIR / "models" / "property_buyer"

# =========================================================
# Column Names
# =========================================================
TARGET_COL_RAW = "Blerësi"
TARGET_COL = "bleresi"
PROFILE_COL = "buyer_profile"
MANDATORY_BUSINESS_COL = "arbk_nllojibiznesitid"

# =========================================================
# Important Columns for Data Quality
# =========================================================
LEAKAGE_COLUMNS = [
    "arbk_telefoni",
    "arbk_email",
    "arbk_webfaqja",
    "arbk_emribiznesit_gjetur",
    "arbk_matchfield",
    "arbk_matchscore",
    "arbk_datashuarjesbiznesit",
]

NUMERIC_LIKE_COLUMNS = [
    "cmimi_i_shitjes_se_asetit",
    "siperfaqja_e_objektit_nese_ka_objekt_te_toka_dhe_dihet_siperfaqja_m2",
    "siperfaqja_e_tokes_ne_metra_katror",
    "arbk_nllojibiznesitid",
    "arbk_numpunetoreve",
    "arbk_kapitali",
    "arbk_pronari_1_kapitali",
    "arbk_pronari_1_kapitaliperqindje",
    "arbk_aktiviteti_1_kodinace",
    "arbk_aktiviteti_2_kodinace",
    "arbk_aktiviteti_3_kodinace",
    "arbk_aktiviteti_4_kodinace",
    "arbk_aktiviteti_5_kodinace",
    "nr",
]

DATE_COLUMNS = ["data_e_kontrates", "arbk_dataregjistrimit"]

AREA_COLUMNS = {
    "object": "siperfaqja_e_objektit_nese_ka_objekt_te_toka_dhe_dihet_siperfaqja_m2",
    "land": "siperfaqja_e_tokes_ne_metra_katror",
}

PRICE_COLUMN = "cmimi_i_shitjes_se_asetit"

PROTECTED_COLUMNS = {
    TARGET_COL,
    PROFILE_COL,
    MANDATORY_BUSINESS_COL,
    "cmimi_i_shitjes_se_asetit",
    "siperfaqja_e_objektit_nese_ka_objekt_te_toka_dhe_dihet_siperfaqja_m2",
    "siperfaqja_e_tokes_ne_metra_katror",
    "total_area_m2",
    "price_per_total_m2",
    "price_per_land_m2",
    "price_per_object_m2",
    "log_sale_price",
    "is_land_only",
    "has_object",
    "asset_structure_type",
    "area_price_interaction",
    "raw_has_object_area",
    "raw_has_land_area",
    "siperfaqja_e_objektit_nese_ka_objekt_te_toka_dhe_dihet_siperfaqja_m2_was_missing",
    "siperfaqja_e_tokes_ne_metra_katror_was_missing",
}

IDENTIFIER_COLUMNS = [
    "nr",
    "pak_id",
    "ndermarrja_shoqerore",
]

# =========================================================
# Data Quality Thresholds
# =========================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_TARGET_FREQUENCY = 1
SPARSE_COL_THRESHOLD = 0.85
DOMINANT_VALUE_THRESHOLD = 0.85
AREA_GROUP_MIN_SIZE = 5
GENERIC_GROUP_MIN_SIZE = 4

# =========================================================
# Model Parameters
# =========================================================
FEATURE_SELECTOR_PARAMS = {
    "n_estimators": 250,
    "max_depth": 16,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

FINAL_MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# =========================================================
# Categorical Defaults
# =========================================================
CATEGORICAL_DEFAULTS = {
    "arbk_emribiznesit": "Biznes i panjohur",
    "arbk_nllojibiznesit": "Lloj biznesi i panjohur",
    "arbk_statusiarbk": "Status i panjohur",
    "arbk_statusi": "Status i panjohur",
    "komuna_lokacioni_i_asetit_te_shitur": "Komune e panjohur",
    "qytet_apo_fshat_lokacioni_i_asetit_te_shitur": "Lokacion i panjohur",
    "zona_kadastrale": "Zone kadastrale e panjohur",
    "kategoria_e_asetit": "Kategori e panjohur",
    "menyra_e_shitjes": "Menyre e panjohur",
    "eshte_shitur_si_ndermarrje_e_re_apo_aset_ne_likuidim": "Te panjohura",
}

ZERO_FILL_COLUMNS = {
    "arbk_kapitali",
    "arbk_pronari_1_kapitali",
    "arbk_pronari_1_kapitaliperqindje",
    "arbk_numpunetoreve",
    "cmimi_i_shitjes_se_asetit",
}

# =========================================================
# Output File Names
# =========================================================
OUTPUT_FILES = {
    "cleaned_dataset": f"cleaned_dataset_no_missing_{PIPELINE_VERSION}.xlsx",
    "train_dataset": f"train_dataset_{PIPELINE_VERSION}.xlsx",
    "test_dataset": f"test_dataset_{PIPELINE_VERSION}.xlsx",
    "model": f"buyer_prediction_pipeline_{PIPELINE_VERSION}.joblib",
    "report": f"preparation_report_{PIPELINE_VERSION}.json",
}

ALLOWED_BUYER_PROFILES = {
    "llc__commercial_services",
    "individual__industrial_ops",
    "individual__public_social",
    "individual__commercial_services",
    "llc__industrial_ops",
    "llc__primary",
    "individual__primary",
    "individual__unknown",
}