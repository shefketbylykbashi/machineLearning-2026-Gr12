import re
import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================
SALES_FILE = "sales.xlsx"
RENT_FILE = "rent.xlsx"

OUTPUT_CONVERTED_ONLY = "converted_rents_to_sales.xlsx"
OUTPUT_COMBINED = "sales_with_converted_rents.xlsx"

ROI_YEARS = {
    "lokal": 15,
    "kioska": 10,
    "kiosk": 10,
    "depo": 15,
    "depo dhe tokë": 15,
    "depo dhe toke": 15,
    "fabrikë": 20,
    "fabrike": 20,
    "tokë bujqësore": 25,
    "toke bujqesore": 25,
    "tokë komerciale": 20,
    "toke komerciale": 20,
    "ndërtesë administrative dhe tokë": 15,
    "ndertese administrative dhe toke": 15,
    "ndërtesë administrative": 15,
    "ndertese administrative": 15,
}

DEFAULT_ROI_YEARS = 15

SALES_COLUMNS = [
    "Nr.",
    "Rajoni AKP",
    "PAK ID",
    "Ndërmarrja Shoqërore",
    "Emri i Ndërmarrjes së Re apo Asetit në Likuidim",
    "Kategoria e asetit",
    "Mënyra e shitjes",
    "[m2]",
    "Sipërfaqja e tokës në metra katror",
    "Komuna (lokacioni i asetit të shitur)",
    "Qytet apo Fshat (lokacioni i asetit të shitur)",
    "ZONA KADASTRALE",
    "Është shitur si: Ndërmarrje e Re apo Aset në likuidim",
    "Cmimi i shitjes së asetit",
    "Blerësi",
    "Data e kontratës",
]

RENT_COLUMNS = [
    "Nr.",
    "ZR",
    "Ndermarrja Shoqerore",
    "Prona ne Qira",
    "Adresa e prones ne Qira",
    "Komuna ku gjendet prona",
    "Qiramarresi",
    "Kategoria e e Asetit",
    "Siperfaqja",
    "Vlera Qirase",
    "Data e lidhjes se kontrates se pare",
    "Paeriudha e Faturimit",
]


# =========================================================
# HELPERS
# =========================================================
def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).replace("\n", " ").replace("\r", " ").strip()


def clean_spaces(x):
    return re.sub(r"\s+", " ", norm_text(x)).strip()


def is_blank_row(row):
    return all(clean_spaces(v) == "" for v in row)


def parse_number(x):
    """
    Converts values like:
    '1,350.00' -> 1350.00
    '50' -> 50.00
    '1.897,90' is also handled if it appears in European style
    """
    s = clean_spaces(x)
    if s == "":
        return np.nan

    s = s.replace("€", "").replace("EUR", "").strip()

    if "," in s and "." in s:
        if s.rfind(".") > s.rfind(","):
            s = s.replace(",", "")
        else:
            s = s.replace(".", "").replace(",", ".")
    else:
        if "," in s and "." not in s:
            parts = s.split(",")
            if len(parts[-1]) in (1, 2):
                s = s.replace(",", ".")
            else:
                s = s.replace(",", "")

    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_date(x):
    s = clean_spaces(x)
    if s == "":
        return pd.NaT
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def contains_all(text, words):
    t = clean_spaces(text).lower()
    return all(w.lower() in t for w in words)


def detect_qytet_fshat(address, municipality):
    text = f"{clean_spaces(address)} {clean_spaces(municipality)}".lower()
    if "fsh" in text or "fsh." in text or "fshat" in text:
        return "Fshat"
    return "Qytet"


def get_roi_years(category):
    cat = clean_spaces(category).lower()

    for key, years in ROI_YEARS.items():
        if key.lower() in cat:
            return years

    return DEFAULT_ROI_YEARS


def annualize_rent(value, billing_period):
    period = clean_spaces(billing_period).lower()
    val = parse_number(value)

    if pd.isna(val):
        return np.nan

    if "mujor" in period:
        return val * 12
    if "vjetor" in period:
        return val

    return val


def is_land_category(category):
    cat = clean_spaces(category).lower()
    return "tok" in cat


def is_object_category(category):
    cat = clean_spaces(category).lower()
    object_keywords = ["lokal", "depo", "fabrik", "kiosk", "ndërtes", "ndertese"]
    return any(k in cat for k in object_keywords)


def format_money(x):
    if pd.isna(x):
        return np.nan
    return round(float(x), 2)


# =========================================================
# RENT PARSER
# Handles repeated headers and continuation rows
# =========================================================
def load_rent_table(path):
    raw = pd.read_excel(path, header=None, dtype=str)
    raw = raw.replace({np.nan: ""})

    records = []
    current = None

    for _, row in raw.iterrows():
        vals = [clean_spaces(v) for v in row.tolist()]

        if is_blank_row(vals):
            continue

        joined = " | ".join(vals).lower()

        if (
            ("nr." in joined or re.search(r"\bnr\b", joined))
            and "qiramarresi" in joined
        ):
            continue


        first = vals[0]
        if re.fullmatch(r"\d+", first):

            if current is not None:
                records.append(current)

            current = {col: "" for col in RENT_COLUMNS}

            for i, col in enumerate(RENT_COLUMNS):
                if i < len(vals):
                    current[col] = vals[i]
        else:
            if current is None:
                continue

            for i, col in enumerate(RENT_COLUMNS):
                if i < len(vals):
                    extra = vals[i]
                    if extra != "":
                        if current[col] == "":
                            current[col] = extra
                        else:
                            current[col] = f"{current[col]} {extra}".strip()

    if current is not None:
        records.append(current)

    df = pd.DataFrame(records)

    for c in df.columns:
        df[c] = df[c].apply(clean_spaces)

    df = df[
        ~df["Nr."].str.lower().isin(["nr", "nr.", ""])
    ].copy()

    df["Nr."] = pd.to_numeric(df["Nr."], errors="coerce")
    df["Siperfaqja"] = df["Siperfaqja"].apply(parse_number)
    df["Vlera Qirase"] = df["Vlera Qirase"].apply(parse_number)
    df["Data e lidhjes se kontrates se pare"] = df[
        "Data e lidhjes se kontrates se pare"
    ].apply(parse_date)

    return df


# =========================================================
# SALES LOADER
# =========================================================
def load_sales_table(path):
    raw = pd.read_excel(path, header=None, dtype=str)
    raw = raw.replace({np.nan: ""})

    records = []
    current = None

    for _, row in raw.iterrows():
        vals = [clean_spaces(v) for v in row.tolist()]

        if is_blank_row(vals):
            continue

        joined = " | ".join(vals).lower()

        if ("pak id" in joined and "bler" in joined) or ("cmimi i shitjes" in joined):
            continue

        first = vals[0]
        if re.fullmatch(r"\d+", first):
            if current is not None:
                records.append(current)

            current = {col: "" for col in SALES_COLUMNS}
            for i, col in enumerate(SALES_COLUMNS):
                if i < len(vals):
                    current[col] = vals[i]
        else:
            if current is None:
                continue

            for i, col in enumerate(SALES_COLUMNS):
                if i < len(vals):
                    extra = vals[i]
                    if extra != "":
                        if current[col] == "":
                            current[col] = extra
                        else:
                            current[col] = f"{current[col]} {extra}".strip()

    if current is not None:
        records.append(current)

    df = pd.DataFrame(records)

    for c in df.columns:
        df[c] = df[c].apply(clean_spaces)

    df = df[~df["Nr."].str.lower().isin(["nr", "nr.", ""])].copy()

    df["Nr."] = pd.to_numeric(df["Nr."], errors="coerce")
    df["[m2]"] = df["[m2]"].apply(parse_number)
    df["Sipërfaqja e tokës në metra katror"] = df[
        "Sipërfaqja e tokës në metra katror"
    ].apply(parse_number)
    df["Cmimi i shitjes së asetit"] = df["Cmimi i shitjes së asetit"].apply(parse_number)
    df["Data e kontratës"] = df["Data e kontratës"].apply(parse_date)

    return df


# =========================================================
# CONVERSION: RENT -> SALES
# =========================================================
def convert_rent_to_sales(rent_df, start_nr=1):
    converted_rows = []

    for idx, r in rent_df.iterrows():
        category = clean_spaces(r["Kategoria e e Asetit"])
        roi_years = get_roi_years(category)

        annual_rent = annualize_rent(r["Vlera Qirase"], r["Paeriudha e Faturimit"])
        sale_value = format_money(annual_rent * roi_years)

        surface = parse_number(r["Siperfaqja"])

        object_m2 = np.nan
        land_m2 = np.nan

        if is_land_category(category) and not is_object_category(category):
            land_m2 = surface
        elif is_object_category(category) and not is_land_category(category):
            object_m2 = surface
        else:

            object_m2 = surface

        address = clean_spaces(r["Adresa e prones ne Qira"])
        municipality = clean_spaces(r["Komuna ku gjendet prona"])
        city_or_village = detect_qytet_fshat(address, municipality)

        row = {
            "Nr.": start_nr + len(converted_rows),
            "Rajoni AKP": clean_spaces(r["ZR"]),
            "PAK ID": f"RENT_{int(r['Nr.'])}" if not pd.isna(r["Nr."]) else f"RENT_{idx+1}",
            "Ndërmarrja Shoqërore": clean_spaces(r["Ndermarrja Shoqerore"]),
            "Emri i Ndërmarrjes së Re apo Asetit në Likuidim": clean_spaces(r["Prona ne Qira"]),
            "Kategoria e asetit": category,
            "Mënyra e shitjes": f"Konvertuar nga qira me ROI {roi_years} vjet",
            "[m2]": object_m2,
            "Sipërfaqja e tokës në metra katror": land_m2,
            "Komuna (lokacioni i asetit të shitur)": municipality,
            "Qytet apo Fshat (lokacioni i asetit të shitur)": city_or_village,
            "ZONA KADASTRALE": address,
            "Është shitur si: Ndërmarrje e Re apo Aset në likuidim": "Aset në likuidim (i konvertuar nga qira)",
            "Cmimi i shitjes së asetit": sale_value,
            "Blerësi": clean_spaces(r["Qiramarresi"]),
            "Data e kontratës": r["Data e lidhjes se kontrates se pare"],
        }

        converted_rows.append(row)

    converted_df = pd.DataFrame(converted_rows, columns=SALES_COLUMNS)
    return converted_df


# =========================================================
# MAIN
# =========================================================
def main():
    sales_path = Path(SALES_FILE)
    rent_path = Path(RENT_FILE)

    if not sales_path.exists():
        raise FileNotFoundError(f"Missing file: {SALES_FILE}")

    if not rent_path.exists():
        raise FileNotFoundError(f"Missing file: {RENT_FILE}")

    print("Loading sales...")
    sales_df = load_sales_table(sales_path)

    print("Loading rent...")
    rent_df = load_rent_table(rent_path)

    for col in SALES_COLUMNS:
        if col not in sales_df.columns:
            sales_df[col] = np.nan

    sales_df = sales_df[SALES_COLUMNS].copy()

    max_sales_nr = pd.to_numeric(sales_df["Nr."], errors="coerce").max()
    if pd.isna(max_sales_nr):
        max_sales_nr = 0

    print("Converting rent records to sales records...")
    converted_df = convert_rent_to_sales(rent_df, start_nr=int(max_sales_nr) + 1)

    combined_df = pd.concat([sales_df, converted_df], ignore_index=True)

    combined_df["Nr."] = range(1, len(combined_df) + 1)

    converted_df.to_excel(OUTPUT_CONVERTED_ONLY, index=False)
    combined_df.to_excel(OUTPUT_COMBINED, index=False)

    print(f"Done.")
    print(f"Converted rent-only file: {OUTPUT_CONVERTED_ONLY}")
    print(f"Combined file: {OUTPUT_COMBINED}")
    print(f"Original sales rows: {len(sales_df)}")
    print(f"Converted rent rows: {len(converted_df)}")
    print(f"Final combined rows: {len(combined_df)}")


if __name__ == "__main__":
    main()