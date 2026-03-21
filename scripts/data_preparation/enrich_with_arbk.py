import re
import unicodedata
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process

INPUT_FILE = "sales_with_converted_rents.xlsx"
ARBK_FILE = "arbk.xlsx"
OUTPUT_FILE = "sales_with_converted_rents_enriched_with_arbk.xlsx"

BUSINESS_THRESHOLD = 88
OWNER_THRESHOLD = 85


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_text(x):
    s = clean_text(x).lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    replacements = {
        "sh.p.k.": "shpk",
        "sh.p.k": "shpk",
        "sh.a.": "sha",
        "sh.a": "sha",
        "l.l.c.": "llc",
        "l.l.c": "llc",
        "&": " dhe ",
        "/": " ",
        ",": " ",
        ".": " ",
        ";": " ",
        '"': " ",
        "'": " ",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_multiple_names(name_value):
    raw = clean_text(name_value)
    if not raw:
        return []

    parts = [p.strip() for p in re.split(r",|/|;| dhe | and ", raw, flags=re.IGNORECASE) if p.strip()]

    candidates = [raw]
    candidates.extend(parts)
    if len(parts) > 1:
        candidates.append(" ".join(parts))

    out = []
    seen = set()
    for c in candidates:
        n = normalize_text(c)
        if n and n not in seen:
            seen.add(n)
            out.append(c)
    return out


def parse_owner_first(owner_field):
    s = clean_text(owner_field)
    if not s:
        return "", "", ""

    parts = [p.strip() for p in s.split(";")]
    owner = parts[0] if len(parts) > 0 else ""
    capital = parts[1] if len(parts) > 1 else ""
    pct = parts[2] if len(parts) > 2 else ""
    return owner, capital, pct


def parse_nace_first_five(nace_field):
    s = clean_text(nace_field)
    if not s:
        return [("", "", "") for _ in range(5)]

    parts = s.split(";")
    codes = [x.strip() for x in parts[0].split(",")] if len(parts) > 0 and parts[0] else []
    descs = [x.strip() for x in parts[1].split(",")] if len(parts) > 1 and parts[1] else []
    types = [x.strip() for x in parts[2].split(",")] if len(parts) > 2 and parts[2] else []

    result = []
    for i in range(5):
        code = codes[i] if i < len(codes) else ""
        desc = descs[i] if i < len(descs) else ""
        typ = types[i] if i < len(types) else ""
        result.append((code, desc, typ))
    return result


def prepare_arbk(df):
    df = df.copy()

    needed_cols = [
        "EmriBiznesit",
        "Pronari;Kapitali;KapitaliPerqindje",
        "KodiNace;Pershkrimi;LlojiAktivitetit",
        "nLlojiBiznesitID",
        "LlojiBiznesit",
        "Telefoni",
        "Email",
        "WebFaqja",
        "NumriPunetoreve",
        "Kapitali",
        "DataRegjistrimit",
        "DataShuarjesBiznesit",
        "StatusiARBK",
    ]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = ""

    df["__business_norm"] = df["EmriBiznesit"].map(normalize_text)

    owner_parts = df["Pronari;Kapitali;KapitaliPerqindje"].map(parse_owner_first)
    df["__owner_first"] = owner_parts.map(lambda t: t[0])
    df["__owner_first_norm"] = df["__owner_first"].map(normalize_text)

    return df


def build_exact_maps(df_arbk):
    business_map = {}
    owner_map = {}

    for idx, row in df_arbk.iterrows():
        b = row["__business_norm"]
        o = row["__owner_first_norm"]

        if b and b not in business_map:
            business_map[b] = idx
        if o and o not in owner_map:
            owner_map[o] = idx

    return business_map, owner_map


def enrich_from_row(arbk_row, match_field="", match_score=np.nan):
    owner_name, owner_capital, owner_pct = parse_owner_first(
        arbk_row.get("Pronari;Kapitali;KapitaliPerqindje", "")
    )

    nace = parse_nace_first_five(
        arbk_row.get("KodiNace;Pershkrimi;LlojiAktivitetit", "")
    )

    data = {
        "ARBK_nLlojiBiznesitID": arbk_row.get("nLlojiBiznesitID", ""),
        "ARBK_LlojiBiznesit": arbk_row.get("LlojiBiznesit", ""),
        "ARBK_Telefoni": arbk_row.get("Telefoni", ""),
        "ARBK_Email": arbk_row.get("Email", ""),
        "ARBK_WebFaqja": arbk_row.get("WebFaqja", ""),
        "ARBK_NumriPunetoreve": arbk_row.get("NumriPunetoreve", ""),
        "ARBK_Kapitali": arbk_row.get("Kapitali", ""),
        "ARBK_DataRegjistrimit": arbk_row.get("DataRegjistrimit", ""),
        "ARBK_DataShuarjesBiznesit": arbk_row.get("DataShuarjesBiznesit", ""),
        "ARBK_StatusiARBK": arbk_row.get("StatusiARBK", ""),
        "ARBK_Pronari_1": owner_name,
        "ARBK_Pronari_1_Kapitali": owner_capital,
        "ARBK_Pronari_1_KapitaliPerqindje": owner_pct,
        "ARBK_EmriBiznesit_Gjetur": arbk_row.get("EmriBiznesit", ""),
        "ARBK_MatchField": match_field,
        "ARBK_MatchScore": match_score,
    }

    for i, (code, desc, typ) in enumerate(nace, start=1):
        data[f"ARBK_Aktiviteti_{i}_KodiNace"] = code
        data[f"ARBK_Aktiviteti_{i}_Pershkrimi"] = desc
        data[f"ARBK_Aktiviteti_{i}_LlojiAktivitetit"] = typ

    return data


def empty_enrichment():
    data = {
        "ARBK_nLlojiBiznesitID": "",
        "ARBK_LlojiBiznesit": "",
        "ARBK_Telefoni": "",
        "ARBK_Email": "",
        "ARBK_WebFaqja": "",
        "ARBK_NumriPunetoreve": "",
        "ARBK_Kapitali": "",
        "ARBK_DataRegjistrimit": "",
        "ARBK_DataShuarjesBiznesit": "",
        "ARBK_StatusiARBK": "",
        "ARBK_Pronari_1": "",
        "ARBK_Pronari_1_Kapitali": "",
        "ARBK_Pronari_1_KapitaliPerqindje": "",
        "ARBK_EmriBiznesit_Gjetur": "",
        "ARBK_MatchField": "",
        "ARBK_MatchScore": np.nan,
    }
    for i in range(1, 6):
        data[f"ARBK_Aktiviteti_{i}_KodiNace"] = ""
        data[f"ARBK_Aktiviteti_{i}_Pershkrimi"] = ""
        data[f"ARBK_Aktiviteti_{i}_LlojiAktivitetit"] = ""
    return data


def main():
    print("Duke lexuar sales_with_converted_rents...")
    df_sales = pd.read_excel(INPUT_FILE, engine="openpyxl")
    print(f"sales rows: {len(df_sales)}")

    print("Duke lexuar ARBK...")
    df_arbk = pd.read_excel(ARBK_FILE, engine="openpyxl")
    print(f"arbk rows: {len(df_arbk)}")

    print("Duke përgatitur ARBK index...")
    df_arbk = prepare_arbk(df_arbk)
    business_map, owner_map = build_exact_maps(df_arbk)

    business_choices = {row["__business_norm"]: idx for idx, row in df_arbk.iterrows() if row["__business_norm"]}
    owner_choices = {row["__owner_first_norm"]: idx for idx, row in df_arbk.iterrows() if row["__owner_first_norm"]}

    enriched_rows = []

    for i, row in df_sales.iterrows():
        if i % 100 == 0:
            print(f"Procesuar {i}/{len(df_sales)}")

        buyer = clean_text(row.get("Blerësi", ""))
        candidates = split_multiple_names(buyer)

        matched_idx = None
        matched_field = ""
        matched_score = np.nan

        # 1. Exact business match
        for cand in candidates:
            norm = normalize_text(cand)
            if norm in business_map:
                matched_idx = business_map[norm]
                matched_field = "EmriBiznesit_exact"
                matched_score = 100
                break

        # 2. Exact owner match
        if matched_idx is None:
            for cand in candidates:
                norm = normalize_text(cand)
                if norm in owner_map:
                    matched_idx = owner_map[norm]
                    matched_field = "Pronari_exact"
                    matched_score = 100
                    break

        # 3. Fuzzy business match
        if matched_idx is None and candidates:
            best = None
            for cand in candidates:
                norm = normalize_text(cand)
                if not norm:
                    continue
                result = process.extractOne(
                    norm,
                    business_choices.keys(),
                    scorer=fuzz.token_sort_ratio
                )
                if result:
                    value, score, _ = result
                    if score >= BUSINESS_THRESHOLD:
                        idx_found = business_choices[value]
                        if best is None or score > best[1]:
                            best = (idx_found, score, "EmriBiznesit_fuzzy")
            if best:
                matched_idx, matched_score, matched_field = best

        # 4. Fuzzy owner match
        if matched_idx is None and candidates:
            best = None
            for cand in candidates:
                norm = normalize_text(cand)
                if not norm:
                    continue
                result = process.extractOne(
                    norm,
                    owner_choices.keys(),
                    scorer=fuzz.token_sort_ratio
                )
                if result:
                    value, score, _ = result
                    if score >= OWNER_THRESHOLD:
                        idx_found = owner_choices[value]
                        if best is None or score > best[1]:
                            best = (idx_found, score, "Pronari_fuzzy")
            if best:
                matched_idx, matched_score, matched_field = best

        if matched_idx is not None:
            arbk_row = df_arbk.loc[matched_idx]
            enriched_rows.append(enrich_from_row(arbk_row, matched_field, matched_score))
        else:
            enriched_rows.append(empty_enrichment())

    enriched_df = pd.DataFrame(enriched_rows)
    final_df = pd.concat(
        [df_sales.reset_index(drop=True), enriched_df.reset_index(drop=True)],
        axis=1
    )

    print("Duke ruajtur rezultatin...")
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        final_df.to_excel(writer, index=False, sheet_name="enriched")
        ws = writer.sheets["enriched"]
        ws.freeze_panes = "A2"

    print(f"U krijua file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()