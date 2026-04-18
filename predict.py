import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin


# =========================================================
# REQUIRED: same custom transformer used during training
# =========================================================

class PropertyFeatureEngineer(BaseEstimator, TransformerMixin):
    OBJECT_AREA_COL = "siperfaqja_e_objektit_nese_ka_objekt_te_toka_dhe_dihet_siperfaqja_m2"
    LAND_AREA_COL = "siperfaqja_e_tokes_ne_metra_katror"
    PRICE_COL = "cmimi_i_shitjes_se_asetit"
    ASSET_CAT_COL = "kategoria_e_asetit"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        required_cols = [
            self.OBJECT_AREA_COL,
            self.LAND_AREA_COL,
            self.PRICE_COL,
            self.ASSET_CAT_COL,
            "komuna_lokacioni_i_asetit_te_shitur",
            "qytet_apo_fshat_lokacioni_i_asetit_te_shitur",
            "zona_kadastrale",
            "eshte_shitur_si_ndermarrje_e_re_apo_aset_ne_likuidim",
        ]

        for col in required_cols:
            if col not in X.columns:
                X[col] = np.nan

        for col in [self.OBJECT_AREA_COL, self.LAND_AREA_COL, self.PRICE_COL]:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        object_area = X[self.OBJECT_AREA_COL].fillna(0.0)
        land_area = X[self.LAND_AREA_COL].fillna(0.0)
        price = X[self.PRICE_COL].fillna(0.0)

        total_area = object_area + land_area

        X["total_area_m2_custom"] = total_area
        X["has_object_custom"] = (object_area > 0).astype(int)
        X["has_land_custom"] = (land_area > 0).astype(int)
        X["is_land_only_custom"] = ((land_area > 0) & (object_area <= 0)).astype(int)
        X["is_object_only_custom"] = ((object_area > 0) & (land_area <= 0)).astype(int)
        X["is_mixed_asset_custom"] = ((object_area > 0) & (land_area > 0)).astype(int)

        X["object_to_land_ratio_custom"] = np.where(
            land_area > 0,
            object_area / np.maximum(land_area, 1e-6),
            0.0
        )

        X["price_per_total_m2_custom"] = np.where(
            total_area > 0,
            price / np.maximum(total_area, 1e-6),
            0.0
        )

        X["price_per_land_m2_custom"] = np.where(
            land_area > 0,
            price / np.maximum(land_area, 1e-6),
            0.0
        )

        X["price_per_object_m2_custom"] = np.where(
            object_area > 0,
            price / np.maximum(object_area, 1e-6),
            0.0
        )

        X["log_price_custom"] = np.log1p(np.maximum(price, 0))
        X["log_total_area_custom"] = np.log1p(np.maximum(total_area, 0))

        text_cols = [
            "kategoria_e_asetit",
            "komuna_lokacioni_i_asetit_te_shitur",
            "qytet_apo_fshat_lokacioni_i_asetit_te_shitur",
            "zona_kadastrale",
            "eshte_shitur_si_ndermarrje_e_re_apo_aset_ne_likuidim",
        ]

        for col in text_cols:
            X[col] = (
                X[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": np.nan, "none": np.nan, "": np.nan})
            )

        asset_cat = X["kategoria_e_asetit"].fillna("").astype(str).str.lower()

        X["asset_type_bucket_custom"] = np.select(
            [
                asset_cat.str.contains("tokë", na=False),
                asset_cat.str.contains("lokal|zyre|ndërtes|ndert", na=False),
                asset_cat.str.contains("pomp|kioska|shop", na=False),
            ],
            [
                "land",
                "object",
                "special_object",
            ],
            default="other"
        )

        return X


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = r"models\v2\property_buyer_rf_model.joblib"
ENCODER_PATH = r"models\v2\property_buyer_label_encoder.joblib"

TOP_K = 10
MIN_PROB = 0.0

pipeline = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_top_candidate_buyers(
    kategoria_e_asetit,
    object_area_m2,
    land_area_m2,
    komuna,
    qytet_apo_fshat,
    zona_kadastrale,
    sale_type,
    asking_price,
    top_k=10,
    min_prob=0.0,
    exclude_other=False
):
    input_df = pd.DataFrame([{
        "kategoria_e_asetit": kategoria_e_asetit,
        "siperfaqja_e_objektit_nese_ka_objekt_te_toka_dhe_dihet_siperfaqja_m2": object_area_m2,
        "siperfaqja_e_tokes_ne_metra_katror": land_area_m2,
        "komuna_lokacioni_i_asetit_te_shitur": komuna,
        "qytet_apo_fshat_lokacioni_i_asetit_te_shitur": qytet_apo_fshat,
        "zona_kadastrale": zona_kadastrale,
        "eshte_shitur_si_ndermarrje_e_re_apo_aset_ne_likuidim": sale_type,
        "cmimi_i_shitjes_se_asetit": asking_price,
    }])

    probabilities = pipeline.predict_proba(input_df)[0]
    sorted_indices = probabilities.argsort()[::-1]

    results = []
    other_probability = 0.0

    for idx in sorted_indices:
        prob = float(probabilities[idx])
        buyer = label_encoder.inverse_transform([idx])[0]

        if buyer == "__OTHER__":
            other_probability = prob
            if exclude_other:
                continue

        if prob < min_prob:
            continue

        results.append({
            "buyer": buyer,
            "probability": prob
        })

        if len(results) >= top_k:
            break

    return {
        "other_probability": other_probability,
        "predictions": results
    }


# =========================================================
# TEST WITH YOUR ROW
# =========================================================

if __name__ == "__main__":
    result = predict_top_candidate_buyers(
        kategoria_e_asetit="fabrikë",
        object_area_m2=7113,
        land_area_m2=34819,
        komuna="Prishtinë",
        qytet_apo_fshat="Qytet",
        zona_kadastrale="Prishtinë",
        sale_type="Ndërmarrje e Re",
        asking_price=750000,
        top_k=10,
        min_prob=0.0,
        exclude_other=False
    )

    print("\nRare/OTHER probability:", f"{result['other_probability']:.6f}")
    print("\nTop candidate buyers:")
    for i, r in enumerate(result["predictions"], start=1):
        print(f"{i:02d}. {r['buyer']} -> {r['probability']:.6f}")