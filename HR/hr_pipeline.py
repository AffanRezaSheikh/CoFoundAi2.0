"""
Non-interactive HR shortlisting pipeline.

Wraps the same logic as reweight.py + catshortlist.py (fairness reweighting +
CatBoost shortlist + bias audit) into a single callable function so a web
backend can invoke it without the original input() prompts.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATASET_PRESETS = {
    "adult.csv": {
        "label": "UCI Adult Income",
        "gender_col": "sex",
        "target_col": "income",
        "positive_val": ">50K",
        "occupation_col": "occupation",
    },
    "attrition.csv": {
        "label": "IBM HR Analytics",
        "gender_col": "Gender",
        "target_col": "Attrition",
        "positive_val": "Yes",
        "occupation_col": "JobRole",
    },
}


def _auto_detect_protected_cols(df, gender_col, threshold=0.3):
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    gender_encoded = df_encoded[gender_col]
    flagged = []
    for col in df_encoded.columns:
        if col == gender_col:
            continue
        corr = abs(df_encoded[col].corr(gender_encoded))
        if pd.isna(corr):
            continue
        if corr >= threshold:
            flagged.append(col)
    return flagged


def _apply_fairness_weights(df, gender_col, target_col, positive_val):
    n = len(df)
    p_gender = df[gender_col].value_counts() / n
    p_target = df[target_col].value_counts() / n
    p_joint = df.groupby([gender_col, target_col]).size() / n

    def weight(row):
        g, y = row[gender_col], row[target_col]
        joint = p_joint.get((g, y), 1e-9)
        return (p_gender[g] * p_target[y]) / joint

    # Damping exponent (<1) softens the Kamiran-Calders correction so it moves
    # toward demographic parity without overshooting into reverse bias on
    # imbalanced data (e.g. Adult's 2:1 male/female split).
    df["fairness_weight"] = df.apply(weight, axis=1) ** 0.5

    group_rates = {}
    for g, grp in df.groupby(gender_col):
        group_rates[g] = (grp[target_col] == positive_val).mean()
    rates = list(group_rates.values())
    di_before = round(min(rates) / max(rates), 4) if max(rates) > 0 else 0.0
    return df, di_before


def run_shortlist(csv_path, gender_col, target_col, positive_val,
                  top_n=100, occupation_col=None, target_occupation=None,
                  sample_rows=6000):
    """
    Run the full bias-aware shortlist pipeline and return a JSON-safe dict.

    Returns: dataset stats, fairness metrics (before/after), gender breakdown,
    feature importance, and the shortlisted candidates.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    # keep runtime reasonable for a web request
    if sample_rows and len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=42).reset_index(drop=True)

    # drop UCI noise columns if present
    for c in ["fnlwgt", "capital.loss", "capital.gain"]:
        if c in df.columns:
            df = df.drop(columns=c)

    # binarize numeric target
    if df[target_col].dtype in ["int64", "float64"]:
        threshold = df[target_col].median()
        df[target_col] = (df[target_col] > threshold).map({True: "Yes", False: "No"})
        positive_val = "Yes"

    df, di_before = _apply_fairness_weights(df, gender_col, target_col, positive_val)

    # optional occupation filter
    is_occ_filtered = False
    if occupation_col and target_occupation and occupation_col in df.columns:
        df = df[df[occupation_col] == target_occupation].copy()
        is_occ_filtered = True
        if len(df) < 10:
            return {"error": f"Too few candidates for '{target_occupation}' ({len(df)})."}

    flagged = _auto_detect_protected_cols(df, gender_col)
    remove_cols = list(set(
        flagged + [target_col, "fairness_weight", "merit_score", "final_score"]
    ))
    remove_cols = [c for c in remove_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in remove_cols]

    X = df[feature_cols].copy()
    y = (df[target_col] == positive_val).astype(int)

    if y.nunique() < 2:
        return {"error": "Target has only one class after filtering — cannot train."}

    sample_weights = df["fairness_weight"].values

    cat_idx = []
    for i, col in enumerate(X.columns):
        if X[col].dtype == object:
            X[col] = X[col].astype(str).fillna("missing")
            cat_idx.append(i)

    X_train, X_test, y_train, y_test, sw_train, _ = train_test_split(
        X, y, sample_weights, test_size=0.3, random_state=42, stratify=y
    )

    model = CatBoostClassifier(
        iterations=200, learning_rate=0.05, depth=6,
        random_state=42, eval_metric="AUC", verbose=0,
    )
    model.fit(X_train, y_train, sample_weight=sw_train,
              cat_features=cat_idx, eval_set=(X_test, y_test),
              early_stopping_rounds=30)

    df["predicted_prob"] = model.predict_proba(X)[:, 1]

    total_pool = len(df)
    top_n = min(top_n, total_pool)
    shortlist = df.nlargest(top_n, "predicted_prob").copy()

    # fairness audit on the shortlist
    groups = list(df[gender_col].unique())
    shortlist_rates, gender_breakdown = {}, {}
    for g in groups:
        total_in_group = int((df[gender_col] == g).sum())
        selected = int((shortlist[gender_col] == g).sum())
        rate = selected / total_in_group if total_in_group else 0
        shortlist_rates[g] = rate
        gender_breakdown[g] = {
            "pool": total_in_group,
            "selected": selected,
            "rate": round(rate * 100, 1),
        }

    rates = list(shortlist_rates.values())
    di_after = round(min(rates) / max(rates), 4) if max(rates) > 0 else 0.0

    # equal-opportunity (selection rate among qualified only)
    eo_rates = {}
    for g in groups:
        qualified = int(((df[gender_col] == g) & (df[target_col] == positive_val)).sum())
        qsel = int(((shortlist[gender_col] == g) & (shortlist[target_col] == positive_val)).sum())
        eo_rates[g] = qsel / qualified if qualified else 0
    eo_vals = list(eo_rates.values())
    eo_di = round(min(eo_vals) / max(eo_vals), 4) if max(eo_vals) and max(eo_vals) > 0 else 0.0

    if di_after < 0.8 and eo_di >= 0.8:
        diagnosis = "Disparity originates from the data (qualified pool imbalance), not the model."
    elif di_after < 0.8 and eo_di < 0.8:
        diagnosis = "Disparity present in both the data and the model — both need remediation."
    elif di_after >= 0.8 and eo_di < 0.8:
        diagnosis = "Model introduces bias on otherwise balanced data — retrain with fairness constraints."
    else:
        diagnosis = "Selection is fair on both overall and equal-opportunity measures."

    feat_imp = pd.Series(model.get_feature_importance(), index=feature_cols)
    feat_imp = (feat_imp / feat_imp.sum() * 100).round(2)
    feature_importance = [
        {"feature": f, "importance": float(v)}
        for f, v in feat_imp.sort_values(ascending=False).items()
    ]

    # build candidate rows for the UI (drop internal cols)
    display_cols = [c for c in shortlist.columns
                    if c not in ["fairness_weight", "merit_score", "final_score"]]
    candidates = []
    for _, row in shortlist[display_cols].head(top_n).iterrows():
        rec = {}
        for c in display_cols:
            val = row[c]
            if c == "predicted_prob":
                rec["match_score"] = round(float(val) * 100, 1)
            elif isinstance(val, (np.integer,)):
                rec[c] = int(val)
            elif isinstance(val, (np.floating,)):
                rec[c] = round(float(val), 2)
            else:
                rec[c] = str(val)
        candidates.append(rec)

    verdict = "FAIR" if di_after >= 0.8 else ("BORDERLINE" if di_after >= 0.6 else "BIASED")

    return {
        "dataset_rows": total_pool,
        "features_used": feature_cols,
        "protected_removed": flagged,
        "target_occupation": target_occupation if is_occ_filtered else None,
        "shortlisted": top_n,
        "metrics": {
            "di_before_reweight": di_before,
            "di_after": di_after,
            "equal_opportunity_di": eo_di,
            "verdict": verdict,
            "diagnosis": diagnosis,
        },
        "gender_breakdown": gender_breakdown,
        "feature_importance": feature_importance,
        "candidates": candidates,
    }


def list_occupations(csv_path, occupation_col):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if occupation_col not in df.columns:
        return []
    return sorted(str(x).strip() for x in df[occupation_col].dropna().unique())
