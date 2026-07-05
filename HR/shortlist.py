import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ─────────────────────────────────────────────
# AUTO DETECT PROTECTED COLUMNS FUNCTION
# ─────────────────────────────────────────────
def auto_detect_protected_cols(df, gender_col, threshold=0.3):
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    gender_encoded = df_encoded[gender_col]
    correlations = {}

    for col in df_encoded.columns:
        if col == gender_col:
            continue
        corr = abs(df_encoded[col].corr(gender_encoded))
        if pd.isna(corr):
            continue
        correlations[col] = round(corr, 4)

    correlations = dict(sorted(correlations.items(),
                               key=lambda x: x[1], reverse=True))
    flagged = [col for col, corr in correlations.items() if corr >= threshold]

    return correlations, flagged

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────
print("="*50)
print("SHORTLIST.PY — Random Forest Shortlisting")
print("="*50)

path = input("\nEnter path to unbiased CSV (output from reweight.py): ")
df = pd.read_csv(path)

# Strip whitespace
df.columns = df.columns.str.strip()
for col in df.select_dtypes('object').columns:
    df[col] = df[col].str.strip()

# ─────────────────────────────────────────────
# AUTO DETECT DATASET + USER CONFIRM
# ─────────────────────────────────────────────
if 'education.num' in df.columns:
    dataset_name      = "UCI Adult Income"
    suggested_gender  = 'sex'
    suggested_income  = 'income'
    suggested_pos_val = '>50K'
    suggested_occ     = 'occupation'

elif 'Attrition' in df.columns:
    dataset_name      = "IBM HR Analytics"
    suggested_gender  = 'Gender'
    suggested_income  = 'HighIncome'
    suggested_pos_val = 'Yes'
    suggested_occ     = 'JobRole'

else:
    dataset_name      = "Unknown"
    suggested_gender  = ''
    suggested_income  = ''
    suggested_pos_val = ''
    suggested_occ     = ''

print(f"\n✅ Dataset detected: {dataset_name}")
print("\nColumns:", df.columns.tolist())

# Always ask user to confirm or override
gender_col   = input(f"\nEnter gender column name (detected: '{suggested_gender}'): ").strip() or suggested_gender
income_col   = input(f"Enter target column name (detected: '{suggested_income}'): ").strip() or suggested_income
positive_val = input(f"Enter positive outcome value (detected: '{suggested_pos_val}'): ").strip() or suggested_pos_val
occ_col      = input(f"Enter occupation column name (detected: '{suggested_occ}'): ").strip() or suggested_occ

# ─────────────────────────────────────────────
# AUTO DETECT PROTECTED COLUMNS
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("AUTO DETECTING PROXY/PROTECTED COLUMNS")
print("="*50)

correlations, flagged = auto_detect_protected_cols(df, gender_col)

print("\nCorrelation with gender column:")
for col, corr in correlations.items():
    bar = '█' * int(corr * 20)
    flag = "⚠️  PROXY" if corr >= 0.3 else "✅ safe"
    print(f"  {col:<25} {bar:<20} {corr} {flag}")

print(f"\nAuto flagged for removal: {flagged}")
confirm = input("\nPress Enter to accept OR type custom list (comma separated): ").strip()

if confirm:
    auto_remove = [c.strip() for c in confirm.split(',')]
else:
    auto_remove = flagged
# ─────────────────────────────────────────────
# OCCUPATION FILTER
# ─────────────────────────────────────────────
target_occupation = None
is_occ_filtered = False

if occ_col and occ_col in df.columns:
    print(f"\nAvailable {occ_col}s:")
    occupations = df[occ_col].unique()
    for i, occ in enumerate(occupations):
        print(f"  [{i}] {occ}")

    target_occupation = input(f"\nEnter {occ_col} to shortlist for: ").strip()

    if target_occupation not in df[occ_col].values:
        print(f"❌ '{target_occupation}' not found. Please type exactly as shown above.")
        exit()

    df = df[df[occ_col] == target_occupation].copy()
    is_occ_filtered = True
    print(f"\n✅ Filtered to '{target_occupation}': {len(df)} candidates found")

    if len(df) < 10:
        print("⚠️ Too few candidates for this occupation")
        exit()

# ─────────────────────────────────────────────
# STEP 2 — PREPARE FEATURES
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 2: PREPARING FEATURES")
print("="*50)

# Extract fairness weights
if 'fairness_weight' in df.columns:
    sample_weights = df['fairness_weight'].values
    print("✅ Fairness weights found — reweighting active")
else:
    sample_weights = np.ones(len(df))
    print("⚠️ No fairness weights — training without reweighting")

# Build remove list
remove_cols = auto_remove + [income_col, 'fairness_weight',
                              'merit_score', 'final_score']
remove_cols = list(set([c for c in remove_cols if c in df.columns]))

print(f"\nRemoving columns: {remove_cols}")

feature_cols = [c for c in df.columns if c not in remove_cols]
print(f"Features used:    {feature_cols}")

X = df[feature_cols].copy()
y = (df[income_col] == positive_val).astype(int)

# Encode categorical columns
le_dict = {}
for col in X.columns:
    if X[col].dtype == object:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

print(f"Total samples:    {len(X)}")

# ─────────────────────────────────────────────
# STEP 3 — TRAIN RANDOM FOREST
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 3: TRAINING RANDOM FOREST MODEL")
print("="*50)

X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X, y, sample_weights, test_size=0.3, random_state=42, stratify=y
)


n_trees = 100

model = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)
model.fit(X_train, y_train, sample_weight=sw_train)

print(f"\n✅ Model trained with {n_trees} trees on {len(X_train)} samples")

y_pred = model.predict(X_test)
print("\nModel Performance:")
print(classification_report(y_test, y_pred))

print("\nFeature Importance:")
feat_imp = pd.Series(model.feature_importances_, index=feature_cols)
for feat, imp in feat_imp.sort_values(ascending=False).items():
    bar = '█' * int(imp * 100)
    print(f"  {feat:<20} {bar} {round(imp * 100, 2)}%")

# ─────────────────────────────────────────────
# STEP 4 — PREDICT AND SHORTLIST
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 4: SHORTLISTING")
print("="*50)

df['predicted_prob'] = model.predict_proba(X)[:, 1]

# ── Occupation context block ──────────────────
print("\n" + "─"*50)
if is_occ_filtered:
    print(f"📋 CANDIDATE POOL — {occ_col}: {target_occupation}")
else:
    print("📋 CANDIDATE POOL — all occupations")
print("─"*50)

total_pool = len(df)
print(f"Total candidates available:  {total_pool}")

print(f"\nGender split:")
for g in df[gender_col].unique():
    n = (df[gender_col] == g).sum()
    pct = round(n / total_pool * 100, 1)
    print(f"  {g}: {n}  ({pct}%)")

n_qualified = (df[income_col] == positive_val).sum()
print(f"\nQualified ({income_col}={positive_val}): {n_qualified} / {total_pool}")
print("─"*50)

occ_label = f" from {target_occupation}" if is_occ_filtered else ""
try:
    top_n = int(input(f"\nHow many candidates to shortlist{occ_label} (out of {total_pool})? "))
except ValueError:
    top_n = 100

if top_n > total_pool:
    print(f"⚠️ Requested {top_n} exceeds pool ({total_pool}). Using {total_pool}.")
    top_n = total_pool

shortlist = df.nlargest(top_n, 'predicted_prob').copy()

print(f"\nTop {top_n} candidates selected.")
print(shortlist[[gender_col, income_col, 'predicted_prob']].head(10).to_string(index=False))

# ─────────────────────────────────────────────
# STEP 5 — FAIRNESS AUDIT
# ─────────────────────────────────────────────
audit_scope = f"WITHIN {target_occupation}" if is_occ_filtered else "OVERALL"
print("\n" + "="*50)
print(f"STEP 5: SHORTLIST FAIRNESS AUDIT — BIAS {audit_scope}")
print("="*50)

groups = df[gender_col].unique()

print(f"\nGender selection rate ({audit_scope.lower()}) — what % of each gender got shortlisted:")
shortlist_rates = {}
for g in groups:
    total_in_group = len(df[df[gender_col] == g])
    shortlisted_in_group = len(shortlist[shortlist[gender_col] == g])
    rate = shortlisted_in_group / total_in_group
    shortlist_rates[g] = round(rate, 4)
    print(f"  {g}: {shortlisted_in_group}/{total_in_group} = {round(rate*100,1)}% shortlisted")

di = round(min(shortlist_rates.values()) / max(shortlist_rates.values()), 4)
parity = round(abs(list(shortlist_rates.values())[0] - list(shortlist_rates.values())[1]), 4)

scope_tag = f" ({target_occupation})" if is_occ_filtered else ""
print(f"\nDisparate Impact Score{scope_tag}:     {di}")
print(f"Statistical Parity Diff{scope_tag}:    {parity}")

if di >= 0.8:
    print("✅ FAIR — Selection rate is fair across genders")
elif di >= 0.6:
    print("⚠️  BORDERLINE — Slight imbalance remains")
else:
    print("❌ BIASED — Shortlist still skewed")

print("\nAverage predicted probability by gender:")
for g in groups:
    avg = shortlist[shortlist[gender_col] == g]['predicted_prob'].mean()
    print(f"  {g}: {round(avg*100, 2)}%")

# ─── DIAGNOSTIC: DATA BIAS vs MODEL BIAS ──────
print("\n" + "─"*50)
print("DIAGNOSTIC: Source of Disparity (Data vs Model)")
print("─"*50)

print("\nPool qualification rate (% of each gender meeting the qualification criteria):")
qualification_rates = {}
for g in groups:
    total_in_group = (df[gender_col] == g).sum()
    qualified_in_group = ((df[gender_col] == g) & (df[income_col] == positive_val)).sum()
    rate = qualified_in_group / total_in_group if total_in_group > 0 else 0
    qualification_rates[g] = round(rate, 4)
    print(f"  {g}: {qualified_in_group}/{total_in_group} = {round(rate*100,1)}% qualified")

print("\nEqual Opportunity (selection rate among qualified candidates only):")
eo_breakdown = {}
eo_rates = {}
for g in groups:
    total_qualified = int(((df[gender_col] == g) & (df[income_col] == positive_val)).sum())
    qualified_selected = int(((shortlist[gender_col] == g) & (shortlist[income_col] == positive_val)).sum())
    rate = qualified_selected / total_qualified if total_qualified > 0 else 0
    eo_rates[g] = round(rate, 4)
    eo_breakdown[g] = {
        'qualified': total_qualified,
        'qualified_selected': qualified_selected,
        'eo_rate': round(rate, 4)
    }
    print(f"  {g}: {qualified_selected}/{total_qualified} = {round(rate*100,1)}% of qualified shortlisted")

eo_di = round(min(eo_rates.values()) / max(eo_rates.values()), 4) if max(eo_rates.values()) > 0 else 0.0

print(f"\nEqual Opportunity DI:       {eo_di}")
print(f"Overall DI (for reference): {di}")

print("\n" + "─"*50)
if di < 0.8 and eo_di >= 0.8:
    print("📋 DIAGNOSIS: Disparity originates from the data, not the AI model")
    print("\n   When evaluating only candidates who meet the qualification criteria,")
    print("   the model selected men and women at comparable rates. The imbalance")
    print("   in the overall shortlist mirrors the imbalance in the qualified")
    print("   candidate pool itself — the historical data contains fewer qualified")
    print("   candidates from the underrepresented group. Remediation should focus")
    print("   on data sourcing and recruitment pipelines, not the model.")
elif di < 0.8 and eo_di < 0.8:
    print("📋 DIAGNOSIS: Disparity present in both data and model")
    print("\n   The qualified candidate pool is imbalanced, AND the model selects")
    print("   qualified candidates from each group at materially different rates.")
    print("   Both the data pipeline and the model require remediation.")
elif di >= 0.8 and eo_di < 0.8:
    print("📋 DIAGNOSIS: Model is introducing bias on balanced data")
    print("\n   While overall selection rates appear balanced, the model treats")
    print("   equally qualified candidates differently across groups. The model")
    print("   itself requires retraining with stronger fairness constraints.")
else:
    print("📋 DIAGNOSIS: Selection is fair on both measures")
    print("\n   The model treats candidates fairly — both in overall selection")
    print("   rates and when matched on qualification level.")
print("─"*50)

# ─────────────────────────────────────────────
# STEP 6 — FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50)
print(f"Dataset:                    {dataset_name}")
print(f"Features used:              {feature_cols}")
print(f"Protected attrs removed:    {auto_remove}")
print(f"Total candidates:           {len(df)}")
print(f"Shortlisted:                {top_n}")
print(f"DI Score:                   {di}")
print(f"Statistical Parity:         {parity}")
print(f"Verdict: {'FAIR ✅' if di >= 0.8 else 'NEEDS REVIEW ⚠️'}")

# ─────────────────────────────────────────────
# STEP 7 — EXPORT
# ─────────────────────────────────────────────
export = input("\nExport shortlist to CSV? (yes/no): ").strip().lower()
if export == 'yes':
    out_path = input("Enter output file path: ").strip()
    shortlist.to_csv(out_path, index=False)
    print(f"✅ Shortlist saved to {out_path}")

# ─────────────────────────────────────────────
# STEP 8 — AUDIT REPORT (HTML)
# ─────────────────────────────────────────────
gen_report = input("\nGenerate human-readable audit report (HTML)? (yes/no): ").strip().lower()
if gen_report == 'yes':
    from HR.audit_report import generate_audit_report

    report_path = input("Enter report output path (e.g., audit_report.html): ").strip() or 'audit_report.html'

    gender_breakdown = {}
    for g in groups:
        applied = int((df[gender_col] == g).sum())
        selected = int((shortlist[gender_col] == g).sum())
        rate = selected / applied if applied > 0 else 0
        gender_breakdown[g] = {'applied': applied, 'selected': selected, 'rate': rate}

    feature_importance = feat_imp.sort_values(ascending=False).to_dict()

    generate_audit_report(
        output_path=report_path,
        dataset_name=dataset_name,
        model_name="Random Forest Classifier",
        target_occupation=target_occupation,
        is_occ_filtered=is_occ_filtered,
        total_pool=total_pool,
        gender_col=gender_col,
        gender_breakdown=gender_breakdown,
        top_n=top_n,
        di=di,
        feature_importance=feature_importance,
        removed_cols=auto_remove,
        fairness_applied='fairness_weight' in df.columns,
        eo_di=eo_di,
        qualification_rates=qualification_rates,
        eo_breakdown=eo_breakdown,
    )
    print(f"✅ Audit report saved to {report_path}")
    print(f"   Open it in a browser — use Ctrl+P to save as PDF.")