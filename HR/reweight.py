import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────
path = input("Enter path to CSV file: ")
df = pd.read_csv(path)
df.columns = df.columns.str.strip()
for col in df.select_dtypes('object').columns:
    df[col] = df[col].str.strip()

# Drop UCI specific columns if present
cols_to_drop = ['fnlwgt', 'capital.loss', 'capital.gain']
cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=cols_to_drop)

print("\nColumns:", df.columns.tolist())

gender_col = input("\nEnter gender column name (e.g. 'sex'): ")
target_col = input("Enter target/income column name (e.g. 'income'): ")

# Auto convert continuous target to binary
if df[target_col].dtype in ['int64', 'float64']:
    threshold = df[target_col].median()
    binary_col = target_col + '_binary'
    df[binary_col] = (df[target_col] > threshold).map({True: 'Yes', False: 'No'})
    print(f"\n✅ '{target_col}' is numeric — auto converted to binary")
    print(f"   Threshold (median): {threshold:,.0f}")
    print(f"   Yes (above median): {(df[binary_col]=='Yes').sum()}")
    print(f"   No  (below median): {(df[binary_col]=='No').sum()}")
    income_col = binary_col
    positive_val = 'Yes'

else:
    income_col = target_col
    positive_val = input(f"Enter positive outcome value (e.g. '>50K'): ").strip()
# ─────────────────────────────────────────────
# STEP 2 — FAIRNESS REWEIGHTING (remove gender bias)
# Formula: weight = P(gender) * P(income) / P(gender, income)
# This makes gender and income statistically independent,
# so a female and male with equal merit score equally.
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 2: FAIRNESS REWEIGHTING")
print("="*50)

n = len(df)
p_gender = df[gender_col].value_counts() / n # it counts the number of occurrences of each gender and divides by total to get the probability of each gender(n) : P(gender = male)/P(TOTAL ROWS).
p_income = df[income_col].value_counts() / n # it counts the number of occurrences of each income level and divides by total to get the probability of each income level(n)
p_joint  = df.groupby([gender_col, income_col]).size() / n # it groups the data by gender and income level, counts the occurrences of each combination, and divides by total to get the joint probability

def compute_fairness_weight(row):
    g = row[gender_col] # eg first row male
    y = row[income_col] # eg first row '>50K'
    joint = p_joint.get((g, y), 1e-9) # get the joint probability for this combination
    return (p_gender[g] * p_income[y]) / joint # p(male) * p(>50K) / p(male, >50K) . 

df['fairness_weight'] = df.apply(compute_fairness_weight, axis=1) # creates new column 'fairness_weight' and by function saves new weight for each row based on the gender and income level of that row. This weight will be used in the RF model to ensure fair learning.
print("\nWeight distribution:")
print(df['fairness_weight'].describe())
print("\nWeights by gender + income combination:")
print(df.groupby([gender_col, income_col])['fairness_weight'].mean())
# Show weight distribution per gender group
print("\nAverage fairness weight by gender:")
for g, grp in df.groupby(gender_col):
    print(f"  {g}: {round(grp['fairness_weight'].mean(), 4)}")

print("\n✅ Reweighting applied — gender bias neutralised")
# ─────────────────────────────────────────────
# CHECK IF REWEIGHTING IS ACTUALLY NEEDED
# ─────────────────────────────────────────────
group_rates = {}
for g, grp in df.groupby(gender_col):
    rate = (grp[income_col] == positive_val).mean()
    group_rates[g] = round(rate, 4)

rates = list(group_rates.values())
di_check = round(min(rates) / max(rates), 4) if max(rates) > 0 else 0

if di_check >= 0.8:
    print(f"\n⚠️  WARNING: DI Score is already {di_check}")
    print("   Dataset appears fair — reweighting may not be needed")
    proceed = input("   Continue with reweighting anyway? (yes/no): ").strip().lower()
    if proceed != 'yes':
     print("✅ Skipping reweighting — data is already fair")
     print("   Adding neutral fairness weights (all = 1.0)")
    
    # Add neutral weights so shortlist.py doesn't break
    df['fairness_weight'] = 1.0
    
    # Export anyway
    drop_cols = [c for c in df.columns if c.startswith('_norm_')]
    df_out = df.drop(columns=drop_cols)
    out_path = input("\nEnter output path (e.g. fair_data.csv): ").strip()
    df_out.to_csv(out_path, index=False)
    print(f"✅ Dataset saved to {out_path}")
    print("   fairness_weight = 1.0 (no reweighting applied)")
    exit()
else:
    print(f"\n✅ Reweighting needed — DI Score is {di_check}")
# ─────────────────────────────────────────────
# STEP 3 — MERIT SCORING WEIGHTS (user-defined)
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 3: MERIT SCORING WEIGHTS")
print("="*50)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['fairness_weight']]
print("\nNumeric columns available for scoring:")
for i, col in enumerate(numeric_cols):
    print(f"  [{i}] {col}")

print("\nEnter weight (0-100) for each feature. Enter 0 to skip.")
scoring_weights = {}
for col in numeric_cols:
    try:
        w = float(input(f"  Weight for '{col}': "))
    except ValueError:
        w = 0
    if w > 0:
        scoring_weights[col] = w

if not scoring_weights:
    print("No features selected — using default: education.num=40, hours.per.week=35, age=25")
    scoring_weights = {'education.num': 40, 'hours.per.week': 35, 'age': 25}

# Normalise weights so they sum to 1
total_weight = sum(scoring_weights.values())
scoring_weights = {k: v / total_weight for k, v in scoring_weights.items()}

print("\nFinal normalised weights:")
for feat, w in scoring_weights.items():
    print(f"  {feat}: {round(w * 100, 1)}%")

# ─────────────────────────────────────────────
# STEP 4 — COMPUTE SCORES
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 4: COMPUTING SCORES")
print("="*50)

# Min-max normalise each scoring feature
for feat in scoring_weights:
    min_val = df[feat].min()
    max_val = df[feat].max()
    df[f'_norm_{feat}'] = (df[feat] - min_val) / (max_val - min_val + 1e-9)

# Merit score = weighted sum of normalised features
df['merit_score'] = sum(
    df[f'_norm_{feat}'] * w for feat, w in scoring_weights.items()
)

# Final score = merit score adjusted by fairness weight
# This upweights candidates from underrepresented groups
# who have equal merit, giving them a fair shot
df['final_score'] = df['merit_score'] * df['fairness_weight']

print("Scores computed.")
print(f"  Merit score range:  {df['merit_score'].min():.4f} – {df['merit_score'].max():.4f}")
print(f"  Final score range:  {df['final_score'].min():.4f} – {df['final_score'].max():.4f}")

# ─────────────────────────────────────────────
# STEP 5 — EXPORT UNBIASED DATASET
# Drops internal normalisation columns, keeps fairness_weight
# so shortlist.py can use it as sample_weight in the RF model
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 5: EXPORT UNBIASED CSV")
print("="*50)

drop_cols = [c for c in df.columns if c.startswith('_norm_')]
df_unbiased = df.drop(columns=drop_cols)

out_path = input("\nEnter output path for unbiased CSV (e.g. unbiased_data.csv): ").strip()
df_unbiased.to_csv(out_path, index=False)

print(f"\n✅ Unbiased dataset saved to: {out_path}")
print(f"   Rows: {len(df_unbiased)} | Columns: {df_unbiased.columns.tolist()}")
print("\nNow run shortlist.py and enter this file path when prompted.")