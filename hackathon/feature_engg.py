"""
STEP 2: Feature Engineering
Creates derived features that improve model accuracy

Input:  cra_dataset_50k.csv
Output: cra_dataset_engineered.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 70)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading dataset...")
df = pd.read_csv("cra_dataset_50k.csv")
print(f"✓ Loaded {len(df):,} records with {len(df.columns)} features\n")

# ============================================================================
# DERIVED FEATURES (Optimized for CRA form prediction)
# ============================================================================

print("Creating derived features...\n")

# -----------------------------------------
# 1. Income ratios
# -----------------------------------------

print("  [1/5] Income ratios...")

df["total_income"] = df["total_income"].replace(0, 1)

df["business_income_ratio"] = df["business_revenue"] / df["total_income"]
df["employment_income_ratio"] = df["employment_income"] / df["total_income"]
df["investment_income_ratio"] = (
    df["investment_income"] + df["dividend_income"]
) / df["total_income"]
df["pension_income_ratio"] = df["pension_income"] / df["total_income"]

# -----------------------------------------
# 2. Age groups
# -----------------------------------------

print("  [2/5] Age groups...")

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 35, 50, 65, 100],
    labels=[
        "young",
        "early_career",
        "mid_career",
        "pre_retirement",
        "retirement",
    ],
)

# -----------------------------------------
# 3. Complexity indicators
# -----------------------------------------

print("  [3/5] Complexity scores...")

df["business_complexity_score"] = (
    (df["business_revenue"] > 0).astype(int)
    + (df["gst_registered"] == "yes").astype(int)
    + (df["num_business_employees"] > 0).astype(int)
)

df["tax_complexity"] = (
    df["income_type_count"]
    + (df["has_capital_gains"] == "yes").astype(int)
    + (df["foreign_assets"] > 0).astype(int)
    + (df["rental_income"] > 0).astype(int)
)

df["family_complexity"] = (
    (df["marital_status"].isin(["married", "common_law"])).astype(int)
    + df["num_dependents"]
)

# -----------------------------------------
# 4. Core binary flags for CRA form logic
# -----------------------------------------

print("  [4/5] Core binary indicators...")

df["has_any_investment"] = (
    (df["investment_income"] > 0)
    | (df["dividend_income"] > 0)
    | (df["has_capital_gains"] == "yes")
).astype(int)

df["has_property_income"] = (
    (df["rental_income"] > 0) | (df["property_tax"] > 0)
).astype(int)

df["has_international_ties"] = (
    (df["foreign_assets"] > 0)
    | (df["foreign_tax_paid"] > 0)
    | (df["residency_status"] == "newcomer")
).astype(int)

df["has_employment_expenses"] = (
    (df["union_dues"] > 0)
    | (df["professional_fees"] > 0)
    | (df["work_from_home"] == "yes")
).astype(int)

# -----------------------------------------
# 5. Interaction features (only impactful ones)
# -----------------------------------------

print("  [5/5] Interaction features...")

df["newcomer_entrepreneur"] = (
    (df["residency_status"] == "newcomer")
    & (df["business_revenue"] > 0)
).astype(int)

df["property_investor"] = (
    (df["rental_income"] > 0) & (df["owns_home"] == "yes")
).astype(int)

df["high_income_self_employed"] = (
    (df["self_employed"] == "yes") & (df["total_income"] > 165430)
).astype(int)

print("\n✓ Created optimized feature set (reduced and cleaner)")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 70)

original_features = 65  # from dataset generator
new_features = len(df.columns) - original_features - 2  # exclude form fields

print(f"\nOriginal features:  {original_features}")
print(f"Derived features:   {new_features}")
print(f"Total features:     {len(df.columns)}")

print("\nFeature Categories Added:")
print("  • Income ratios:               4")
print("  • Age groups:                  1")
print("  • Complexity indicators:       3")
print("  • Binary flags:                4")
print("  • Interaction features:        3")
print("  ----------------------------------")
print("  • Total derived features:      15")

# ============================================================================
# HANDLE MISSING VALUES SAFELY
# ============================================================================

print("\nChecking for missing values...")

nan_count = df.isnull().sum().sum()

if nan_count > 0:
    print(f"⚠️  Warning: Found {nan_count} NaN values")
    print("   Filling with type-appropriate defaults...")

    # numeric fields
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # categorical fields
    cat_cols = df.select_dtypes(include=["category"]).columns
    for col in cat_cols:
        df[col] = df[col].cat.add_categories(["unknown"]).fillna("unknown")

    # object/string fields
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].fillna("unknown")

else:
    print("✓ No missing values detected")

# ============================================================================
# EXPORT
# ============================================================================

filename = "cra_dataset_engineered.csv"
df.to_csv(filename, index=False)

print(f"\n{'SUCCESS':^70}")
print("=" * 70)
print(f"✓ Engineered dataset saved to: {filename}")
print(f"✓ File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
print(f"✓ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
print("\nNext Step: Run 'step3_train_model.py'")
print("=" * 70)
