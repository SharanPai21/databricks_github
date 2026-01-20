"""
STEP 1: Enhanced CRA Form Dataset Generator
Run this first to generate your training data

Output: cra_dataset_50k.csv (50,000 rows, 65+ features)
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 70)
print("CRA FORM PREDICTION - DATASET GENERATOR")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

np.random.seed(42)
N = 50000  # Generate 50,000 records

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def rb(p=0.5):
    """Random boolean with probability p"""
    return np.random.rand() < p

def weighted_choice(choices, weights):
    """Weighted random selection"""
    return np.random.choice(choices, p=np.array(weights)/sum(weights))

def lognormal_income(median, spread=0.6):
    """Generate realistic income with lognormal distribution"""
    if median == 0:
        return 0
    value = int(np.random.lognormal(np.log(median), spread))
    return max(0, value)

# ============================================================================
# CONFIGURATION
# ============================================================================

provinces = ["ON", "BC", "AB", "QC", "MB", "SK", "NS", "NB", "PE", "NL"]
residency_options = ["resident", "newcomer", "non_resident"]
residency_probs = [0.82, 0.12, 0.06]

industries = [
    "consulting", "construction", "design", "retail", "software", 
    "photography", "landscaping", "marketing", "carpentry", "finance", 
    "real_estate", "delivery", "manufacturing", "healthcare", "legal",
    "education", "hospitality", "transportation", "agriculture"
]

business_structures = ["none", "sole_prop", "corporation", "partnership"]

# ============================================================================
# FORM DETERMINATION LOGIC
# ============================================================================

def determine_forms(row):
    """Determine all required CRA forms for a taxpayer"""
    forms = []
    
    # === BASE FORMS ===
    if row["residency_status"] in ["resident", "newcomer"]:
        forms.append("T1")
    
    if row["residency_status"] == "non_resident":
        forms.append("NR4")
        if row["business_revenue"] > 0:
            forms.append("T2")
        return list(set(forms))
    
    # === NEWCOMERS ===
    if row["residency_status"] == "newcomer":
        forms.append("RC151")
        if row["arrival_year"] >= 2023:
            forms.append("T1135")
    
    # === BUSINESS ===
    if row["business_structure"] == "corporation":
        forms.append("T2")
        if row["business_revenue"] >= 30000:
            forms.append("GST34")
    
    if row["self_employed"] == "yes":
        forms.append("T2125")
        if row["gst_registered"] == "yes" or row["business_revenue"] >= 30000:
            forms.append("GST34")
        if row["home_office_percentage"] > 0:
            forms.append("T2200")
        if row["vehicle_business_use"] == "yes":
            forms.append("T2200")
    
    if row["business_structure"] == "partnership":
        forms.append("T5013")
        forms.append("T2125")
    
    # === RENTAL INCOME ===
    if row["rental_income"] > 0:
        forms.append("T776")
    
    # === INVESTMENT ===
    if row["investment_income"] > 0 or row["dividend_income"] > 0:
        forms.append("T5")
    
    if row["has_capital_gains"] == "yes":
        forms.append("SCH3")
    
    if row["capital_loss_carryforward"] > 0:
        forms.append("SCH3")
    
    if row["foreign_assets"] > 100000:
        forms.append("T1135")
    
    if row["foreign_tax_paid"] > 0:
        forms.append("T2209")
    
    if row["trust_income"] > 0:
        forms.append("T3")
    
    # === RETIREMENT ===
    if row["pension_income"] > 0:
        forms.append("T4RIF")
        if row["marital_status"] in ["married", "common_law"] and row["age"] >= 65:
            if row["spouse_income"] > 0 and row["spouse_income"] < row["pension_income"]:
                forms.append("T1032")  # Pension splitting
    
    if row["rrsp_withdrawal"] > 0:
        forms.append("T4RSP")
    
    if row["rrsp_contribution"] > 0:
        forms.append("SCH7")
    
    # === EMPLOYMENT ===
    if row["employment_income"] > 0:
        if row["work_from_home"] == "yes":
            forms.append("T2200")
        if row["union_dues"] > 0 or row["professional_fees"] > 0:
            forms.append("T1")  # Already added, but these go on T1
        if row["province"] == "QC" and row["work_from_home"] == "yes":
            forms.append("TP64.3")
    
    # === STUDENTS ===
    if row["student_status"] == "yes":
        forms.append("T2202")
        forms.append("SCH11")
        if row["student_loan_interest"] > 0:
            forms.append("SCH11")
    
    # === FAMILY & CHILDREN ===
    if row["has_children"] == "yes":
        forms.append("RC66")
        if row["num_dependents"] >= 2:
            forms.append("RC62")
    
    if row["childcare_expenses"] > 0:
        forms.append("T778")
    
    if row["alimony_paid"] > 0 or row["alimony_received"] > 0:
        forms.append("T1032-ALIMONY")
    
    # === MEDICAL & DONATIONS ===
    if row["medical_expenses"] > 3000:  # Must exceed threshold
        forms.append("SCH1-MEDICAL")
    
    if row["charitable_donations"] > 0:
        forms.append("SCH9")
    
    # === HOUSING ===
    if row["first_time_homebuyer"] == "yes" and row["rrsp_withdrawal"] > 0:
        forms.append("RC096")  # Home Buyers' Plan
    
    if row["property_sold"] == "yes":
        forms.append("T2091")
    
    # === PROVINCIAL ===
    if row["province"] == "ON":
        forms.append("ON479")
    elif row["province"] == "QC":
        forms.append("TP1")
    
    # === SPECIAL SITUATIONS ===
    if row["tfsa_excess"] == "yes":
        forms.append("RC243")
    
    if row["moved_province_this_year"] == "yes":
        forms.append("T1-M")
    
    if row["bankruptcy_filed"] == "yes":
        forms.append("T1-BANKRUPTCY")
    
    return list(set(forms))

# ============================================================================
# MAIN GENERATION LOOP
# ============================================================================

rows = []
print(f"Generating {N:,} taxpayer records...\n")

for i in range(N):
    if (i + 1) % 10000 == 0:
        print(f"  Progress: {i+1:,} / {N:,} ({(i+1)/N*100:.1f}%)")
    
    # === BASIC INFO ===
    age = np.random.randint(18, 85)
    residency = np.random.choice(residency_options, p=residency_probs)
    province = np.random.choice(provinces) if residency != "non_resident" else ""
    marital_status = weighted_choice(
        ["single", "married", "common_law", "divorced", "widowed"],
        [50, 30, 12, 6, 2]
    )
    
    # === EMPLOYMENT ===
    employment_income = 0
    num_t4_slips = 0
    work_from_home = "no"
    hours_worked_per_week = 0
    union_dues = 0
    professional_fees = 0
    
    if residency == "resident":
        if age < 65:
            employment_income = lognormal_income(62400, 0.7) if rb(0.72) else 0
        else:
            employment_income = lognormal_income(35000, 0.6) if rb(0.28) else 0
        
        if employment_income > 0:
            num_t4_slips = weighted_choice([1, 2, 3], [80, 15, 5])
            work_from_home = "yes" if rb(0.42) else "no"
            hours_worked_per_week = weighted_choice([35, 40, 45, 50], [15, 50, 25, 10])
            union_dues = np.random.randint(300, 1500) if rb(0.25) else 0
            professional_fees = np.random.randint(200, 3000) if rb(0.15) else 0
    
    # === BUSINESS ===
    self_employed_bool = False
    business_structure = "none"
    industry = ""
    business_revenue = 0
    business_expenses = 0
    years_in_business = 0
    gst_registered = "no"
    num_business_employees = 0
    home_office_percentage = 0
    vehicle_business_use = "no"
    
    if residency == "resident":
        self_employed_bool = rb(0.18) if age >= 25 else rb(0.06)
        
        if self_employed_bool:
            business_structure = weighted_choice(
                ["sole_prop", "corporation", "partnership"],
                [65, 30, 5]
            )
        elif rb(0.03):
            business_structure = "corporation"
    
    self_employed = "yes" if self_employed_bool else "no"
    
    if business_structure != "none":
        industry = np.random.choice(industries)
        
        if business_structure == "sole_prop":
            business_revenue = lognormal_income(45000, 1.2)
            business_revenue = min(business_revenue, 500000)
        elif business_structure == "corporation":
            business_revenue = lognormal_income(150000, 1.4)
            business_revenue = min(business_revenue, 5000000)
        else:
            business_revenue = lognormal_income(80000, 1.3)
            business_revenue = min(business_revenue, 1000000)
        
        business_expenses = int(business_revenue * np.random.uniform(0.35, 0.75))
        years_in_business = weighted_choice([1, 2, 3, 5, 8, 12, 20], [20, 15, 12, 15, 15, 12, 11])
        gst_registered = "yes" if (business_revenue >= 30000 and rb(0.88)) else "no"
        
        if business_structure in ["corporation", "partnership"]:
            num_business_employees = weighted_choice([0, 2, 5, 10, 25], [25, 35, 25, 10, 5])
        
        if self_employed_bool:
            home_office_percentage = weighted_choice([0, 10, 20, 30, 50], [30, 25, 25, 15, 5])
            vehicle_business_use = "yes" if rb(0.35) else "no"
    
    # === SPOUSE INFO ===
    spouse_income = 0
    if marital_status in ["married", "common_law"]:
        spouse_income = lognormal_income(55000, 0.7) if rb(0.65) else 0
    
    # === RENTAL ===
    rental_income = 0
    if residency == "resident" and age >= 30:
        rental_income = lognormal_income(18000, 0.6) if rb(0.14) else 0
    
    # === INVESTMENT ===
    investment_income = 0
    dividend_income = 0
    num_t5_slips = 0
    has_capital_gains = "no"
    capital_loss_carryforward = 0
    foreign_assets = 0
    foreign_tax_paid = 0
    trust_income = 0
    
    if residency == "resident":
        if age >= 40:
            investment_income = lognormal_income(3500, 0.8) if rb(0.48) else 0
        else:
            investment_income = lognormal_income(1200, 0.7) if rb(0.22) else 0
        
        if investment_income > 0:
            num_t5_slips = weighted_choice([1, 2, 3, 4], [60, 25, 10, 5])
            dividend_income = int(investment_income * np.random.uniform(0.2, 0.6))
            has_capital_gains = "yes" if rb(0.35) else "no"
        
        if has_capital_gains == "yes":
            capital_loss_carryforward = np.random.randint(0, 15000) if rb(0.20) else 0
        
        if age >= 35:
            foreign_assets = lognormal_income(50000, 1.2) if rb(0.12) else 0
            if foreign_assets > 0:
                foreign_tax_paid = int(foreign_assets * 0.02 * np.random.uniform(0.5, 1.5))
        
        if age >= 50:
            trust_income = np.random.randint(5000, 50000) if rb(0.07) else 0
    
    # === RETIREMENT ===
    pension_income = 0
    num_t4a_slips = 0
    rrsp_withdrawal = 0
    rrsp_contribution = 0
    rrsp_contribution_room = 0
    
    if residency == "resident":
        if age >= 60:
            pension_income = lognormal_income(28000, 0.6) if rb(0.58) else 0
            if pension_income > 0:
                num_t4a_slips = weighted_choice([1, 2], [75, 25])
        
        if age >= 55:
            rrsp_withdrawal = np.random.randint(5000, 50000) if rb(0.22) else 0
        elif age >= 25:
            rrsp_withdrawal = np.random.randint(5000, 35000) if rb(0.04) else 0
        
        if age >= 25 and employment_income > 0:
            rrsp_contribution = int(employment_income * np.random.uniform(0.02, 0.12)) if rb(0.45) else 0
            rrsp_contribution_room = int(employment_income * 0.18 * np.random.uniform(0.5, 2.0))
    
    # === STUDENTS ===
    student_status = "no"
    tuition_paid = 0
    student_loan_interest = 0
    
    if age <= 30:
        student_status = "yes" if rb(0.33) else "no"
        if student_status == "yes":
            tuition_paid = np.random.randint(3000, 18000)
            student_loan_interest = np.random.randint(0, 2000) if rb(0.60) else 0
    
    # === CHILDREN ===
    has_children = "no"
    num_dependents = 0
    dependent_ages = []
    childcare_expenses = 0
    
    if age >= 25:
        if marital_status in ["married", "common_law"]:
            has_children = "yes" if rb(0.42) else "no"
        else:
            has_children = "yes" if rb(0.18) else "no"
        
        if has_children == "yes":
            num_dependents = weighted_choice([1, 2, 3, 4], [35, 40, 20, 5])
            dependent_ages = [np.random.randint(0, 18) for _ in range(num_dependents)]
            
            # Childcare for kids under 7
            kids_under_7 = sum(1 for age in dependent_ages if age < 7)
            if kids_under_7 > 0:
                childcare_expenses = kids_under_7 * np.random.randint(5000, 15000)
    
    # === FAMILY SUPPORT ===
    alimony_paid = 0
    alimony_received = 0
    if marital_status == "divorced":
        alimony_paid = np.random.randint(6000, 24000) if rb(0.25) else 0
        alimony_received = np.random.randint(6000, 24000) if rb(0.15) else 0
    
    # === DEDUCTIONS ===
    medical_expenses = 0
    charitable_donations = 0
    
    if residency == "resident":
        medical_expenses = lognormal_income(2000, 0.8) if rb(0.35) else 0
        charitable_donations = lognormal_income(800, 0.9) if rb(0.42) else 0
    
    # === RESIDENCY ===
    arrival_year = ""
    if residency == "newcomer":
        arrival_year = weighted_choice([2022, 2023, 2024], [20, 35, 45])
    
    # === BENEFITS ===
    ei_received = "yes" if (residency == "resident" and rb(0.08)) else "no"
    social_assistance = "yes" if (residency == "resident" and rb(0.04)) else "no"
    disability_benefit = "yes" if (residency == "resident" and rb(0.05)) else "no"
    
    # === HOUSING ===
    owns_home = "no"
    property_tax = 0
    property_sold = "no"
    first_time_homebuyer = "no"
    mortgage_amount = 0
    
    if age >= 30 and residency == "resident":
        owns_home = "yes" if rb(0.52) else "no"
        if owns_home == "yes":
            property_tax = np.random.randint(2500, 12000)
            property_sold = "yes" if rb(0.04) else "no"
            mortgage_amount = lognormal_income(250000, 0.5)
        elif age <= 40:
            first_time_homebuyer = "yes" if rb(0.15) else "no"
    
    rent_paid = 0
    if owns_home == "no" and residency != "non_resident":
        rent_paid = np.random.randint(8000, 36000)
    
    # === LIFE EVENTS ===
    major_life_event = weighted_choice(
        ["none", "moved_province", "married", "new_child", "job_change", "retired"],
        [55, 11, 8, 8, 12, 6]
    )
    moved_province_this_year = "yes" if major_life_event == "moved_province" else "no"
    
    # === OTHER ===
    tfsa_excess = "yes" if (residency == "resident" and rb(0.02)) else "no"
    bankruptcy_filed = "yes" if rb(0.008) else "no"
    
    # === CALCULATED FIELDS ===
    total_income = (employment_income + business_revenue + investment_income + 
                   pension_income + rental_income + trust_income + dividend_income)
    
    net_business_income = business_revenue - business_expenses if business_revenue > 0 else 0
    
    income_type_count = (
        (1 if employment_income > 0 else 0) +
        (1 if business_revenue > 0 else 0) +
        (1 if investment_income > 0 else 0) +
        (1 if pension_income > 0 else 0) +
        (1 if rental_income > 0 else 0) +
        (1 if trust_income > 0 else 0)
    )
    
    # === BUILD ROW ===
    row = {
        "id": i + 1,
        "age": age,
        "province": province,
        "marital_status": marital_status,
        "residency_status": residency,
        "arrival_year": arrival_year,
        
        # Employment
        "employment_income": employment_income,
        "num_t4_slips": num_t4_slips,
        "work_from_home": work_from_home,
        "hours_worked_per_week": hours_worked_per_week,
        "union_dues": union_dues,
        "professional_fees": professional_fees,
        
        # Business
        "self_employed": self_employed,
        "business_structure": business_structure,
        "industry": industry,
        "business_revenue": business_revenue,
        "business_expenses": business_expenses,
        "net_business_income": net_business_income,
        "years_in_business": years_in_business,
        "gst_registered": gst_registered,
        "num_business_employees": num_business_employees,
        "home_office_percentage": home_office_percentage,
        "vehicle_business_use": vehicle_business_use,
        
        # Spouse
        "spouse_income": spouse_income,
        
        # Investment
        "investment_income": investment_income,
        "dividend_income": dividend_income,
        "num_t5_slips": num_t5_slips,
        "has_capital_gains": has_capital_gains,
        "capital_loss_carryforward": capital_loss_carryforward,
        "foreign_assets": foreign_assets,
        "foreign_tax_paid": foreign_tax_paid,
        "trust_income": trust_income,
        
        # Rental
        "rental_income": rental_income,
        
        # Retirement
        "pension_income": pension_income,
        "num_t4a_slips": num_t4a_slips,
        "rrsp_withdrawal": rrsp_withdrawal,
        "rrsp_contribution": rrsp_contribution,
        "rrsp_contribution_room": rrsp_contribution_room,
        
        # Student
        "student_status": student_status,
        "tuition_paid": tuition_paid,
        "student_loan_interest": student_loan_interest,
        
        # Children
        "has_children": has_children,
        "num_dependents": num_dependents,
        "childcare_expenses": childcare_expenses,
        
        # Family
        "alimony_paid": alimony_paid,
        "alimony_received": alimony_received,
        
        # Deductions
        "medical_expenses": medical_expenses,
        "charitable_donations": charitable_donations,
        
        # Benefits
        "ei_received": ei_received,
        "social_assistance": social_assistance,
        "disability_benefit": disability_benefit,
        
        # Housing
        "owns_home": owns_home,
        "rent_paid": rent_paid,
        "property_tax": property_tax,
        "property_sold": property_sold,
        "first_time_homebuyer": first_time_homebuyer,
        "mortgage_amount": mortgage_amount,
        
        # Other
        "major_life_event": major_life_event,
        "moved_province_this_year": moved_province_this_year,
        "tfsa_excess": tfsa_excess,
        "bankruptcy_filed": bankruptcy_filed,
        
        # Aggregates
        "total_income": total_income,
        "income_type_count": income_type_count,
    }
    
    # === DETERMINE FORMS ===
    recommended_forms = determine_forms(row)
    row["recommended_forms"] = ",".join(sorted(recommended_forms))
    row["num_forms_required"] = len(recommended_forms)
    
    rows.append(row)

# ============================================================================
# CREATE DATAFRAME
# ============================================================================

df = pd.DataFrame(rows)

print("\n" + "=" * 70)
print("DATASET STATISTICS")
print("=" * 70)

print(f"\nTotal Records: {len(df):,}")
print(f"Total Features: {len(df.columns)}")
print(f"\nFeature Breakdown:")
print(f"  Demographics:  9")
print(f"  Employment:    6")
print(f"  Business:      11")
print(f"  Investment:    8")
print(f"  Retirement:    5")
print(f"  Family:        8")
print(f"  Housing:       6")
print(f"  Other:         12")

# Form statistics
from collections import Counter
form_counts = Counter()
for forms_str in df["recommended_forms"]:
    for form in forms_str.split(","):
        form_counts[form] += 1

print(f"\n{'Form Distribution (Top 20):':^70}")
print("-" * 70)
print(f"{'Form':<15} {'Count':>10} {'Percentage':>12} {'Bar'}")
print("-" * 70)

for form, count in sorted(form_counts.items(), key=lambda x: -x[1])[:20]:
    pct = (count / len(df)) * 100
    bar = "█" * int(pct / 2)
    print(f"{form:<15} {count:>10,} {pct:>11.1f}% {bar}")

print(f"\n{'Summary Statistics':^70}")
print("-" * 70)
print(f"Average forms per taxpayer: {df['num_forms_required'].mean():.2f}")
print(f"Median forms per taxpayer:  {df['num_forms_required'].median():.0f}")
print(f"Max forms for one taxpayer: {df['num_forms_required'].max()}")
print(f"Total unique forms:         {len(form_counts)}")

# ============================================================================
# EXPORT
# ============================================================================

filename = "cra_dataset_50k.csv"
df.to_csv(filename, index=False)

print(f"\n{'SUCCESS':^70}")
print("=" * 70)
print(f"✓ Dataset saved to: {filename}")
print(f"✓ File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
print(f"✓ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
print("\nNext Step: Run 'step2_feature_engineering.py'")
print("=" * 70)