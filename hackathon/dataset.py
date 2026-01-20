import pandas as pd
import numpy as np
 
np.random.seed(42)
 
N = 10000
 
# Helper function
def rb(p=0.5):
    return np.random.rand() < p
 
# Provinces
provinces = ["ON","BC","AB","QC","MB","SK","NS"]
 
# Residency options
residency_options = ["resident", "newcomer", "non_resident"]
residency_probs = [0.85, 0.10, 0.05]
 
# Industries for self-employment/corporations
industries = [
    "consulting", "construction", "design", "retail", "software",
    "photography", "landscaping", "marketing", "carpentry", "finance",
    "real_estate", "delivery", "manufacturing"
]
 
def determine_form(row):
    # Non-resident
    if row["residency_status"] == "non_resident":
        return "NR4"
    # Newcomer
    if row["residency_status"] == "newcomer":
        return "RC151"
 
    # Corporation
    if row["business_structure"] == "corporation":
        return "T2"
 
    # Self-employed
    if row["self_employed"] == "yes":
        return "T2125"
 
    # Students
    if row["student_status"] == "yes":
        return "T2202"
 
    # Pension
    if row["pension_income"] > 0:
        return "T4RIF"
 
    # Children (benefits)
    if row["has_children"] == "yes":
        return "RC66"
 
    # Default
    return "T1"
 
 
rows = []
 
for i in range(N):
 
    age = np.random.randint(18, 85)
    residency = np.random.choice(residency_options, p=residency_probs)
 
    # Base features
    province = np.random.choice(provinces) if residency != "non_resident" else ""
 
    marital_status = np.random.choice(["single","married","common_law"], p=[0.55,0.30,0.15])
 
    employment_income = np.random.randint(0, 150000) if residency == "resident" else 0
    num_t4_slips = np.random.randint(0, 4) if employment_income > 0 else 0
 
    self_employed_bool = rb(0.15) if residency == "resident" else False
    self_employed = "yes" if self_employed_bool else "no"
 
    business_structure = (
        "sole_prop" if self_employed_bool else
        ("corporation" if rb(0.05) else "none")
    )
 
    industry = np.random.choice(industries) if (self_employed_bool or business_structure=="corporation") else ""
 
    business_revenue = (
        np.random.randint(5000, 250000) if self_employed_bool else
        (np.random.randint(20000, 1000000) if business_structure=="corporation" else 0)
    )
 
    years_in_business = (
        np.random.randint(1, 15) if (self_employed_bool or business_structure=="corporation") else 0
    )
 
    gst_registered = "yes" if (self_employed_bool or business_structure=="corporation") and rb(0.30) else "no"
 
    # Investment
    investment_income = np.random.randint(0, 15000) if rb(0.35) else 0
    num_t5_slips = np.random.randint(0, 3) if investment_income > 0 else 0
 
    # Pension
    pension_income = np.random.randint(0, 40000) if (age >= 60 and residency=="resident") else 0
    num_t4a_slips = np.random.randint(0, 3) if pension_income > 0 else 0
 
    rrsp_withdrawal = np.random.randint(0, 20000) if rb(0.05) else 0
 
    # Education
    student_status = "yes" if (age < 30 and rb(0.30)) else "no"
    tuition_paid = np.random.randint(1000, 15000) if student_status=="yes" else 0
 
    # Children
    has_children = "yes" if (age >= 25 and rb(0.25)) else "no"
    num_dependents = np.random.randint(1, 3) if has_children=="yes" else 0
 
    # Residency details
    arrival_year = np.random.choice([2022, 2023, 2024]) if residency=="newcomer" else ""
 
    # Government benefits
    ei_received = "yes" if rb(0.10) else "no"
    social_assistance = "yes" if rb(0.05) else "no"
    disability_benefit = "yes" if rb(0.04) else "no"
 
    # Housing
    rent_paid = np.random.randint(4000, 24000) if rb(0.40) else 0
    property_tax = np.random.randint(1000, 7000) if rb(0.15) else 0
 
    # Events
    major_life_event = np.random.choice(
        ["none","moved_province","married","new_child","job_change"],
        p=[0.60,0.15,0.10,0.08,0.07]
    )
 
    # Income types count
    income_type_count = (
        (1 if employment_income>0 else 0)
        + (1 if self_employed=="yes" else 0)
        + (1 if investment_income>0 else 0)
        + (1 if pension_income>0 else 0)
        + (1 if rrsp_withdrawal>0 else 0)
    )
 
    # ----- NEW 5 FEATURES -----
    hours_worked_per_week = np.random.randint(10, 60) if employment_income > 0 else 0
    owns_home = "yes" if rb(0.45) else "no"
    moved_province_this_year = "yes" if major_life_event=="moved_province" else "no"
    has_capital_gains = "yes" if rb(0.20) else "no"
    num_business_employees = np.random.randint(0, 50) if (self_employed_bool or business_structure=="corporation") else 0
 
    row = {
        "id": i+1,
        "age": age,
        "province": province,
        "marital_status": marital_status,
        "employment_income": employment_income,
        "num_t4_slips": num_t4_slips,
        "self_employed": self_employed,
        "business_structure": business_structure,
        "industry": industry,
        "business_revenue": business_revenue,
        "years_in_business": years_in_business,
        "gst_registered": gst_registered,
        "investment_income": investment_income,
        "num_t5_slips": num_t5_slips,
        "pension_income": pension_income,
        "num_t4a_slips": num_t4a_slips,
        "rrsp_withdrawal": rrsp_withdrawal,
        "student_status": student_status,
        "tuition_paid": tuition_paid,
        "has_children": has_children,
        "num_dependents": num_dependents,
        "residency_status": residency,
        "arrival_year": arrival_year,
        "ei_received": ei_received,
        "social_assistance": social_assistance,
        "disability_benefit": disability_benefit,
        "rent_paid": rent_paid,
        "property_tax": property_tax,
        "major_life_event": major_life_event,
        "income_type_count": income_type_count,
        # new 5 features
        "hours_worked_per_week": hours_worked_per_week,
        "owns_home": owns_home,
        "moved_province_this_year": moved_province_this_year,
        "has_capital_gains": has_capital_gains,
        "num_business_employees": num_business_employees,
    }
 
    # Determine CRA form (LABEL)
    row["recommended_form"] = determine_form(row)
 
    rows.append(row)
 
# Convert to DataFrame
df = pd.DataFrame(rows)
 
# Export
df.to_csv("cra_10000.csv", index=False)
 
print("Generated cra_10000.csv with", len(df), "rows and", len(df.columns), "columns.")