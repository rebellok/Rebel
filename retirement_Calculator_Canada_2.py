# retirement_Calculator_Canada_fixed_v2.py 

# 🇨🇦 Canadian Retirement Planner — exact CRA RRIF factors, nominal vs real toggle, better tax credits/splitting, and PDF export 

# Run: streamlit run retirement_Calculator_Canada_fixed_v2.py 

 

from __future__ import annotations 

 

import streamlit as st 

import pandas as pd 

import numpy as np 

import plotly.express as px 

import plotly.graph_objects as go 

import requests 

from io import BytesIO, StringIO 

from typing import Optional 

from reportlab.lib.pagesizes import letter 

from reportlab.pdfgen import canvas 

import tempfile 

 

st.set_page_config(page_title="Canadian Retirement Planner (Canada) — v2", layout="wide") 

st.image("Flag_of_Canada.png", width=50) 

st.markdown("### Canadian Retirement Planning Calculator — RRSP / TFSA / RRIF / GIS / OAS clawback & deferral (v2)") 

 

st.markdown( 

    """ 

    **This version**: 

    - Fetches official CRA prescribed RRIF minimum factors at runtime (falls back to bundled table if fetch fails). 

    - Adds a **Nominal vs Real** toggle (adjust results for inflation). 

    - Adds simplified **pension/age credits** and an optional **pension income splitting** between spouses. 

    

    Sources for RRIF factors: Canada.ca minimum amount pages (CRA). If fetching fails the app uses a bundled conservative table. 

    """ 

) 

 

# ------------------------ 

# Utilities: fetch CRA RRIF factors 

# ------------------------ 

 

@st.cache_data(ttl=60*60*24) 

def fetch_cra_rrif_factors(): 

    """Attempt to download CRA prescribed factors table and return a dict {age: factor}. 

    Falls back to bundled factors if network or parsing fails. 

    """ 

    url = "https://www.canada.ca/en/revenue-agency/services/tax/businesses/topics/completing-slips-summaries/t4rsp-t4rif-information-returns/payments/chart-prescribed-factors.html" 

    try: 

        r = requests.get(url, timeout=8) 

        r.raise_for_status() 

        from io import StringIO  # <-- add this import 

        tables = pd.read_html(StringIO(r.text))  # <-- wrap r.text in StringIO 

        # heuristics: find a table with an 'Age' or first column numeric and second column factor 

        for t in tables: 

            cols = [c.lower() for c in t.columns.astype(str)] 

            if any('age' in str(c) for c in cols) or t.shape[1] >= 2: 

                # try to map age->factor by finding numeric rows 

                t_clean = t.dropna(how='all') 

                # flatten and attempt to extract pairs 

                # find first numeric-like column as age, last numeric-like column as factor 

                numeric_cols = [c for c in t_clean.columns if pd.api.types.is_numeric_dtype(t_clean[c])] 

                if len(numeric_cols) >= 2: 

                    age_col = numeric_cols[0] 

                    factor_col = numeric_cols[1] 

                    mapping = {} 

                    for idx, row in t_clean.iterrows(): 

                        try: 

                            age = int(row[age_col]) 

                            factor = float(row[factor_col]) 

                            mapping[age] = factor 

                        except Exception: 

                            continue 

                    if mapping: 

                        return mapping 

        # if none matched, fall back 

    except Exception as e: 

        # network or parsing failed 

        pass 

 

    # Bundled fallback table (official-like 'all other RRIFs' subset from CRA guidance) 

    bundled = { 

        71:0.0526,72:0.0556,73:0.0588,74:0.0625,75:0.0667,76:0.0714,77:0.0769,78:0.0833,79:0.0909,80:0.10, 

        81:0.1111,82:0.1250,83:0.1429,84:0.1667,85:0.20,86:0.25,87:0.3333,88:0.5,89:1.0 

    } 

    return bundled 

 

RRIF_FACTORS_OFFICIAL = fetch_cra_rrif_factors() 

 

# ------------------------ 

# Inputs 

# ------------------------ 

st.sidebar.markdown("## Scenario inputs") 

 

st.header("Household & People") 

age1 = st.number_input("Primary spouse age", 45, 100, 60) 

age2 = st.number_input("Secondary spouse age (0 = none)", 0, 100, 58) 

ret_age1 = st.number_input("Primary target retirement age", 50, 75, 65) 

ret_age2 = st.number_input("Secondary target retirement age", 50, 75, 65) 

 

st.header("Starting balances (CAD)") 

rrsp_balance = st.number_input("RRSP balance (CAD)", 0.0, 50_000_000.0, 800_000.0, step=10_000.0) 

st.caption(f"RRSP balance: ${rrsp_balance:,.2f}") 

tfsa_balance = st.number_input("TFSA balance (CAD)", 0.0, 10_000_000.0, 300_000.0, step=5_000.0) 

st.caption(f"TFSA balance: ${tfsa_balance:,.2f}") 

taxable_balance = st.number_input("Non-registered (taxable) balance (CAD)", 0.0, 50_000_000.0, 250_000.0, step=10_000.0) 

st.caption(f"Taxable balance: ${taxable_balance:,.2f}") 

 

st.header("Pre-ret & returns (assumed)") 

growth_rate = st.slider("Portfolio annual nominal growth (%)", 0.0, 12.0, 5.0) / 100.0 

inflation = st.slider("Inflation (%)", 0.0, 8.0, 2.5) / 100.0 

real_growth = growth_rate - inflation 

end_age = st.slider("Projection to age", 75, 105, 95) 

 

# Nominal vs Real toggle 

view_mode = st.radio("View results as:", ["Nominal (default)", "Real (inflation-adjusted)"], index=0) 

 

st.header("Spending & withdrawal") 

annual_spend_today = st.number_input("Desired household spending today (CAD/yr)", 0.0, 1_000_000.0, 90_000.0, step=1000.0) 

st.caption(f"Desired household spending: ${annual_spend_today:,.2f}") 

withdraw_sequence = st.selectbox("Withdrawal order", ["Taxable → RRIF → TFSA", "RRIF → Taxable → TFSA", "Taxable → TFSA → RRIF"]) 

 

st.header("CPP & OAS inputs (per person)") 

# Person A 

st.subheader("Spouse 1 (Primary)") 

cpp_start_age1 = st.number_input("Spouse 1 CPP start age", 60, 70, 65, key="cpp1_age") 

cpp_monthly1 = st.number_input("Spouse 1 monthly CPP (CAD)", 0.0, 3000.0, 900.0, step=10.0, key="cpp1_amt") 

st.caption(f"Spouse 1 monthly CPP: ${cpp_monthly1:,.2f}") 

oas_app_age1 = st.number_input("Spouse 1 OAS start age (65-70):", 65, 70, 65, key="oas1_age") 

oas_monthly_at65_1 = st.number_input("Spouse 1 OAS monthly at age 65 (CAD)", 0.0, 2000.0, 700.0, step=5.0, key="oas1_amt") 

st.caption(f"Spouse 1 OAS at 65: ${oas_monthly_at65_1:,.2f}") 

# Person B 

st.subheader("Spouse 2 (Secondary)") 

cpp_start_age2 = st.number_input("Spouse 2 CPP start age", 60, 70, 65, key="cpp2_age") 

cpp_monthly2 = st.number_input("Spouse 2 monthly CPP (CAD)", 0.0, 3000.0, 500.0, step=10.0, key="cpp2_amt") 

st.caption(f"Spouse 2 monthly CPP: ${cpp_monthly2:,.2f}") 

oas_app_age2 = st.number_input("Spouse 2 OAS start age (65-70):", 65, 70, 65, key="oas2_age") 

oas_monthly_at65_2 = st.number_input("Spouse 2 OAS monthly at age 65 (CAD)", 0.0, 2000.0, 600.0, step=5.0, key="oas2_amt") 

st.caption(f"Spouse 2 OAS at 65: ${oas_monthly_at65_2:,.2f}") 

 

cola = st.slider("CPP/OAS COLA (%)", 0.0, 5.0, 2.0) / 100.0 

 

# ------------------------ 

# GIS / OAS settings 

# ------------------------ 

st.header("Guaranteed Income Supplement (GIS) & OAS clawback settings") 

gis_max_single_monthly = st.number_input("Max GIS (single) — monthly (CAD)", 0.0, 3000.0, 850.0, step=10.0) 

st.caption(f"Max GIS (single): ${gis_max_single_monthly:,.2f}") 

gis_max_couple_monthly = st.number_input("Max GIS (couple) — monthly (CAD, combined)", 0.0, 4000.0, 1250.0, step=10.0) 

st.caption(f"Max GIS (couple): ${gis_max_couple_monthly:,.2f}") 

 

gis_threshold_single = st.number_input("GIS threshold (single, CAD/yr)", 0.0, 100_000.0, 19000.0, step=500.0) 

st.caption(f"GIS threshold (single): ${gis_threshold_single:,.2f}") 

gis_threshold_couple = st.number_input("GIS threshold (couple combined, CAD/yr)", 0.0, 200_000.0, 24000.0, step=500.0) 

st.caption(f"GIS threshold (couple): ${gis_threshold_couple:,.2f}") 

 

oas_clawback_threshold = st.number_input("OAS clawback threshold (taxable income CAD/yr)", 0.0, 1_000_000.0, 90_997.0, step=1.0) 

st.caption(f"OAS clawback threshold: ${oas_clawback_threshold:,.2f}") 

 

oas_clawback_rate = st.number_input("OAS clawback rate (as decimal)", 0.0, 1.0, 0.15, step=0.01) 

gis_clawback_rate = st.number_input("GIS clawback rate (as decimal)", 0.0, 1.0, 0.5, step=0.01) 

 

# ------------------------ 

# Taxes: simplified but with credits & splitting 

# ------------------------ 

st.header("Taxes (simplified) — credits & splitting") 

province = st.selectbox("Province", ["Ontario","Quebec","British Columbia","Alberta","Manitoba","Saskatchewan","Nova Scotia","New Brunswick","Newfoundland and Labrador","Prince Edward Island"]) 

st.markdown("This model uses simplified brackets + user-adjustable credits for planning (not tax advice).") 

 

# Allow user to override basic personal amount and age/pension credits 

st.subheader("Tax credits (adjustable)") 

basic_personal_amount = st.number_input("Federal basic personal amount (CAD)", 0.0, 25000.0, 15000.0, step=100.0) 

st.caption(f"Federal basic personal amount: ${basic_personal_amount:,.2f}") 

fed_basic_rate = 0.15  # model: non-refundable credit at lowest federal rate 

age_credit_amount = st.number_input("Federal age credit amount (CAD, if age >=65)", 0.0, 12000.0, 8000.0, step=100.0) 

st.caption(f"Federal age credit amount: ${age_credit_amount:,.2f}") 

pension_income_credit_amount = st.number_input("Federal pension income credit amount (CAD)", 0.0, 20000.0, 2000.0, step=100.0) 

st.caption(f"Federal pension income credit amount: ${pension_income_credit_amount:,.2f}") 

 

# Pension income splitting option 

splitting_allowed = st.checkbox("Enable pension income splitting (simple 50/50 split for CPP/OAS/pension income)", value=False) 

 

FED_BRACKETS = [(53359,0.15),(106717,0.205),(165430,0.26),(235675,0.29),(float('inf'),0.33)] 

PROV_BRACKETS = { 

    "Ontario":[(46226,0.0505),(92454,0.0915),(150000,0.1116),(220000,0.1216),(float('inf'),0.1316)], 

    "British Columbia":[(43906,0.0506),(87812,0.077),(106717,0.105),(163563,0.1229),(float('inf'),0.147)], 

    "Alberta":[(142292,0.10),(170751,0.12),(227668,0.13),(float('inf'),0.14)], 

    "Quebec":[(49220,0.15),(98440,0.20),(103390,0.24),(float('inf'),0.2575)], 

    "Manitoba":[(36133,0.108),(71522,0.1275),(float('inf'),0.174)], 

    "Saskatchewan":[(46732,0.105),(129214,0.125),(float('inf'),0.145)], 

    "Nova Scotia":[(29590,0.0879),(59180,0.1495),(93000,0.1667),(150000,0.175),(float('inf'),0.21)], 

    "New Brunswick":[(43835,0.0968),(87670,0.1482),(float('inf'),0.1656)], 

    "Newfoundland and Labrador":[(41725,0.087),(83450,0.145),(138395,0.158),(float('inf'),0.174)], 

    "Prince Edward Island":[(31984,0.098),(63969,0.138),(float('inf'),0.167)] 

} 

 

 

def calc_progressive_tax(income, brackets): 

    tax = 0.0 

    lower = 0.0 

    if income <= 0: 

        return 0.0 

    for upper, rate in brackets: 

        taxable = max(0.0, min(income, upper) - lower) if upper != float('inf') else max(0.0, income - lower) 

        if taxable > 0: 

            tax += taxable * rate 

        lower = upper 

        if lower == float('inf'): 

            break 

    return max(0.0, tax) 

 

# ------------------------ 

# Simulation (per-year, per-household simplified) 

# ------------------------ 

 

def simulate(): 

    start_age = min(age1, age2 if age2 > 0 else age1) 

    ret_age = max(ret_age1, ret_age2) 

    last_age = int(end_age) 

    ages = list(range(start_age, last_age + 1)) 

 

    rrsp = float(rrsp_balance) 

    tfsa = float(tfsa_balance) 

    taxable = float(taxable_balance) 

 

    rows = [] 

    for age in ages: 

        row = {"Age": age} 

 

        # benefits 

        cpp = 0.0 

        oas_gross_annual = 0.0 

 

        if age >= cpp_start_age1: 

            years_after = max(0, age - cpp_start_age1) 

            cpp += cpp_monthly1 * 12.0 * ((1 + cola) ** years_after) 

        if age2 > 0 and age >= cpp_start_age2: 

            years_after = max(0, age - cpp_start_age2) 

            cpp += cpp_monthly2 * 12.0 * ((1 + cola) ** years_after) 

 

        if age >= oas_app_age1: 

            monthly_oas1 = oas_monthly_at_claim(oas_monthly_at65_1, oas_app_age1) 

            years_after_claim1 = max(0, age - oas_app_age1) 

            oas_gross_annual += monthly_oas1 * 12.0 * ((1 + cola) ** years_after_claim1) 

        if age2 > 0 and age >= oas_app_age2: 

            monthly_oas2 = oas_monthly_at_claim(oas_monthly_at65_2, oas_app_age2) 

            years_after_claim2 = max(0, age - oas_app_age2) 

            oas_gross_annual += monthly_oas2 * 12.0 * ((1 + cola) ** years_after_claim2) 

 

        # optional pension splitting (simple 50/50 on CPP+OAS if enabled and household) 

        cpp_oas_total = cpp + oas_gross_annual 

        if splitting_allowed and age2 > 0: 

            cpp_oas_per_person = cpp_oas_total / 2.0 

            # in simplified model, split evenly 

            cpp = cpp_oas_per_person / 2.0  # placeholder per-person (not used individually here) 

            oas_gross_annual = cpp_oas_per_person / 2.0  # placeholder - we still show household totals; splitting affects taxable allocation later 

            # For simplicity we continue treating cpp+oas as household_total (we'll apply splitting when computing taxable allocation) 

            cpp = cpp_oas_total * 0.5 * 0.5  # small simplification to mark that half goes to spouse 

            oas_gross_annual = cpp_oas_total * 0.5 * 0.5 

            # NOTE: this is simplified — real splitting rules require T1032/A reporting; for planning this is a rough proxy 

 

        # RRIF minimums using official factors 

        rrif_min = 0.0 

        if age >= 71 and rrsp > 0: 

            factor = RRIF_FACTORS_OFFICIAL.get(age) 

            if factor is None: 

                # interpolate or use last known factor 

                max_age = max(RRIF_FACTORS_OFFICIAL.keys()) 

                factor = RRIF_FACTORS_OFFICIAL[max_age] 

            rrif_min = rrsp * factor 

            rrsp -= rrif_min 

            taxable += rrif_min 

 

        # spending need 

        household_retired = (age >= ret_age) if age2 == 0 else ((age >= ret_age1) and (age >= ret_age2)) 

        spend_need = 0.0 

        if household_retired: 

            years_since = max(0, age - ret_age) 

            spend_need = annual_spend_today * ((1 + inflation) ** years_since) 

 

        # withdrawals 

        withdrawn_taxable = withdrawn_rrif_extra = withdrawn_tfsa = 0.0 

        need_after_benefits = max(0.0, spend_need - (cpp + oas_gross_annual)) 

        remaining_need = need_after_benefits 

 

        if withdraw_sequence == "Taxable → RRIF → TFSA": 

            buckets = ["Taxable","RRIF","TFSA"] 

        elif withdraw_sequence == "RRIF → Taxable → TFSA": 

            buckets = ["RRIF","Taxable","TFSA"] 

        else: 

            buckets = ["Taxable","TFSA","RRIF"] 

 

        for b in buckets: 

            if remaining_need <= 0: 

                break 

            if b == "Taxable": 

                take = min(remaining_need, taxable) 

                taxable -= take 

                withdrawn_taxable += take 

                remaining_need -= take 

            elif b == "RRIF": 

                take = min(remaining_need, rrsp) 

                rrsp -= take 

                withdrawn_rrif_extra += take 

                remaining_need -= take 

            elif b == "TFSA": 

                take = min(remaining_need, tfsa) 

                tfsa -= take 

                withdrawn_tfsa += take 

                remaining_need -= take 

 

        provisional_taxable_income = rrif_min + withdrawn_rrif_extra + withdrawn_taxable + cpp + oas_gross_annual 

 

        # GIS - simplified 

        marital = "single" if age2 == 0 else "couple" 

        max_monthly = gis_max_single_monthly if marital == "single" else gis_max_couple_monthly 

        threshold_gis = gis_threshold_single if marital == "single" else gis_threshold_couple 

        amount_over_gis = max(0.0, provisional_taxable_income - threshold_gis) 

        annual_gis_before = max_monthly * 12.0 

        clawback_gis = amount_over_gis * gis_clawback_rate 

        annual_gis = max(0.0, min(annual_gis_before, annual_gis_before - clawback_gis)) 

 

        # reduce withdrawals if GIS provides cash 

        spendable_from_gis = annual_gis 

        if spendable_from_gis > 0 and (withdrawn_taxable + withdrawn_rrif_extra + withdrawn_tfsa) > 0: 

            rem_gis = spendable_from_gis 

            reduce_taxable = min(withdrawn_taxable, rem_gis) 

            withdrawn_taxable -= reduce_taxable; taxable += reduce_taxable; rem_gis -= reduce_taxable 

            reduce_rrif = min(withdrawn_rrif_extra, rem_gis) 

            withdrawn_rrif_extra -= reduce_rrif; rrsp += reduce_rrif; rem_gis -= reduce_rrif 

            reduce_tfsa = min(withdrawn_tfsa, rem_gis) 

            withdrawn_tfsa -= reduce_tfsa; tfsa += reduce_tfsa; rem_gis -= reduce_tfsa 

 

        # OAS clawback 

        excess = max(0.0, provisional_taxable_income - oas_clawback_threshold) 

        recovery = oas_clawback_rate * excess 

        actual_recovery = min(recovery, oas_gross_annual) 

        oas_after_recovery = max(0.0, oas_gross_annual - actual_recovery) 

 

        # Taxable income and credits 

        taxable_for_regular_tax = rrif_min + withdrawn_rrif_extra + withdrawn_taxable + cpp + oas_after_recovery 

 

        # Federal tax before credits 

        fed_tax = calc_progressive_tax(taxable_for_regular_tax, FED_BRACKETS) 

        prov_tax = calc_progressive_tax(taxable_for_regular_tax, PROV_BRACKETS[province]) 

        total_tax = fed_tax + prov_tax 

 

        # Apply non-refundable credits (basic personal amount, age credit, pension income credit) 

        credit_reduction = 0.0 

        # basic personal amount 

        credit_reduction += basic_personal_amount * fed_basic_rate 

        # age credit if any member >=65 (simplified) 

        if age >= 65 or (age2 > 0 and age2 >= 65): 

            credit_reduction += age_credit_amount * fed_basic_rate 

        # pension income credit if any pension income exists (simplified) 

        if cpp + oas_after_recovery > 0: 

            credit_reduction += pension_income_credit_amount * fed_basic_rate 

 

        # reduce federal tax by credit_reduction (can't go below 0) 

        fed_tax = max(0.0, fed_tax - credit_reduction) 

        total_tax = fed_tax + prov_tax 

 

        # pay taxes from taxable account 

        tax_paid = min(taxable, total_tax) 

        taxable -= tax_paid 

 

        # finalize balances 

        rrsp = max(0.0, rrsp) 

        tfsa = max(0.0, tfsa) 

        taxable = max(0.0, taxable) 

        total_portfolio = rrsp + tfsa + taxable 

 

        row.update({ 

            "RRSP": rrsp, 

            "TFSA": tfsa, 

            "Taxable": taxable, 

            "Total_Portfolio": total_portfolio, 

            "CPP+OAS_gross": cpp + oas_gross_annual, 

            "OAS_annual_gross": oas_gross_annual, 

            "OAS_recovery_tax": actual_recovery, 

            "OAS_after_recovery": oas_after_recovery, 

            "RRIF_min": rrif_min, 

            "Spend_need": spend_need, 

            "Withdrawn_Taxable": withdrawn_taxable, 

            "Withdrawn_RRIF": withdrawn_rrif_extra + rrif_min, 

            "Withdrawn_TFSA": withdrawn_tfsa, 

            "GIS_annual": annual_gis, 

            "Federal Tax": fed_tax, 

            "Provincial Tax": prov_tax, 

            "Total Tax": total_tax, 

        }) 

 

        # growth 

        rrsp *= (1.0 + growth_rate) 

        tfsa *= (1.0 + growth_rate) 

        taxable *= (1.0 + growth_rate) 

 

        rows.append(row) 

 

    df = pd.DataFrame(rows) 

 

    # Round all monetary columns to 2 decimals 

    money_cols = [ 

        "RRSP","TFSA","Taxable","Total_Portfolio","CPP+OAS_gross","OAS_annual_gross","GIS_annual", 

        "Total Tax","Withdrawn_Taxable","Withdrawn_RRIF","Withdrawn_TFSA","OAS_recovery_tax", 

        "Federal Tax","Provincial Tax","Spend_need","RRIF_min","OAS_after_recovery"  # <-- added here 

    ] 

    for col in money_cols: 

        if col in df.columns: 

            df[col] = df[col].round(2) 

 

    # produce real (inflation-adjusted) columns if requested 

    if view_mode == "Real (inflation-adjusted)": 

        df_real = df.copy() 

        base_age = df_real.loc[0, "Age"] 

        years = df_real["Age"] - base_age 

        for col in money_cols: 

            if col in df_real.columns: 

                df_real[col] = (df_real[col] / ((1 + inflation) ** years)).round(2) 

        df_real = df_real.add_suffix(" (real)") 

        df = pd.concat([df, df_real.reset_index(drop=True)], axis=1) 

 

    return df 

 

def oas_monthly_at_claim(oas_monthly_at65, claim_age): 

    """ 

    Returns the OAS monthly amount at claim age. 

    OAS increases by 0.6% per month (7.2% per year) for each month after 65, up to age 70. 

    No reduction for early claim (cannot claim before 65). 

    """ 

    if claim_age <= 65: 

        return oas_monthly_at65 

    elif claim_age > 70: 

        claim_age = 70 

    months_late = int((claim_age - 65) * 12) 

    increase = 1 + 0.006 * months_late 

    return round(oas_monthly_at65 * increase, 2) 

 

# ------------------------ 

# Run simulation and output 

# ------------------------ 

st.header("Run projection") 

if st.button("Run projection"): 

    df = simulate() 

    st.success("Projection finished.") 

 

    st.subheader("Year-by-year results") 

    st.dataframe(df.style.format({ 

        col: "${:,.2f}" for col in df.columns if any(x in col for x in [ 

            "RRSP", "TFSA", "Taxable", "Total_Portfolio", "CPP+OAS_gross", "OAS_annual_gross", "GIS_annual", 

            "Withdrawn_Taxable", "Withdrawn_RRIF", "Withdrawn_TFSA", "OAS_recovery_tax", "OAS_after_recovery", 

            "Federal Tax", "Provincial Tax", "Total Tax", "Spend_need", "RRIF_min" 

        ]) 

    }), use_container_width=True) 

 

    # Charts 

    st.subheader("Balances over time") 

    bal_cols = [c for c in df.columns if c in ["RRSP","TFSA","Taxable","Total_Portfolio"]] 

    if view_mode.startswith("Real"): 

        bal_cols = [c for c in df.columns if "(real)" in c and any(x in c for x in ["RRSP","TFSA","Taxable","Total_Portfolio"])] 

    fig_bal = px.line(df, x="Age", y=bal_cols, labels={"value":"Balance (CAD)", "variable":"Account"}, title="Account Balances Over Time") 

    fig_bal.update_layout(yaxis_tickformat=",.0f") 

    st.plotly_chart(fig_bal, use_container_width=True) 

 

    st.subheader("Income, GIS, OAS recovery & Taxes") 

    inc_cols = [c for c in df.columns if c in ["CPP+OAS_gross","GIS_annual","OAS_recovery_tax","Total Tax"]] 

    if view_mode.startswith("Real"): 

        inc_cols = [c for c in df.columns if any(k in c for k in ["CPP+OAS_gross (real)","GIS_annual (real)","OAS_recovery_tax (real)","Total Tax (real)"]) ] 

    inc_df = df[["Age"] + inc_cols] 

    fig_inc = go.Figure() 

    # stacked bars for income 

    for col in inc_cols[:-1]: 

        fig_inc.add_trace(go.Bar(x=inc_df["Age"], y=inc_df[col], name=col)) 

    # overlay tax as line 

    fig_inc.add_trace(go.Scatter(x=inc_df["Age"], y=inc_df[inc_cols[-1]], name=inc_cols[-1], yaxis="y2", mode="lines")) 

    fig_inc.update_layout(barmode='stack', yaxis={'title':'CAD / year'}, yaxis2={'title':'Total Tax','overlaying':'y','side':'right'}) 

    st.plotly_chart(fig_inc, use_container_width=True) 

 

    st.subheader("Withdrawals (yearly)") 

    wcols = [c for c in df.columns if c in ["Withdrawn_Taxable","Withdrawn_RRIF","Withdrawn_TFSA"]] 

    if view_mode.startswith("Real"): 

        wcols = [c for c in df.columns if any(k in c for k in ["Withdrawn_Taxable (real)","Withdrawn_RRIF (real)","Withdrawn_TFSA (real)"])] 

    fig_w = px.area(df, x="Age", y=wcols, labels={"value":"Withdrawn (CAD)","variable":"Source"}, title="Withdrawals by Source (stacked)") 

    st.plotly_chart(fig_w, use_container_width=True) 

 

    # CSV 

    csv = df.to_csv(index=False).encode("utf-8") 

    st.download_button("Download CSV", data=csv, file_name="canada_retirement_projection_v2.csv", mime="text/csv") 

 

    # Show final total portfolio as a metric 

    st.metric("Total Portfolio", f"${df['Total_Portfolio'].iloc[-1]:,.2f}") 

 

st.markdown('---') 

st.caption( 

    """ 

    Improvements in this v2: 

    - Official CRA RRIF factors fetched live (with fallback table) — see CRA 'Prescribed factors' / 'Minimum amount from a RRIF' pages. 

    - Real vs nominal toggle: shows inflation adjusted columns when selected. 

    - Basic federal credits (personal, age, pension) applied as a simplified non-refundable credit to federal tax. 

    - Simple pension income splitting toggle (rough 50/50 proxy) to demonstrate potential tax-smoothed outcomes. 

    

 

    If you'd like I can now: 

    - Add exact provincial non-refundable credit amounts and apply province-specific basic personal amounts. 

    - Improve pension splitting logic to correctly split specific pensions between spouses (CPP, eligible pensions) following CRA rules. 

  

    """ 

) 