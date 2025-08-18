# 🇮🇳 Retirement Planning Calculator — India (Single or Couple)
# Run with:  streamlit run retirement_Calculator_India_app.py
#
# Notes:
# - End-to-end single-file Streamlit app.
# - Clean UI with tabs, consistent 2‑decimal currency formatting in Indian style.
# - Charts offer Line/Bar/Area options.
# - Matches & clarifies assumptions embedded in the original file, with safer currency formatting and refactors.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# =============================
# Utilities
# =============================

def inr_format(x: float) -> str:
    """Format a number as Indian Rupees with two decimals and Indian digit grouping.
    Example: 12345678.9 -> ₹1,23,45,678.90
    """
    try:
        neg = x < 0
        x = abs(float(x))
    except Exception:
        return str(x)

    s = f"{x:.2f}"
    if "." in s:
        whole, frac = s.split(".")
    else:
        whole, frac = s, "00"

    if len(whole) <= 3:
        g = whole
    else:
        g = whole[-3:]
        whole = whole[:-3]
        while whole:
            g = whole[-2:] + "," + g
            whole = whole[:-2]
    sign = "-" if neg else ""
    return f"{sign}₹{g}.{frac}"


def real_to_nominal(real_rate: float, inflation: float) -> float:
    return (1 + real_rate) * (1 + inflation) - 1


def nominal_to_real(nominal_rate: float, inflation: float) -> float:
    return (1 + nominal_rate) / (1 + inflation) - 1


# =============================
# Data Models
# =============================

@dataclass
class PersonInputs:
    name: str
    current_age: int
    retire_age: int
    life_expectancy: int

    # Balances (₹)
    bal_epf: float
    bal_nps: float
    bal_ppf: float
    bal_other: float

    # Salary & contributions
    annual_salary: float
    epf_employee_pct: float
    epf_employer_pct: float
    nps_employee_pct: float
    nps_employer_pct: float
    ppf_annual: float
    other_annual_invest: float

    # Returns (nominal)
    r_epf_pre: float
    r_nps_pre: float
    r_ppf_pre: float
    r_other_pre: float

    r_epf_post: float
    r_nps_post: float
    r_ppf_post: float
    r_other_post: float

    # Effective tax sliders
    tax_on_salary: float
    tax_on_withdrawals: float
    tax_on_annuity: float

    # NPS decisions
    nps_lumpsum_pct: float
    nps_annuity_rate: float


@dataclass
class HouseholdInputs:
    include_spouse: bool
    inflation: float
    med_inflation: float
    monthly_spend_today: float
    medical_spend_today: float
    one_time_goal_today: float
    one_time_goal_year: int
    invest_real_return_after_ret: float


# =============================
# UI Builders
# =============================

def person_form(label: str, default_name: str, key_prefix: str = "") -> PersonInputs:
    st.subheader(f"{label} inputs")

    colA, colB, colC = st.columns(3)
    with colA:
        name = st.text_input("Name", default_name, key=f"{key_prefix}name")
        current_age = int(st.number_input("Current age", 20, 80, 40, key=f"{key_prefix}age"))
        retire_age = int(st.number_input("Target retirement age", 30, 75, 60, key=f"{key_prefix}retire_age"))
    with colB:
        life_expectancy = int(st.number_input("Life expectancy (age)", retire_age+1, 110, 90, key=f"{key_prefix}life"))
        annual_salary = float(st.number_input("Annual salary (₹)", 0.0, 10_00_00_000.0, 12_00_000.0, step=50_000.0, key=f"{key_prefix}salary"))
        tax_on_salary = st.slider("Effective tax on salary (%)", 0.0, 50.0, 10.0, 0.5, key=f"{key_prefix}taxsal")/100
    with colC:
        tax_on_withdrawals = st.slider("Effective tax on withdrawals (%)", 0.0, 50.0, 5.0, 0.5, key=f"{key_prefix}taxw")/100
        tax_on_annuity = st.slider("Effective tax on annuity (%)", 0.0, 50.0, 5.0, 0.5, key=f"{key_prefix}taxann")/100

    st.markdown("**Current Balances (₹)**")
    c1, c2, c3, c4 = st.columns(4)
    bal_epf = float(c1.number_input("EPF balance", 0.0, 50_00_00_000.0, 5_00_000.0, step=25_000.0, key=f"{key_prefix}bal_epf"))
    bal_nps = float(c2.number_input("NPS balance", 0.0, 50_00_00_000.0, 2_00_000.0, step=25_000.0, key=f"{key_prefix}bal_nps"))
    bal_ppf = float(c3.number_input("PPF balance", 0.0, 50_00_00_000.0, 1_00_000.0, step=25_000.0, key=f"{key_prefix}bal_ppf"))
    bal_other = float(c4.number_input("Other/Taxable balance", 0.0, 50_00_00_000.0, 3_00_000.0, step=25_000.0, key=f"{key_prefix}bal_other"))

    st.markdown("**Annual Contributions**")
    c5, c6, c7, c8 = st.columns(4)
    epf_employee_pct = c5.slider("EPF employee (% of salary)", 0.0, 50.0, 12.0, 0.5, key=f"{key_prefix}epf_emp")/100
    epf_employer_pct = c6.slider("EPF employer (% of salary)", 0.0, 50.0, 12.0, 0.5, key=f"{key_prefix}epf_empr")/100
    nps_employee_pct = c7.slider("NPS employee (% of salary)", 0.0, 50.0, 10.0, 0.5, key=f"{key_prefix}nps_emp")/100
    nps_employer_pct = c8.slider("NPS employer (% of salary)", 0.0, 50.0, 0.0, 0.5, key=f"{key_prefix}nps_empr")/100

    c9, c10 = st.columns(2)
    ppf_annual = float(c9.number_input("PPF (₹ per year)", 0.0, 15_00_00_000.0, 1_50_000.0, step=10_000.0, key=f"{key_prefix}ppf"))
    other_annual_invest = float(c10.number_input("Other investments (₹ per year)", 0.0, 50_00_00_000.0, 1_00_000.0, step=10_000.0, key=f"{key_prefix}otherinv"))

    st.markdown("**Expected Annual Returns (Nominal %)**")
    r1, r2, r3, r4 = st.columns(4)
    r_epf_pre = r1.slider("EPF (pre-retire)", 0.0, 20.0, 8.0, 0.1, key=f"{key_prefix}r_epf_pre")/100
    r_nps_pre = r2.slider("NPS (pre-retire)", 0.0, 20.0, 10.0, 0.1, key=f"{key_prefix}r_nps_pre")/100
    r_ppf_pre = r3.slider("PPF (pre-retire)", 0.0, 20.0, 7.1, 0.1, key=f"{key_prefix}r_ppf_pre")/100
    r_other_pre = r4.slider("Other (pre-retire)", 0.0, 25.0, 10.0, 0.1, key=f"{key_prefix}r_oth_pre")/100

    s1, s2, s3, s4 = st.columns(4)
    r_epf_post = s1.slider("EPF (post-retire)", 0.0, 20.0, 7.5, 0.1, key=f"{key_prefix}r_epf_post")/100
    r_nps_post = s2.slider("NPS (post-retire)", 0.0, 20.0, 8.0, 0.1, key=f"{key_prefix}r_nps_post")/100
    r_ppf_post = s3.slider("PPF (post-retire)", 0.0, 20.0, 7.0, 0.1, key=f"{key_prefix}r_ppf_post")/100
    r_other_post = s4.slider("Other (post-retire)", 0.0, 25.0, 8.0, 0.1, key=f"{key_prefix}r_oth_post")/100

    st.markdown("**NPS at Retirement**")
    t1, t2 = st.columns(2)
    nps_lumpsum_pct = t1.slider("NPS lump sum at retirement (%)", 0.0, 60.0, 60.0, 1.0, key=f"{key_prefix}nps_ls")/100
    nps_annuity_rate = t2.slider("Annuity yield on remaining NPS (%)", 0.0, 12.0, 6.0, 0.1, key=f"{key_prefix}nps_ann")/100

    return PersonInputs(
        name=name,
        current_age=current_age,
        retire_age=retire_age,
        life_expectancy=life_expectancy,
        bal_epf=bal_epf,
        bal_nps=bal_nps,
        bal_ppf=bal_ppf,
        bal_other=bal_other,
        annual_salary=annual_salary,
        epf_employee_pct=epf_employee_pct,
        epf_employer_pct=epf_employer_pct,
        nps_employee_pct=nps_employee_pct,
        nps_employer_pct=nps_employer_pct,
        ppf_annual=ppf_annual,
        other_annual_invest=other_annual_invest,
        r_epf_pre=r_epf_pre,
        r_nps_pre=r_nps_pre,
        r_ppf_pre=r_ppf_pre,
        r_other_pre=r_other_pre,
        r_epf_post=r_epf_post,
        r_nps_post=r_nps_post,
        r_ppf_post=r_ppf_post,
        r_other_post=r_other_post,
        tax_on_salary=tax_on_salary,
        tax_on_withdrawals=tax_on_withdrawals,
        tax_on_annuity=tax_on_annuity,
        nps_lumpsum_pct=nps_lumpsum_pct,
        nps_annuity_rate=nps_annuity_rate,
    )


def household_form() -> HouseholdInputs:
    st.subheader("Household & Assumptions")
    include_spouse = st.checkbox("Add spouse", value=True)
    c1, c2, c3 = st.columns(3)
    inflation = c1.slider("CPI inflation (%)", 0.0, 12.0, 5.0, 0.1)/100
    med_inflation = c2.slider("Medical inflation (%)", 0.0, 15.0, 7.0, 0.1)/100
    invest_real_return_after_ret = c3.slider("Expected real return after retirement (%)", -5.0, 10.0, 2.0, 0.1)/100

    c4, c5 = st.columns(2)
    monthly_spend_today = c4.number_input("Monthly household spending today (₹)", 0.0, 1_00_00_000.0, 1_00_000.0, step=5_000.0)
    medical_spend_today = c5.number_input("Monthly medical spending today (₹)", 0.0, 1_00_00_000.0, 10_000.0, step=1_000.0)

    c6, c7 = st.columns(2)
    one_time_goal_today = c6.number_input("One-time goal at/after retirement (₹ today)", 0.0, 50_00_00_000.0, 0.0, step=50_000.0)
    one_time_goal_year = int(c7.number_input("One-time goal year offset (0 = at retirement)", -10, 40, 0))

    return HouseholdInputs(
        include_spouse=include_spouse,
        inflation=inflation,
        med_inflation=med_inflation,
        monthly_spend_today=monthly_spend_today,
        medical_spend_today=medical_spend_today,
        one_time_goal_today=one_time_goal_today,
        one_time_goal_year=one_time_goal_year,
        invest_real_return_after_ret=invest_real_return_after_ret,
    )


# =============================
# Core Math
# =============================

def grow_pre_retirement(p: PersonInputs, years: int) -> Tuple[float, float, float, float]:
    epf, nps, ppf, other = p.bal_epf, p.bal_nps, p.bal_ppf, p.bal_other
    for _ in range(years):
        # Contributions
        epf += (p.annual_salary * (p.epf_employee_pct + p.epf_employer_pct)) * (1 - p.tax_on_salary)
        nps += (p.annual_salary * (p.nps_employee_pct + p.nps_employer_pct)) * (1 - p.tax_on_salary)
        ppf += p.ppf_annual
        other += p.other_annual_invest
        # Growth
        epf *= (1 + p.r_epf_pre)
        nps *= (1 + p.r_nps_pre)
        ppf *= (1 + p.r_ppf_pre)
        other *= (1 + p.r_other_pre)
    return epf, nps, ppf, other


def nps_split_at_retirement(nps_corpus: float, p: PersonInputs) -> Tuple[float, float]:
    ls = nps_corpus * p.nps_lumpsum_pct
    return ls, nps_corpus - ls


def annuity_income_from_principal(principal: float, rate: float) -> float:
    return principal * rate


def real_required_spend(hh: HouseholdInputs, y_from_ret: int) -> float:
    base = hh.monthly_spend_today * 12 * ((1 + hh.inflation) ** y_from_ret)
    medical = hh.medical_spend_today * 12 * ((1 + hh.med_inflation) ** y_from_ret)
    return base + medical


def one_time_goal_nominal(hh: HouseholdInputs, years_from_ret: int) -> float:
    return hh.one_time_goal_today * ((1 + hh.inflation) ** years_from_ret)


def withdraw_in_order(dem: float, balances: Dict[str, float], post_rates: Dict[str, float], tax_rate_withdrawals: float) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Waterfall: Other -> PPF -> EPF -> NPS lump sum.
    Returns: (tax_paid, new_balances, withdrawn_breakdown)
    Assumed taxation: PPF & NPS lump = tax-free; Other & EPF taxed at tax_rate_withdrawals.
    """
    remaining = dem
    tax_paid = 0.0
    w = {k: 0.0 for k in balances.keys()}

    for k in ["other", "ppf", "epf", "nps_lumpsum"]:
        if remaining <= 0:
            break
        take = min(remaining, balances.get(k, 0.0))
        if take > 0:
            w[k] += take
            balances[k] -= take
            if k in ("other", "epf"):
                tax_paid += take * tax_rate_withdrawals
            remaining -= take

    # Grow what remains for next year
    for k, r in post_rates.items():
        balances[k] *= (1 + r)

    return tax_paid, balances, w


# =============================
# Simulation
# =============================

def simulate(hh: HouseholdInputs, p1: PersonInputs, p2: Optional[PersonInputs]) -> pd.DataFrame:
    # Timeline
    end_age = max(p1.life_expectancy, (p2.life_expectancy if p2 else p1.life_expectancy))

    years_to_ret1 = p1.retire_age - p1.current_age
    p1_epf, p1_nps, p1_ppf, p1_other = grow_pre_retirement(p1, max(0, years_to_ret1))

    p2_epf = p2_nps = p2_ppf = p2_other = 0.0
    if p2:
        years_to_ret2 = p2.retire_age - p2.current_age
        p2_epf, p2_nps, p2_ppf, p2_other = grow_pre_retirement(p2, max(0, years_to_ret2))

    # Align household retirement at the later retiree's age
    ret_start_age = max(p1.retire_age, (p2.retire_age if p2 else p1.retire_age))

    def grow_late(epf, nps, ppf, other, years, r_epf, r_nps, r_ppf, r_other):
        for _ in range(max(0, years)):
            epf *= (1 + r_epf)
            nps *= (1 + r_nps)
            ppf *= (1 + r_ppf)
            other *= (1 + r_other)
        return epf, nps, ppf, other

    # Grow earlier retiree forward to ret_start_age
    if p1.retire_age < ret_start_age:
        offs1 = ret_start_age - p1.retire_age
        p1_epf, p1_nps, p1_ppf, p1_other = grow_late(p1_epf, p1_nps, p1_ppf, p1_other, offs1, p1.r_epf_post, p1.r_nps_post, p1.r_ppf_post, p1.r_other_post)
    if p2 and p2.retire_age < ret_start_age:
        offs2 = ret_start_age - p2.retire_age
        p2_epf, p2_nps, p2_ppf, p2_other = grow_late(p2_epf, p2_nps, p2_ppf, p2_other, offs2, p2.r_epf_post, p2.r_nps_post, p2.r_ppf_post, p2.r_other_post)

    # Combine balances
    epf = p1_epf + p2_epf
    nps = p1_nps + p2_nps
    ppf = p1_ppf + p2_ppf
    other = p1_other + p2_other

    # NPS split & annuity
    ls1, ann_pr1 = nps_split_at_retirement(p1_nps, p1)
    ls2, ann_pr2 = (0.0, 0.0)
    if p2:
        ls2, ann_pr2 = nps_split_at_retirement(p2_nps, p2)
    nps_lumpsum = ls1 + ls2
    nps_annuity_principal = ann_pr1 + ann_pr2
    annuity_income = annuity_income_from_principal(ann_pr1, p1.nps_annuity_rate) + (annuity_income_from_principal(ann_pr2, p2.nps_annuity_rate) if p2 else 0.0)

    # Average post-retirement rates across persons (simple approach)
    denom = 2 if p2 else 1
    post_rates = {
        "epf": (p1.r_epf_post + (p2.r_epf_post if p2 else p1.r_epf_post)) / denom,
        "nps_lumpsum": (p1.r_nps_post + (p2.r_nps_post if p2 else p1.r_nps_post)) / denom,
        "ppf": (p1.r_ppf_post + (p2.r_ppf_post if p2 else p1.r_ppf_post)) / denom,
        "other": (p1.r_other_post + (p2.r_other_post if p2 else p1.r_other_post)) / denom,
    }

    # Iterate years from ret_start_age up to end_age
    ret_years = end_age - ret_start_age + 1
    balances = {"epf": epf, "nps_lumpsum": nps_lumpsum, "ppf": ppf, "other": other}

    rows: List[Dict[str, float]] = []
    for y in range(max(0, ret_years)):
        required_spend = real_required_spend(hh, y)
        goal = one_time_goal_nominal(hh, y) if (hh.one_time_goal_today > 0 and y == hh.one_time_goal_year) else 0.0
        demand = required_spend + goal

        # Annuity & its tax
        denom_tax = 2 if p2 else 1
        ann_tax_rate = (p1.tax_on_annuity + (p2.tax_on_annuity if p2 else p1.tax_on_annuity)) / denom_tax
        ann_tax = annuity_income * ann_tax_rate
        demand_after_annuity = max(0.0, demand - (annuity_income - ann_tax))

        tax_paid, balances, w = withdraw_in_order(
            demand_after_annuity,
            balances,
            post_rates,
            tax_rate_withdrawals=(p1.tax_on_withdrawals + (p2.tax_on_withdrawals if p2 else p1.tax_on_withdrawals)) / denom_tax,
        )

        end_total = sum(balances.values())
        rows.append({
            "Year From Retirement": y,
            "Required Spend (₹)": demand,
            "Annuity Income (₹)": annuity_income,
            "Tax on Annuity (₹)": ann_tax,
            "To Fund from Corpus (₹)": demand_after_annuity,
            "Withdrawn-Other (₹)": w.get("other", 0.0),
            "Withdrawn-PPF (₹)": w.get("ppf", 0.0),
            "Withdrawn-EPF (₹)": w.get("epf", 0.0),
            "Withdrawn-NPS Lump (₹)": w.get("nps_lumpsum", 0.0),
            "Tax on Withdrawals (₹)": tax_paid,
            "End Bal EPF (₹)": balances.get("epf", 0.0),
            "End Bal NPS Lump (₹)": balances.get("nps_lumpsum", 0.0),
            "End Bal PPF (₹)": balances.get("ppf", 0.0),
            "End Bal Other (₹)": balances.get("other", 0.0),
            "End Total Corpus (₹)": end_total,
        })

    return pd.DataFrame(rows)


# =============================
# App
# =============================

st.set_page_config(page_title="India Retirement Planner (Single/Couple)", layout="wide")

st.title("🇮🇳 Retirement Planning Calculator — India (Single or Couple)")

hh = household_form()
left, right = st.columns(2)
with left:
    p1 = person_form("Person A", "You", key_prefix="p1_")
with right:
    p2 = person_form("Person B (Spouse)", "Spouse", key_prefix="p2_") if hh.include_spouse else None

if p1.retire_age < p1.current_age or (p2 and p2.retire_age < p2.current_age):
    st.error("Retirement age must be greater than current age.")
    st.stop()

if st.button("Run Projection", type="primary"):
    df = simulate(hh, p1, p2)

    # ================= UI: Tabs =================
    t1, t2, t3, t4 = st.tabs(["Summary", "Charts", "Year-by-Year Table", "Notes"])

    with t1:
        st.subheader("Key Takeaways")
        end_corpus = df["End Total Corpus (₹)"].iloc[-1]
        first_req = df["Required Spend (₹)"].iloc[0]
        first_ann = df["Annuity Income (₹)"].iloc[0]
        first_ann_tax = df["Tax on Annuity (₹)"].iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("End Total Corpus", inr_format(end_corpus))
        col2.metric("First-Year Spending (nominal)", inr_format(first_req))
        col3.metric("First-Year Annuity (pre-tax)", inr_format(first_ann))
        st.caption(f"Taxes on annuity in first year: {inr_format(first_ann_tax)}")

    with t2:
        style = st.radio("Chart style", ["Line", "Bar", "Area"], horizontal=True)

        st.markdown("#### Balances Over Time")
        idx = df.set_index("Year From Retirement")
        bal_cols = ["End Bal EPF (₹)", "End Bal NPS Lump (₹)", "End Bal PPF (₹)", "End Bal Other (₹)"]
        if style == "Line":
            st.line_chart(idx[bal_cols])
        elif style == "Bar":
            st.bar_chart(idx[bal_cols])
        else:
            st.area_chart(idx[bal_cols])

        st.markdown("#### Withdrawals Breakdown")
        w_cols = ["Withdrawn-Other (₹)", "Withdrawn-PPF (₹)", "Withdrawn-EPF (₹)", "Withdrawn-NPS Lump (₹)"]
        if style == "Line":
            st.line_chart(idx[w_cols])
        elif style == "Bar":
            st.bar_chart(idx[w_cols])
        else:
            st.area_chart(idx[w_cols])

        st.markdown("#### Spending vs Funding")
        comp = idx[["Required Spend (₹)", "Annuity Income (₹)", "To Fund from Corpus (₹)"]]
        if style == "Line":
            st.line_chart(comp)
        elif style == "Bar":
            st.bar_chart(comp)
        else:
            st.area_chart(comp)

    with t3:
        st.subheader("Year-by-Year Details")
        fmt_df = df.copy()
        money_cols = [c for c in fmt_df.columns if "₹" in c]
        for c in money_cols:
            fmt_df[c] = fmt_df[c].apply(inr_format)
        st.dataframe(fmt_df, use_container_width=True)
        st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="india_retirement_projection.csv", mime="text/csv")

    with t4:
        st.markdown(
            """
            **Modeling assumptions (simplified, user-controlled):**
            - Household spending and medical spending inflate at separate rates.
            - EPF, NPS, PPF, and Other balances grow at your chosen nominal rates.
            - Contributions are simple % of salary (for EPF/NPS) or rupee amounts (PPF/Other).
            - Salary tax slider reduces investable amount going into EPF/NPS (a simple approximation).
            - At retirement, NPS is split into a tax-free lump sum (by your chosen %) and an annuity from the remainder at your chosen yield.
            - Withdrawals occur in order: **Other → PPF → EPF → NPS lump**. PPF & NPS lump are treated as tax-free; EPF & Other are taxed at your withdrawal-tax slider.
            - Post-retirement growth rates are averaged across spouses for the combined household pools.
            - All currency values display with 2 decimals in Indian format.
            """
        )
