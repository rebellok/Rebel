# retirement_planner_app.py
# Streamlit Retirement Planner with Roth Conversions, RMDs, IRMAA, and Flexible Tax Brackets
# Author: ChatGPT (GPT-5 Thinking)
# Date: 2025-09-02
"""
Single-file Streamlit app that:
- Tracks taxable, tax-deferred, and tax-free (Roth) balances for self (+optional spouse).
- Accepts annual savings until retirement by bucket.
- Models Roth conversions (taxes paid from taxable) between chosen years.
- Applies progressive federal income tax using a JSON file of brackets for many years; if a future year is missing, uses the latest year in the file.
- Supports withdrawals starting from a chosen year with a configurable bucket order and tax-source choice.
- Computes RMDs using IRS Uniform Lifetime Table (simplified) at the applicable age (estimated from birth year), and forces RMDs before other withdrawal choices.
- Estimates Social Security benefits by start age for self and spouse (FRA assumed 67; early/late factors approximated) and includes basic taxation of SS benefits.
- Estimates Medicare IRMAA Part B & D surcharges from an external JSON file (with a 2-year MAGI lookback; if missing, uses latest year).
- Builds a full-year grid of the scenario and shows interactive charts.

Files expected (optional; sensible fallbacks included):
- data/tax_brackets.json     # progressive brackets, standard deductions by filing status per calendar year
- data/irmaa.json            # IRMAA thresholds and surcharges per year and filing status

Run:  
    streamlit run retirement_planner_app.py
"""

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# Utility: Safe JSON loading
# -----------------------------

def load_json_safe(path: str, fallback: dict) -> dict:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return fallback

# -----------------------------
# Fallback Data (can be edited)
# -----------------------------

FALLBACK_TAX_BRACKETS = {
    "2024": {
        "standard_deduction": {"single": 14600, "married": 29200},
        "brackets": {
            "single": [
                {"rate": 0.10, "up_to": 11600},
                {"rate": 0.12, "up_to": 47150},
                {"rate": 0.22, "up_to": 100525},
                {"rate": 0.24, "up_to": 191950},
                {"rate": 0.32, "up_to": 243725},
                {"rate": 0.35, "up_to": 609350},
                {"rate": 0.37, "up_to": None},
            ],
            "married": [
                {"rate": 0.10, "up_to": 23200},
                {"rate": 0.12, "up_to": 94300},
                {"rate": 0.22, "up_to": 201050},
                {"rate": 0.24, "up_to": 383900},
                {"rate": 0.32, "up_to": 487450},
                {"rate": 0.35, "up_to": 731200},
                {"rate": 0.37, "up_to": None},
            ],
        },
        # Qualified dividends/LTCG simple brackets (approximate; for demo)
        "qdi_brackets": {
            "single": [
                {"rate": 0.00, "up_to": 47125},
                {"rate": 0.15, "up_to": 518900},
                {"rate": 0.20, "up_to": None},
            ],
            "married": [
                {"rate": 0.00, "up_to": 94250},
                {"rate": 0.15, "up_to": 583750},
                {"rate": 0.20, "up_to": None},
            ],
        },
    },
    # 2025 mirrors 2024 for fallback; update your JSON when official values differ
    "2025": {
        "standard_deduction": {"single": 15000, "married": 30000},
        "brackets": {
            "single": [
                {"rate": 0.10, "up_to": 12000},
                {"rate": 0.12, "up_to": 48000},
                {"rate": 0.22, "up_to": 102000},
                {"rate": 0.24, "up_to": 195000},
                {"rate": 0.32, "up_to": 247000},
                {"rate": 0.35, "up_to": 620000},
                {"rate": 0.37, "up_to": None},
            ],
            "married": [
                {"rate": 0.10, "up_to": 24000},
                {"rate": 0.12, "up_to": 96000},
                {"rate": 0.22, "up_to": 204000},
                {"rate": 0.24, "up_to": 390000},
                {"rate": 0.32, "up_to": 494000},
                {"rate": 0.35, "up_to": 740000},
                {"rate": 0.37, "up_to": None},
            ],
        },
        "qdi_brackets": {
            "single": [
                {"rate": 0.00, "up_to": 48000},
                {"rate": 0.15, "up_to": 525000},
                {"rate": 0.20, "up_to": None},
            ],
            "married": [
                {"rate": 0.00, "up_to": 96000},
                {"rate": 0.15, "up_to": 590000},
                {"rate": 0.20, "up_to": None},
            ],
        },
    },
}

FALLBACK_IRMAA = {
    "2024": {
        "lookback_years": 2,
        "part_b": {
            "single": [
                {"up_to": 103000, "monthly": 174.70},
                {"up_to": 129000, "monthly": 244.60},
                {"up_to": 161000, "monthly": 349.40},
                {"up_to": 193000, "monthly": 454.20},
                {"up_to": 500000, "monthly": 559.00},
                {"up_to": None,   "monthly": 594.00},
            ],
            "married": [
                {"up_to": 206000, "monthly": 174.70},
                {"up_to": 258000, "monthly": 244.60},
                {"up_to": 322000, "monthly": 349.40},
                {"up_to": 386000, "monthly": 454.20},
                {"up_to": 750000, "monthly": 559.00},
                {"up_to": None,   "monthly": 594.00},
            ],
        },
        "part_d": {
            "single": [0.00, 12.90, 33.30, 53.80, 74.20, 81.00],
            "married": [0.00, 12.90, 33.30, 53.80, 74.20, 81.00],
        }
    },
    "2025": {
        # Placeholder; update when official values are known
        "lookback_years": 2,
        "part_b": {
            "single": [
                {"up_to": 106000, "monthly": 180.00},
                {"up_to": 132000, "monthly": 252.00},
                {"up_to": 165000, "monthly": 360.00},
                {"up_to": 198000, "monthly": 468.00},
                {"up_to": 510000, "monthly": 576.00},
                {"up_to": None,   "monthly": 612.00},
            ],
            "married": [
                {"up_to": 212000, "monthly": 180.00},
                {"up_to": 264000, "monthly": 252.00},
                {"up_to": 330000, "monthly": 360.00},
                {"up_to": 396000, "monthly": 468.00},
                {"up_to": 760000, "monthly": 576.00},
                {"up_to": None,   "monthly": 612.00},
            ],
        },
        "part_d": {
            "single": [0.00, 13.20, 34.00, 55.00, 75.80, 82.50],
            "married": [0.00, 13.20, 34.00, 55.00, 75.80, 82.50],
        }
    }
}

# Uniform Lifetime Table (abridged)
UNIFORM_LIFETIME_DIVISORS = {age: div for age, div in [
    (73, 26.5), (74, 25.5), (75, 24.6), (76, 23.7), (77, 22.9), (78, 22.0), (79, 21.1),
    (80, 20.2), (81, 19.4), (82, 18.5), (83, 17.7), (84, 16.8), (85, 16.0), (86, 15.2),
    (87, 14.4), (88, 13.7), (89, 12.9), (90, 12.2), (91, 11.5), (92, 10.8), (93, 10.1),
    (94, 9.5), (95, 8.9), (96, 8.4), (97, 7.8), (98, 7.3), (99, 6.8), (100, 6.4), (101, 6.0),
    (102, 5.6), (103, 5.2), (104, 4.9), (105, 4.6), (106, 4.3), (107, 4.1), (108, 3.9), (109, 3.7),
    (110, 3.5)
]}

# -----------------------------
# Helper functions
# -----------------------------

@st.cache_data(show_spinner=False)
def load_tax_table(path: str = "data/tax_brackets.json") -> dict:
    return load_json_safe(path, FALLBACK_TAX_BRACKETS)

@st.cache_data(show_spinner=False)
def load_irmaa_table(path: str = "data/irmaa.json") -> dict:
    return load_json_safe(path, FALLBACK_IRMAA)


def get_latest_year_key(table: dict, year: int) -> str:
    years = sorted(int(y) for y in table.keys())
    eligible = [y for y in years if y <= year]
    if eligible:
        return str(max(eligible))
    return str(max(years))  # if all are in the future, use earliest


def standard_deduction(year: int, filing: str, tax_table: dict) -> float:
    y = get_latest_year_key(tax_table, year)
    return tax_table[y]["standard_deduction"][filing]


def calc_tax_from_brackets(taxable_income: float, year: int, filing: str, tax_table: dict) -> float:
    if taxable_income <= 0:
        return 0.0
    y = get_latest_year_key(tax_table, year)
    brackets = tax_table[y]["brackets"][filing]
    tax = 0.0
    prev_cap = 0.0
    remaining = taxable_income
    for b in brackets:
        cap = b["up_to"] if b["up_to"] is not None else float('inf')
        span = min(remaining, cap - prev_cap)
        if span > 0:
            tax += span * b["rate"]
            remaining -= span
            prev_cap = cap
        if remaining <= 0:
            break
    return max(0.0, tax)


def calc_qdi_tax(qdi: float, year: int, filing: str, tax_table: dict, taxable_income_before_qdi: float) -> float:
    """Very simplified qualified dividend/LTCG tax layering using brackets from table.
    Assumes no other gains stacking complexities; for planning only.
    """
    if qdi <= 0:
        return 0.0
    y = get_latest_year_key(tax_table, year)
    brackets = tax_table[y].get("qdi_brackets", {}).get(filing)
    if not brackets:
        # Fallback: treat QDI as ordinary income if no special table
        return calc_tax_from_brackets(qdi, year, filing, tax_table)

    tax = 0.0
    base = taxable_income_before_qdi
    prev_cap = 0.0
    remaining = qdi
    for b in brackets:
        cap = b["up_to"] if b["up_to"] is not None else float('inf')
        # space available in this bracket after base stacking
        upper_after_base = max(0.0, cap - base) if cap != float('inf') else float('inf')
        take = min(remaining, upper_after_base)
        if take > 0:
            tax += take * b["rate"]
            remaining -= take
        base = max(base, cap)
        if remaining <= 0:
            break
    # anything left taxed at last rate
    if remaining > 0:
        last_rate = brackets[-1]["rate"]
        tax += remaining * last_rate
    return tax


def ss_benefit_by_age(annual_at_fra: float, start_age: int, fra: int = 67) -> float:
    """Approximate SSA early/late credits relative to FRA.
    - Early (monthly): 5/9 of 1% for first 36 months, then 5/12 of 1% beyond -> approx.
    - Delayed retirement credits: 8% per year to age 70.
    Returns annualized benefit at chosen start age.
    """
    if annual_at_fra <= 0:
        return 0.0
    if start_age == fra:
        return annual_at_fra
    if start_age < fra:
        months_early = (fra - start_age) * 12
        first36 = min(36, months_early)
        extra = max(0, months_early - 36)
        red = first36 * (5/9/100) + extra * (5/12/100)
        factor = max(0.0, 1 - red)
        return annual_at_fra * factor
    # delayed
    months_late = min((start_age - fra) * 12, (70 - fra) * 12)
    factor = 1 + months_late * (8/100/12)
    return annual_at_fra * factor


def taxable_ss_portion(ss_total: float, other_income: float, filing: str) -> float:
    """Compute taxable Social Security portion using the provisional income rules (0-85%).
    other_income is AGI without SS plus tax-exempt interest; simplified.
    """
    if ss_total <= 0:
        return 0.0
    if filing == "single":
        base1, base2 = 25000, 34000
    else:
        base1, base2 = 32000, 44000
    provisional = other_income + 0.5 * ss_total
    if provisional <= base1:
        return 0.0
    if provisional <= base2:
        return 0.5 * (provisional - base1)
    # Above base2
    amount = 0.85 * (provisional - base2) + min(0.5 * (base2 - base1), 0.85 * ss_total)
    return min(0.85 * ss_total, amount)


def rmd_divisor_for_age(age: int) -> Optional[float]:
    return UNIFORM_LIFETIME_DIVISORS.get(age)


def rmd_age_from_birth_year(birth_year: int) -> int:
    # SECURE 2.0: age 73 for those who attain age 72 after 2022 and age 75 for those born 1960+
    if birth_year >= 1960:
        return 75
    return 73


def irmaa_monthly_premiums(year: int, filing: str, magi: float, irmaa_table: dict) -> Tuple[float, float]:
    """Return (PartB_monthly_per_person, PartD_monthly_addon_per_person) for the given year and MAGI using 2-year lookback."""
    y_key = get_latest_year_key(irmaa_table, year)
    spec = irmaa_table[y_key]
    lookback = spec.get("lookback_years", 2)
    irmaa_year_key = get_latest_year_key(irmaa_table, year)  # using same year's table values (we're selecting brackets by lookback MAGI outside)
    part_b_tiers = irmaa_table[irmaa_year_key]["part_b"][filing]

    # Determine tier index by comparing MAGI (already lookback-adjusted by caller) to up_to thresholds
    tier_idx = 0
    for i, tier in enumerate(part_b_tiers):
        cap = tier["up_to"] if tier["up_to"] is not None else float('inf')
        if magi <= cap:
            tier_idx = i
            break
    part_b_monthly = part_b_tiers[tier_idx]["monthly"]

    # Part D flat addons by same tier index
    part_d_addons = irmaa_table[irmaa_year_key]["part_d"][filing]
    part_d_monthly = part_d_addons[min(tier_idx, len(part_d_addons)-1)]

    return float(part_b_monthly), float(part_d_monthly)

# -----------------------------
# Simulation
# -----------------------------

def simulate(
    start_year: int,
    end_year: int,
    filing: str,
    include_spouse: bool,
    # Ages
    age_self_now: int,
    retire_age_self: int,
    age_spouse_now: int,
    retire_age_spouse: int,
    # Current balances
    td_now: float,
    tx_now: float,
    roth_now: float,
    # Annual savings until retirement
    save_td: float,
    save_tx: float,
    save_roth: float,
    # SS
    ss_start_self: int,
    ss_fra_amount_self: float,
    ss_start_spouse: int,
    ss_fra_amount_spouse: float,
    # Roth conversions
    conv_start_year: Optional[int],
    conv_end_year: Optional[int],
    conv_amount_per_year: float,
    # Withdrawals
    withdraw_start_year: Optional[int],
    target_spending_real: float,
    withdraw_order: List[str],  # e.g., ["taxable", "tax_deferred", "roth"]
    taxes_from: str,            # e.g., "taxable" or "withhold"
    # Economics
    nominal_growth: float,
    inflation: float,
    # Data tables
    tax_table: dict,
    irmaa_table: dict,
):
    years = list(range(start_year, end_year + 1))

    # Initialize balances
    balances = {
        "tax_deferred": td_now,
        "taxable": tx_now,
        "roth": roth_now,
    }

    # Birth years (approx)
    birth_self = start_year - age_self_now
    birth_sp = start_year - age_spouse_now if include_spouse else None
    rmd_age_self = rmd_age_from_birth_year(birth_self)
    rmd_age_sp = rmd_age_from_birth_year(birth_sp) if include_spouse and birth_sp else None

    # Convenience
    def filing_status():
        return "married" if include_spouse and filing == "married" else "single"

    # SS annual by chosen ages (nominal at today's dollars; we'll inflate benefits)
    ss_self_at_start = ss_benefit_by_age(ss_fra_amount_self, ss_start_self)
    ss_sp_at_start = ss_benefit_by_age(ss_fra_amount_spouse, ss_start_spouse) if include_spouse else 0.0

    records = []
    last_known_tax_year = max(int(y) for y in tax_table.keys())
    last_known_irmaa_year = max(int(y) for y in irmaa_table.keys())

    # MAGI history for IRMAA lookback
    magi_history: Dict[int, float] = {}

    # Loop
    age_self = age_self_now
    age_sp = age_spouse_now if include_spouse else 0

    for year in years:
        # Inflate SS to the year using inflation (simple COLA approx)
        years_since_start = year - start_year
        infl_factor = (1 + inflation) ** years_since_start

        ss_income = 0.0
        if age_self >= ss_start_self:
            ss_income += ss_self_at_start * infl_factor
        if include_spouse and age_sp >= ss_start_spouse:
            ss_income += ss_sp_at_start * infl_factor

        # Growth pre-transaction
        for k in balances:
            balances[k] *= (1 + nominal_growth)

        # Contributions (until each person retires)
        if age_self < retire_age_self:
            balances["tax_deferred"] += save_td
            balances["taxable"] += save_tx
            balances["roth"] += save_roth
        if include_spouse and age_sp < retire_age_spouse:
            # For simplicity, treat spouse savings folded into same buckets
            balances["tax_deferred"] += save_td
            balances["taxable"] += save_tx
            balances["roth"] += save_roth

        # Required Minimum Distributions (from tax-deferred only)
        rmd = 0.0
        if age_self >= rmd_age_self:
            div = rmd_divisor_for_age(age_self)
            if div and balances["tax_deferred"] > 0:
                rmd += balances["tax_deferred"] / div
        if include_spouse and age_sp >= (rmd_age_sp or 200):
            div = rmd_divisor_for_age(age_sp)
            if div and balances["tax_deferred"] > 0:
                rmd += balances["tax_deferred"] / div
        rmd = min(rmd, balances["tax_deferred"]) if balances["tax_deferred"] > 0 else 0.0
        balances["tax_deferred"] -= rmd
        # RMD is taxable ordinary income and cash assumed to land in taxable account
        balances["taxable"] += rmd

        # Roth conversion within window (taxes from taxable)
        roth_conv = 0.0
        if conv_start_year and conv_end_year and conv_amount_per_year > 0:
            if conv_start_year <= year <= conv_end_year:
                roth_conv = min(conv_amount_per_year, balances["tax_deferred"])
                balances["tax_deferred"] -= roth_conv
                balances["roth"] += roth_conv
                # taxes paid later from taxable

        # Withdrawals for spending (real target expressed in today's dollars, inflate to nominal)
        target_nominal = target_spending_real * infl_factor if withdraw_start_year and year >= withdraw_start_year else 0.0

        withdrawals = {"taxable": 0.0, "tax_deferred": 0.0, "roth": 0.0}
        total_withdrawals = 0.0
        spend_needed = max(0.0, target_nominal)

        # Spend RMD first counts as available cash. Use it toward spend before additional withdrawals.
        spend_from_rmd = min(spend_needed, rmd)
        spend_needed -= spend_from_rmd
        total_withdrawals += spend_from_rmd

        # Additional withdrawals by order
        for bucket in withdraw_order:
            if spend_needed <= 0:
                break
            avail = balances[bucket]
            take = min(avail, spend_needed)
            balances[bucket] -= take
            # add cash to taxable for simplicity
            balances["taxable"] += take
            withdrawals[bucket] += take
            total_withdrawals += take
            spend_needed -= take

        # Income and Taxes
        filing_eff = filing_status()
        std_ded = standard_deduction(min(year, last_known_tax_year), filing_eff, tax_table)

        ordinary_income = rmd + roth_conv  # wages assumed 0 in retirement model
        qdi_income = 0.0  # user could extend to include dividends/cap gains
        other_agi = ordinary_income + qdi_income

        # Social Security taxation
        ss_taxable = taxable_ss_portion(ss_income, other_agi, filing_eff)
        agi = other_agi + ss_taxable

        # Taxable income after standard deduction
        taxable_income_before_qdi = max(0.0, agi - std_ded)
        ord_tax = calc_tax_from_brackets(taxable_income_before_qdi, min(year, last_known_tax_year), filing_eff, tax_table)
        qdi_tax = calc_qdi_tax(qdi_income, min(year, last_known_tax_year), filing_eff, tax_table, taxable_income_before_qdi)
        income_tax = ord_tax + qdi_tax

        # Pay conversion taxes from taxable (and any other taxes)
        tax_paid_from_taxable = min(balances["taxable"], income_tax)
        balances["taxable"] -= tax_paid_from_taxable
        tax_shortfall = income_tax - tax_paid_from_taxable
        if tax_shortfall > 0:
            # if not enough taxable, tap taxable next via forced withdrawals from order
            for bucket in withdraw_order:
                if balances[bucket] <= 0 or tax_shortfall <= 0:
                    continue
                take = min(balances[bucket], tax_shortfall)
                balances[bucket] -= take
                tax_shortfall -= take
            if tax_shortfall > 0:
                # as a fail-safe, reduce Roth (last resort)
                take = min(balances["roth"], tax_shortfall)
                balances["roth"] -= take
                tax_shortfall -= take

        # IRMAA using 2-year lookback MAGI (approx: MAGI = AGI + tax-exempt interest (0 here))
        magi = agi  # extend to include muni interest if desired
        magi_history[year] = magi
        lookback_year = year - 2
        lookback_magi = magi_history.get(lookback_year, magi)  # if missing, use current as best-effort

        part_b_monthly, part_d_monthly = irmaa_monthly_premiums(min(year, last_known_irmaa_year), filing_eff, lookback_magi, irmaa_table)
        # Multiply per-person if married filing jointly
        persons = 2 if filing_eff == "married" else 1
        irmaa_total = (part_b_monthly + part_d_monthly) * 12 * persons

        # Record row
        records.append({
            "Year": year,
            "Age Self": age_self,
            "Age Spouse": age_sp if include_spouse else None,
            "Balance Tax-Deferred": round(balances["tax_deferred"], 2),
            "Balance Taxable": round(balances["taxable"], 2),
            "Balance Roth": round(balances["roth"], 2),
            "RMD": round(rmd, 2),
            "Roth Conversion": round(roth_conv, 2),
            "Withdrawals (Taxable)": round(withdrawals["taxable"], 2),
            "Withdrawals (Tax-Deferred)": round(withdrawals["tax_deferred"], 2),
            "Withdrawals (Roth)": round(withdrawals["roth"], 2),
            "Total Withdrawals": round(total_withdrawals, 2),
            "SS Income (Gross)": round(ss_income, 2),
            "Taxable SS": round(ss_taxable, 2),
            "AGI": round(agi, 2),
            "Std Deduction": round(std_ded, 2),
            "Taxable Income": round(taxable_income_before_qdi, 2),
            "Income Tax": round(income_tax, 2),
            "IRMAA (B+D)": round(irmaa_total, 2),
            "Spending Target (Nominal)": round(target_nominal, 2),
            "Inflation Factor": round(infl_factor, 4),
        })

        # Age up
        age_self += 1
        if include_spouse:
            age_sp += 1

    df = pd.DataFrame(records)
    return df

# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="Retirement Planner • Roth • IRMAA • RMDs", layout="wide")
    st.title("Retirement Planner (Taxable • Tax-Deferred • Roth)")
    st.caption("Single-file app. Edit JSONs in /data to update tax/IRMAA tables. Future years use latest available values.")

    with st.sidebar:
        st.header("Scenario Setup")
        current_year = st.number_input("Current Year", value=2025, step=1)
        horizon_year = st.number_input("Plan Through Year", value=2060, step=1, min_value=current_year)
        filing_opt = st.selectbox("Filing Status", ["single", "married"], index=1)
        include_spouse = st.checkbox("Include spouse", value=(filing_opt == "married"))

        st.subheader("Ages & Retirement")
        age_self = st.number_input("Your Current Age", min_value=18, max_value=110, value=60)
        retire_age_self = st.number_input("Your Retirement Age", min_value=40, max_value=80, value=63)

        if include_spouse:
            age_spouse = st.number_input("Spouse Current Age", min_value=18, max_value=110, value=60)
            retire_age_spouse = st.number_input("Spouse Retirement Age", min_value=40, max_value=80, value=63)
        else:
            age_spouse = 0
            retire_age_spouse = 0

        st.subheader("Current Balances ($)")
        td_now = st.number_input("Tax-Deferred (Traditional 401k/IRA)", min_value=0.0, value=1500000.0, step=1000.0)
        tx_now = st.number_input("Taxable (Brokerage/Cash)", min_value=0.0, value=330000.0, step=1000.0)
        roth_now = st.number_input("Tax-Free (Roth)", min_value=0.0, value=1000000.0, step=1000.0)

        st.subheader("Annual Savings Until Retirement ($ per person)")
        save_td = st.number_input("To Tax-Deferred", min_value=0.0, value=0.0, step=1000.0)
        save_tx = st.number_input("To Taxable", min_value=0.0, value=0.0, step=1000.0)
        save_roth = st.number_input("To Roth", min_value=0.0, value=0.0, step=1000.0)

        st.subheader("Roth Conversions")
        conv_on = st.checkbox("Do Roth conversions?")
        if conv_on:
            conv_start = st.number_input("Conversion Start Year", value=current_year, step=1)
            conv_end = st.number_input("Conversion End Year", value=min(current_year+5, horizon_year), step=1)
            conv_amt = st.number_input("Convert Each Year ($)", min_value=0.0, value=100000.0, step=1000.0)
        else:
            conv_start, conv_end, conv_amt = None, None, 0.0

        st.subheader("Withdrawals & Spending")
        withdrawals_on = st.checkbox("Plan spending withdrawals?")
        if withdrawals_on:
            w_start = st.number_input("Withdrawals Start Year", value=current_year, step=1)
            spend_real = st.number_input("Target Annual Spending (today's $)", min_value=0.0, value=240000.0, step=1000.0)
            order = st.multiselect("Withdrawal Order (first to last)", ["taxable", "tax_deferred", "roth"], default=["taxable", "tax_deferred", "roth"])
            if len(order) == 0:
                order = ["taxable", "tax_deferred", "roth"]
            taxes_from = st.radio("Pay income taxes from:", ["taxable", "withhold"], index=0, help="This version pays from taxable regardless; 'withhold' not yet implemented.")
        else:
            w_start, spend_real, order, taxes_from = None, 0.0, ["taxable", "tax_deferred", "roth"], "taxable"

        st.subheader("Social Security (Annual at FRA=67)")
        ss_start_self = st.number_input("Your SS Start Age", min_value=62, max_value=70, value=70)
        ss_fra_self = st.number_input("Your Estimated Annual SS at FRA (67)", min_value=0.0, value=48000.0, step=1000.0)
        if include_spouse:
            ss_start_sp = st.number_input("Spouse SS Start Age", min_value=62, max_value=70, value=67)
            ss_fra_sp = st.number_input("Spouse Estimated Annual SS at FRA (67)", min_value=0.0, value=36000.0, step=1000.0)
        else:
            ss_start_sp, ss_fra_sp = 0, 0.0

        st.subheader("Economics")
        growth = st.number_input("Nominal Growth % (annual)", min_value=-50.0, max_value=50.0, value=5.0, step=0.1) / 100.0
        inflation = st.number_input("Inflation % (annual)", min_value=-5.0, max_value=20.0, value=2.5, step=0.1) / 100.0

        st.subheader("Data Files (optional)")
        tax_path = st.text_input("Tax Brackets JSON path", value="data/tax_brackets.json")
        irmaa_path = st.text_input("IRMAA JSON path", value="data/irmaa.json")

    # Load tables
    tax_table = load_tax_table(tax_path)
    irmaa_table = load_irmaa_table(irmaa_path)

    # Run simulation
    df = simulate(
        start_year=int(current_year),
        end_year=int(horizon_year),
        filing=filing_opt,
        include_spouse=include_spouse,
        age_self_now=int(age_self), retire_age_self=int(retire_age_self),
        age_spouse_now=int(age_spouse), retire_age_spouse=int(retire_age_spouse),
        td_now=float(td_now), tx_now=float(tx_now), roth_now=float(roth_now),
        save_td=float(save_td), save_tx=float(save_tx), save_roth=float(save_roth),
        ss_start_self=int(ss_start_self), ss_fra_amount_self=float(ss_fra_self),
        ss_start_spouse=int(ss_start_sp), ss_fra_amount_spouse=float(ss_fra_sp),
        conv_start_year=conv_start, conv_end_year=conv_end, conv_amount_per_year=float(conv_amt),
        withdraw_start_year=w_start, target_spending_real=float(spend_real),
        withdraw_order=order, taxes_from=taxes_from,
        nominal_growth=growth, inflation=inflation,
        tax_table=tax_table, irmaa_table=irmaa_table,
    )

    st.success("Simulation complete.")

    # Data Grid
    st.subheader("Year-by-Year Grid")
    st.dataframe(df, use_container_width=True)

    # Charts
    st.subheader("Charts")
    col1, col2 = st.columns(2)

    with col1:
        fig_bal = px.line(df, x="Year", y=["Balance Taxable", "Balance Tax-Deferred", "Balance Roth"],
                          title="Account Balances Over Time")
        st.plotly_chart(fig_bal, use_container_width=True)

    with col2:
        fig_tax = px.bar(df, x="Year", y=["Income Tax", "IRMAA (B+D)"], barmode="stack", title="Taxes & IRMAA by Year")
        st.plotly_chart(fig_tax, use_container_width=True)

    fig_income = px.bar(
        df,
        x="Year",
        y=["SS Income (Gross)", "RMD", "Roth Conversion", "Total Withdrawals"],
        barmode="stack",
        title="Income & Cash Flow Components",
    )
    st.plotly_chart(fig_income, use_container_width=True)

    # Download
    st.subheader("Export")
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="retirement_plan.csv", mime="text/csv")

    st.markdown("---")
    with st.expander("About the JSON formats"):
        st.markdown(
            """
            **tax_brackets.json** (example):
            ```json
            {
              "2025": {
                "standard_deduction": {"single": 15000, "married": 30000},
                "brackets": {
                  "single": [
                    {"rate": 0.10, "up_to": 12000},
                    {"rate": 0.12, "up_to": 48000},
                    {"rate": 0.22, "up_to": 102000},
                    {"rate": 0.24, "up_to": 195000},
                    {"rate": 0.32, "up_to": 247000},
                    {"rate": 0.35, "up_to": 620000},
                    {"rate": 0.37, "up_to": null}
                  ],
                  "married": [
                    {"rate": 0.10, "up_to": 24000},
                    {"rate": 0.12, "up_to": 96000},
                    {"rate": 0.22, "up_to": 204000},
                    {"rate": 0.24, "up_to": 390000},
                    {"rate": 0.32, "up_to": 494000},
                    {"rate": 0.35, "up_to": 740000},
                    {"rate": 0.37, "up_to": null}
                  ]
                },
                "qdi_brackets": {
                  "single": [
                    {"rate": 0.00, "up_to": 48000},
                    {"rate": 0.15, "up_to": 525000},
                    {"rate": 0.20, "up_to": null}
                  ],
                  "married": [
                    {"rate": 0.00, "up_to": 96000},
                    {"rate": 0.15, "up_to": 590000},
                    {"rate": 0.20, "up_to": null}
                  ]
                }
              }
            }
            ```

            **irmaa.json** (example):
            ```json
            {
              "2024": {
                "lookback_years": 2,
                "part_b": {
                  "single": [
                    {"up_to": 103000, "monthly": 174.70},
                    {"up_to": 129000, "monthly": 244.60},
                    {"up_to": 161000, "monthly": 349.40},
                    {"up_to": 193000, "monthly": 454.20},
                    {"up_to": 500000, "monthly": 559.00},
                    {"up_to": null,   "monthly": 594.00}
                  ],
                  "married": [
                    {"up_to": 206000, "monthly": 174.70},
                    {"up_to": 258000, "monthly": 244.60},
                    {"up_to": 322000, "monthly": 349.40},
                    {"up_to": 386000, "monthly": 454.20},
                    {"up_to": 750000, "monthly": 559.00},
                    {"up_to": null,   "monthly": 594.00}
                  ]
                },
                "part_d": {
                  "single": [0.00, 12.90, 33.30, 53.80, 74.20, 81.00],
                  "married": [0.00, 12.90, 33.30, 53.80, 74.20, 81.00]
                }
              }
            }
            ```

            You can add future years. If a simulated year has no entry, the app uses the latest available year.
            """
        )

    st.markdown(
        "**Notes & Limitations:** This is a planning model with simplifications (no state taxes, simplified SS taxation/QDI stacking, combined household buckets, same growth rate across accounts, approximate IRMAA). For advice, consult a professional.")


if __name__ == "__main__":
    main()
