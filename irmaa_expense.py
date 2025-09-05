# irmaa_expense.py
import streamlit as st
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data")

@st.cache_data
def load_year_data(year):
    path = DATA_DIR / f"irmaa_{year}.json"
    alt = DATA_DIR / f"irmaa_{year}_projected.json"
    if path.exists():
        return json.loads(path.read_text())
    if alt.exists():
        return json.loads(alt.read_text())
    # fallback: try any file with the year in the name
    for p in DATA_DIR.glob(f"*{year}*.json"):
        return json.loads(p.read_text())
    return None

def find_bracket(data, filing, magi):
    """Return the bracket dict for given filing & magi, or None."""
    if filing not in data["brackets"]:
        raise ValueError("Unknown filing status in JSON.")
    for b in data["brackets"][filing]:
        mn = 0 if b["min"] is None else b["min"]
        mx = float('inf') if b["max"] is None else b["max"]
        if mn <= magi <= mx:
            return b
    return None

def bracket_index_for_magi(brackets, magi):
    """Return the index of bracket matching magi, or None."""
    for i, b in enumerate(brackets):
        mn = 0 if b["min"] is None else b["min"]
        mx = float('inf') if b["max"] is None else b["max"]
        if mn <= magi <= mx:
            return i
    return None

# -------------------------
# App UI
# -------------------------
st.title("Medicare IRMAA calculator (Parts B & D)")

# find available JSON files (years)
years = []
for p in DATA_DIR.glob("irmaa_*.json"):
    try:
        j = json.loads(p.read_text())
        years.append(j["year"])
    except Exception:
        continue
years = sorted(list(set(years)))

if not years:
    st.error("No IRMAA JSON files found in ./data. Create files like data/irmaa_2025.json.")
    st.stop()

year = st.selectbox("Select year (IRMAA table year)", years, index=len(years)-1)
data = load_year_data(year)
st.caption(data.get("note", ""))

filing = st.radio("Filing status", ("single", "married"))

# IRMAA uses a 2-year lookback
income_year = year - 2

# Make the MAGI prompt explicit for married filers (joint MAGI)
magi_label_suffix = " (joint MAGI for couple)" if filing == "married" else ""
magi_input = st.number_input(
    f"Enter Modified Adjusted Gross Income (MAGI) for {income_year}{magi_label_suffix} ($)",
    min_value=0.0, value=80000.0, step=1000.0, format="%.2f"
)
magi = float(magi_input)

st.subheader("Scenario Setup")
magi_before = st.number_input(
    f"MAGI scenario A for {income_year}{magi_label_suffix} (e.g., before Roth conversion)",
    min_value=0.0, value=80000.0, step=1000.0, format="%.2f"
)
magi_after = st.number_input(
    f"MAGI scenario B for {income_year}{magi_label_suffix} (e.g., after Roth conversion)",
    min_value=0.0, value=150000.0, step=1000.0, format="%.2f"
)

# checkbox outside the button so toggling doesn't reset the app
if filing == "married":
    show_couple = st.checkbox("Show totals for couple (combined)", value=False)
else:
    show_couple = False

# compute button (stores results in session_state so displays can update independently)
if st.button("Compare Scenarios"):
    def compute_cost(magi_value):
        bracket = find_bracket(data, filing, magi_value)
        if bracket is None:
            return None
        standard_b = float(data["standard_part_b"])
        part_b_total = float(bracket["part_b_total"])
        part_b_extra = round(part_b_total - standard_b, 2)
        part_d_base = float(data["part_d_base"])
        part_d_extra = float(bracket["part_d_extra"])
        part_d_total = round(part_d_base + part_d_extra, 2)
        monthly_total = part_b_total + part_d_total
        return {
            "magi": magi_value,
            "part_b_standard": standard_b,
            "part_b_extra": part_b_extra,
            "part_b_total": part_b_total,
            "part_d_base": part_d_base,
            "part_d_extra": part_d_extra,
            "part_d_total": part_d_total,
            "monthly_total": monthly_total,
            "yearly_total": monthly_total * 12
        }

    result_a = compute_cost(magi_before)
    result_b = compute_cost(magi_after)

    if result_a and result_b:
        results_df = pd.DataFrame([result_a, result_b], index=["Scenario A", "Scenario B"])
        st.session_state["results_df"] = results_df
        # store the magis as well so bracket highlighting uses what was compared
        st.session_state["magi_before"] = float(magi_before)
        st.session_state["magi_after"] = float(magi_after)
    else:
        st.error("MAGI fell outside bracket definitions.")

# --- Display results if available ---
if "results_df" in st.session_state:
    df = st.session_state["results_df"]

    st.subheader("Monthly & Annual Totals")

    df_display = df.copy()
    # when showing couple totals, only the DISPLAY amounts are doubled;
    # bracket logic uses raw MAGI (which for 'married' should be joint MAGI).
    if filing == "married" and show_couple:
        df_display["monthly_total (couple)"] = df_display["monthly_total"] * 2
        df_display["yearly_total (couple)"] = df_display["yearly_total"] * 2
        st.caption("💡 Values shown are **per person** and combined totals for a couple (when selected).")
    else:
        if filing == "married":
            st.caption("💡 Values shown are **per person**. Check the box above to see couple totals.")
        else:
            st.caption("💡 Values shown are for the individual selected.")

    cols_to_show = ["magi", "part_b_total", "part_d_total", "monthly_total", "yearly_total"]
    if filing == "married" and show_couple:
        cols_to_show += ["monthly_total (couple)", "yearly_total (couple)"]

    st.dataframe(df_display[cols_to_show])

    # --- Comparison Metric (follows couple toggle) ---
    result_a = df.loc["Scenario A"]
    result_b = df.loc["Scenario B"]
    delta_yearly = result_b["yearly_total"] - result_a["yearly_total"]

    if filing == "married" and show_couple:
        label_text = "Difference in annual Medicare (Parts B + D) cost (couple combined)"
        value_text = f"${result_b['yearly_total']*2:,.2f} vs. ${result_a['yearly_total']*2:,.2f}"
        delta_text = f"${delta_yearly*2:,.2f}"
    else:
        label_text = "Difference in annual Medicare (Parts B + D) cost"
        if filing == "married":
            label_text += " (per person)"
        value_text = f"${result_b['yearly_total']:,.2f} vs. ${result_a['yearly_total']:,.2f}"
        delta_text = f"${delta_yearly:,.2f}"

    st.metric(label=label_text, value=value_text, delta=delta_text)

    # --- Bar chart (per person or couple combined) ---
    st.subheader("Visual Comparison")
    fig, ax = plt.subplots()
    if filing == "married" and show_couple:
        ax.bar(df.index, df["yearly_total"] * 2)
        ax.set_ylabel("Yearly Medicare Cost ($, couple)")
        ax.set_title("IRMAA Impact (Combined for Married Couple)")
    else:
        ax.bar(df.index, df["yearly_total"])
        ax.set_ylabel("Yearly Medicare Cost ($, per person)")
        ax.set_title("IRMAA Impact (Per Person)")
    st.pyplot(fig)

    # --- Bracket Table with correct highlighting ---
    brackets = data["brackets"][filing]
    rows = []
    # get magis used when user clicked Compare
    magi_before_used = st.session_state.get("magi_before", float(magi_before))
    magi_after_used = st.session_state.get("magi_after", float(magi_after))

    # find which bracket index each scenario falls into
    idx_before = bracket_index_for_magi(brackets, magi_before_used)
    idx_after = bracket_index_for_magi(brackets, magi_after_used)

    for i, b in enumerate(brackets):
        # use numeric values from JSON
        part_b_total = float(b["part_b_total"])
        part_b_extra = round(part_b_total - float(data["standard_part_b"]), 2)
        part_d_extra = float(b["part_d_extra"])

        # apply display-only doubling if user wants couple totals shown
        display_part_b_total = part_b_total * (2 if (filing == "married" and show_couple) else 1)
        display_part_b_extra = part_b_extra * (2 if (filing == "married" and show_couple) else 1)
        display_part_d_extra = part_d_extra * (2 if (filing == "married" and show_couple) else 1)

        # create income range label
        min_val = 0 if b["min"] is None else int(b["min"])
        max_val = None if b["max"] is None else int(b["max"])
        if max_val is None:
            income_range = f"${min_val:,}+"
        else:
            income_range = f"${min_val:,} – ${max_val:,}"

        rows.append({
            "Income Range": income_range,
            "Part B Total": f"${display_part_b_total:.2f}",
            "Part B IRMAA": f"${display_part_b_extra:.2f}",
            "Part D IRMAA": f"${display_part_d_extra:.2f}"
        })

    df_brackets = pd.DataFrame(rows)

    def highlight_rows(x):
        # x is the whole DataFrame here because axis=None
        style_df = pd.DataFrame('', index=x.index, columns=x.columns)
        # hex colors (safer/reliable in different renderers)
        if idx_before is not None:
            style_df.iloc[idx_before, :] = 'background-color: #fff2a8'  # pale yellow
        if idx_after is not None:
            style_df.iloc[idx_after, :] = 'background-color: #d4f4dd'  # pale green
        return style_df

    st.subheader(f"IRMAA Brackets for {year} ({filing.capitalize()})")
    st.caption("Yellow = Scenario A bracket; Green = Scenario B bracket.")
    # apply styling
    st.dataframe(df_brackets.style.apply(highlight_rows, axis=None), use_container_width=True)
