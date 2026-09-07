"""
Ledger — Account Summary Generator (pure Python / Streamlit version)

Run with:
    streamlit run app.py

Then open the local URL Streamlit prints (usually http://localhost:8501).
"""

import io
import re
from typing import Optional

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Ledger — Account Summary", page_icon="📒", layout="wide")

st.markdown(
    """
    <style>
        .stApp { background-color: #EEF1EE; }
        h1, h2, h3 { color: #16221D; }
        div[data-testid="stMetricValue"] { font-family: 'Courier New', monospace; }
        .block-container { padding-top: 2rem; max-width: 1100px; }
        .stButton>button {
            border-radius: 4px;
            font-weight: 600;
        }
        .account-count { color: #5B665F; font-size: 0.85rem; }
        .skipped-note {
            background: #FBF4E4;
            border: 1px solid #E3CE9C;
            color: #A8792F;
            padding: 10px 14px;
            border-radius: 4px;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

ACCOUNT_NAME_COL = "Account name"

DOLLAR_COLS = [
    "Current value",
    "Today's gain/loss dollar",
    "Total gain/loss dollar",
    "Cost basis total",
]

PERCENT_COLS = [
    "Today's gain/loss percent",
    "Total gain/loss percent",
    "Percent of account",
]

DISPLAY_COLS = [
    "Account name",
    "Number of holdings",
    "Current value",
    "Cost basis total",
    "Total gain/loss dollar",
    "Total gain/loss percent",
    "Today's gain/loss dollar",
    "Today's gain/loss percent",
]


# ----------------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------------

def clean_money(val) -> Optional[float]:
    """Turn '$1,234.56', '-$12.00', '+$5.10', '--', '' into a float (or None)."""
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"--", "-", "n/a", "nan"}:
        return None
    neg = s.startswith("-") or s.startswith("(")
    s = re.sub(r"[^0-9.]", "", s)
    if s == "":
        return None
    try:
        num = float(s)
    except ValueError:
        return None
    return -num if neg else num


def clean_percent(val) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"--", "-", "n/a", "nan"}:
        return None
    neg = s.startswith("-")
    s = re.sub(r"[^0-9.]", "", s)
    if s == "":
        return None
    try:
        num = float(s)
    except ValueError:
        return None
    return -num if neg else num


@st.cache_data(show_spinner=False)
def load_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    lower = filename.lower()
    if lower.endswith(".csv"):
        # index_col=False guards against a common brokerage-export quirk: every data row
        # ends with a trailing comma (an extra blank field) that the header row lacks.
        # Without this, pandas assumes the first column is an unnamed index and silently
        # shifts every other column over by one.
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", index_col=False)
    elif lower.endswith(".xlsx") or lower.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(content))
    else:
        raise ValueError("Unsupported file type. Upload a .csv, .xlsx, or .xls file.")

    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")

    if ACCOUNT_NAME_COL not in df.columns:
        raise ValueError(
            f"Required column '{ACCOUNT_NAME_COL}' not found. Found columns: {list(df.columns)}"
        )

    for col in DOLLAR_COLS:
        if col in df.columns:
            df[col + "_num"] = df[col].apply(clean_money)
    for col in PERCENT_COLS:
        if col in df.columns:
            df[col + "_num"] = df[col].apply(clean_percent)

    df[ACCOUNT_NAME_COL] = df[ACCOUNT_NAME_COL].astype(str).str.strip()
    df = df[df[ACCOUNT_NAME_COL].notna() & (df[ACCOUNT_NAME_COL] != "") & (df[ACCOUNT_NAME_COL] != "nan")]

    return df.reset_index(drop=True)


def compute_summary(df: pd.DataFrame, selected_accounts: list[str]):
    selected_set = set(selected_accounts)
    selected_df = df[df[ACCOUNT_NAME_COL].isin(selected_set)]

    summary_rows = []
    for account, group in selected_df.groupby(ACCOUNT_NAME_COL):
        row = {"Account name": account, "Number of holdings": int(len(group))}
        for col in DOLLAR_COLS:
            num_col = col + "_num"
            row[col] = round(float(group[num_col].fillna(0).sum()), 2) if num_col in group.columns else None

        cost = row.get("Cost basis total") or 0
        value = row.get("Current value") or 0
        row["Total gain/loss percent"] = round(((value - cost) / cost) * 100, 2) if cost else None

        if "Today's gain/loss percent_num" in group.columns and "Current value_num" in group.columns:
            weights = group["Current value_num"].fillna(0)
            pcts = group["Today's gain/loss percent_num"]
            valid = pcts.notna() & (weights > 0)
            if valid.any():
                row["Today's gain/loss percent"] = round(
                    (pcts[valid] * weights[valid]).sum() / weights[valid].sum(), 2
                )
            else:
                row["Today's gain/loss percent"] = None
        else:
            row["Today's gain/loss percent"] = None

        summary_rows.append(row)

    summary_rows.sort(key=lambda r: r["Account name"])

    grand_total = {"Account name": "GRAND TOTAL", "Number of holdings": sum(r["Number of holdings"] for r in summary_rows)}
    for col in DOLLAR_COLS:
        grand_total[col] = round(sum((r.get(col) or 0) for r in summary_rows), 2)
    gt_cost = grand_total.get("Cost basis total") or 0
    gt_value = grand_total.get("Current value") or 0
    grand_total["Total gain/loss percent"] = round(((gt_value - gt_cost) / gt_cost) * 100, 2) if gt_cost else None
    if grand_total.get("Current value"):
        weighted = sum((r.get("Today's gain/loss percent") or 0) * (r.get("Current value") or 0) for r in summary_rows)
        grand_total["Today's gain/loss percent"] = round(weighted / grand_total["Current value"], 2)
    else:
        grand_total["Today's gain/loss percent"] = None

    return pd.DataFrame(summary_rows), grand_total


def fmt_money(val):
    if val is None or pd.isna(val):
        return "—"
    sign = "-" if val < 0 else ""
    return f"{sign}${abs(val):,.2f}"


def fmt_percent(val):
    if val is None or pd.isna(val):
        return "—"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}%"


def style_summary_for_display(summary_df: pd.DataFrame, grand_total: dict) -> pd.DataFrame:
    display_df = summary_df.copy()
    display_df = pd.concat([display_df, pd.DataFrame([grand_total])], ignore_index=True)
    display_df = display_df[DISPLAY_COLS]

    formatted = display_df.copy()
    for col in ["Current value", "Cost basis total", "Total gain/loss dollar", "Today's gain/loss dollar"]:
        formatted[col] = display_df[col].apply(fmt_money)
    for col in ["Total gain/loss percent", "Today's gain/loss percent"]:
        formatted[col] = display_df[col].apply(fmt_percent)
    return formatted


def to_csv_bytes(summary_df: pd.DataFrame, grand_total: dict) -> bytes:
    out_df = pd.concat([summary_df[DISPLAY_COLS], pd.DataFrame([grand_total])[DISPLAY_COLS]], ignore_index=True)
    return out_df.to_csv(index=False).encode("utf-8")


def to_xlsx_bytes(summary_df: pd.DataFrame, grand_total: dict) -> bytes:
    out_df = pd.concat([summary_df[DISPLAY_COLS], pd.DataFrame([grand_total])[DISPLAY_COLS]], ignore_index=True)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="Account Summary")
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------

if "df" not in st.session_state:
    st.session_state.df = None
if "accounts" not in st.session_state:
    st.session_state.accounts = []
if "selected" not in st.session_state:
    st.session_state.selected = {}
if "summary_df" not in st.session_state:
    st.session_state.summary_df = None
if "grand_total" not in st.session_state:
    st.session_state.grand_total = None
if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.title("📒 Ledger — Account Summary")
st.caption("Upload a positions export, choose which accounts to include, and generate a summary by account.")

st.subheader("1 · Upload positions file")
uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"], label_visibility="collapsed")

if uploaded is not None:
    file_id = f"{uploaded.name}-{uploaded.size}"
    if file_id != st.session_state.last_file_id:
        try:
            content = uploaded.getvalue()
            df = load_dataframe(uploaded.name, content)
            st.session_state.df = df
            accounts = sorted(df[ACCOUNT_NAME_COL].unique().tolist())
            st.session_state.accounts = accounts
            st.session_state.selected = {a: True for a in accounts}  # all selected by default
            st.session_state.summary_df = None
            st.session_state.grand_total = None
            st.session_state.last_file_id = file_id
        except Exception as exc:
            st.error(f"Could not read that file: {exc}")
            st.session_state.df = None

if st.session_state.df is not None:
    df = st.session_state.df
    accounts = st.session_state.accounts
    row_counts = df.groupby(ACCOUNT_NAME_COL).size().to_dict()

    st.success(f"Loaded {len(df)} holdings across {len(accounts)} accounts.")

    st.subheader("2 · Choose accounts")

    col_a, col_b, col_c = st.columns([3, 1, 1])
    with col_b:
        if st.button("Select all", use_container_width=True):
            st.session_state.selected = {a: True for a in accounts}
    with col_c:
        if st.button("Deselect all", use_container_width=True):
            st.session_state.selected = {a: False for a in accounts}

    selected_count = sum(1 for v in st.session_state.selected.values() if v)
    with col_a:
        st.markdown(f"<span class='account-count'><b>{selected_count}</b> of <b>{len(accounts)}</b> accounts selected</span>", unsafe_allow_html=True)

    with st.container(border=True):
        n_cols = 2 if len(accounts) > 8 else 1
        cols = st.columns(n_cols)
        for i, account in enumerate(accounts):
            with cols[i % n_cols]:
                checked = st.checkbox(
                    f"{account}  ·  {row_counts.get(account, 0)} holding(s)",
                    value=st.session_state.selected.get(account, True),
                    key=f"chk_{account}",
                )
                st.session_state.selected[account] = checked

    selected_accounts = [a for a, v in st.session_state.selected.items() if v]

    st.write("")
    generate = st.button(
        "Generate summary",
        type="primary",
        disabled=len(selected_accounts) == 0,
    )

    if generate:
        summary_df, grand_total = compute_summary(df, selected_accounts)
        st.session_state.summary_df = summary_df
        st.session_state.grand_total = grand_total

    if st.session_state.summary_df is not None:
        st.subheader("3 · Summary")

        skipped = sorted(set(accounts) - set(selected_accounts))
        if skipped:
            st.markdown(
                f"<div class='skipped-note'>Excluded from this summary: {', '.join(skipped)}</div>",
                unsafe_allow_html=True,
            )
            st.write("")

        display_df = style_summary_for_display(st.session_state.summary_df, st.session_state.grand_total)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇ Download CSV",
                data=to_csv_bytes(st.session_state.summary_df, st.session_state.grand_total),
                file_name="account_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "⬇ Download Excel",
                data=to_xlsx_bytes(st.session_state.summary_df, st.session_state.grand_total),
                file_name="account_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
else:
    st.info("Upload a CSV or Excel positions file to get started.")
