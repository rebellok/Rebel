import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import math


st.set_page_config(
    page_title="Mortgage Calculator",
    page_icon="🏠💰",
    layout="wide"
)


st.title("🏠💰 Comprehensive Mortgage Calculator")


# Sidebar for inputs
st.sidebar.header("Mortgage Details")


# Loan Information
st.sidebar.subheader("💵 Loan Information")
loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=1000, value=400000, step=1000)
st.sidebar.caption(f"Loan Amount: ${loan_amount:,.2f}")


interest_rate = st.sidebar.number_input("Annual Interest Rate (%)", min_value=0.1, max_value=30.0, value=6.5, step=0.1)
loan_term_years = st.sidebar.number_input("Loan Term (years)", min_value=1, max_value=50, value=30, step=1)


# Property Information
st.sidebar.subheader("🏡 Property Information")
home_price = st.sidebar.number_input("Home Price ($)", min_value=1000, value=500000, step=1000)
st.sidebar.caption(f"Home Price: ${home_price:,.2f}")


down_payment = st.sidebar.number_input("Down Payment ($)", min_value=0, value=100000, step=1000)
down_payment_pct = (down_payment / home_price) * 100 if home_price > 0 else 0
st.sidebar.caption(f"Down Payment: {down_payment_pct:.1f}% of home price")


# Additional Costs
st.sidebar.subheader("💸 Additional Monthly Costs")
property_tax_annual = st.sidebar.number_input("Annual Property Tax ($)", min_value=0, value=6000, step=100)
st.sidebar.caption(f"Property Tax: ${property_tax_annual:,.2f}")


home_insurance_annual = st.sidebar.number_input("Annual Home Insurance ($)", min_value=0, value=1200, step=50)
st.sidebar.caption(f"Home Insurance: ${home_insurance_annual:,.2f}")


pmi_rate = st.sidebar.number_input("PMI Rate (% annually)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
hoa_monthly = st.sidebar.number_input("Monthly HOA Fees ($)", min_value=0, value=0, step=25)


# PMI Logic
if down_payment_pct < 20:
    pmi_monthly = (loan_amount * (pmi_rate / 100)) / 12
    st.sidebar.warning(f"PMI Required: ${pmi_monthly:.2f}/month (Down payment < 20%)")
else:
    pmi_monthly = 0
    st.sidebar.success("No PMI Required (Down payment ≥ 20%)")


# Calculate mortgage payment
def calculate_monthly_payment(principal, annual_rate, years):
    """Calculate monthly mortgage payment using standard formula"""
    monthly_rate = annual_rate / 100 / 12
    num_payments = years * 12
   
    if monthly_rate == 0:
        return principal / num_payments
   
    monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    return monthly_payment


# Calculate amortization schedule
def create_amortization_schedule(principal, annual_rate, years):
    """Create detailed amortization schedule"""
    monthly_rate = annual_rate / 100 / 12
    num_payments = years * 12
    monthly_payment = calculate_monthly_payment(principal, annual_rate, years)
   
    schedule = []
    remaining_balance = principal
   
    for payment_num in range(1, num_payments + 1):
        if remaining_balance <= 0:
            break
           
        interest_payment = remaining_balance * monthly_rate
        principal_payment = min(monthly_payment - interest_payment, remaining_balance)
        remaining_balance -= principal_payment
       
        # Calculate date
        payment_date = datetime.now() + timedelta(days=30 * (payment_num - 1))
       
        schedule.append({
            'Payment_Number': payment_num,
            'Payment_Date': payment_date.strftime('%Y-%m-%d'),
            'Monthly_Payment': monthly_payment,
            'Principal_Payment': principal_payment,
            'Interest_Payment': interest_payment,
            'Remaining_Balance': max(0, remaining_balance),
            'Cumulative_Principal': principal - remaining_balance,
            'Cumulative_Interest': sum([s['Interest_Payment'] for s in schedule]) + interest_payment
        })
   
    return pd.DataFrame(schedule)


# Main calculations
monthly_payment = calculate_monthly_payment(loan_amount, interest_rate, loan_term_years)
property_tax_monthly = property_tax_annual / 12
insurance_monthly = home_insurance_annual / 12
total_monthly_payment = monthly_payment + property_tax_monthly + insurance_monthly + pmi_monthly + hoa_monthly


# Create amortization schedule
amortization_df = create_amortization_schedule(loan_amount, interest_rate, loan_term_years)


# Display main results
st.header("📊 Mortgage Summary")


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric("Monthly P&I Payment", f"${monthly_payment:,.2f}")
    st.metric("Total Monthly Payment", f"${total_monthly_payment:,.2f}")


with col2:
    if not amortization_df.empty:
        total_interest = amortization_df['Cumulative_Interest'].iloc[-1]
        st.metric("Total Interest Paid", f"${total_interest:,.2f}")
        st.metric("Total Cost of Loan", f"${loan_amount + total_interest:,.2f}")


with col3:
    loan_to_value = (loan_amount / home_price) * 100 if home_price > 0 else 0
    st.metric("Loan-to-Value Ratio", f"{loan_to_value:.1f}%")
    st.metric("Monthly Property Tax", f"${property_tax_monthly:,.2f}")


with col4:
    st.metric("Monthly Insurance", f"${insurance_monthly:,.2f}")
    if pmi_monthly > 0:
        st.metric("Monthly PMI", f"${pmi_monthly:,.2f}")
    else:
        st.metric("Monthly PMI", "$0.00")


# Monthly payment breakdown chart
st.header("📈 Monthly Payment Breakdown")


payment_breakdown = {
    'Principal & Interest': monthly_payment,
    'Property Tax': property_tax_monthly,
    'Home Insurance': insurance_monthly,
    'PMI': pmi_monthly,
    'HOA': hoa_monthly
}


# Remove zero values
payment_breakdown = {k: v for k, v in payment_breakdown.items() if v > 0}


fig_pie = go.Figure(data=[go.Pie(
    labels=list(payment_breakdown.keys()),
    values=list(payment_breakdown.values()),
    hole=0.3
)])
fig_pie.update_layout(
    title="Monthly Payment Components",
    annotations=[dict(text=f'Total<br>${total_monthly_payment:,.0f}', x=0.5, y=0.5, font_size=16, showarrow=False)]
)
st.plotly_chart(fig_pie, use_container_width=True)


# Amortization analysis
if not amortization_df.empty:
    st.header("📊 Loan Amortization Analysis")
   
    # Payment breakdown over time
    col1, col2 = st.columns(2)
   
    with col1:
        # Create yearly summary
        amortization_df['Year'] = ((amortization_df['Payment_Number'] - 1) // 12) + 1
        yearly_summary = amortization_df.groupby('Year').agg({
            'Principal_Payment': 'sum',
            'Interest_Payment': 'sum',
            'Remaining_Balance': 'last'
        }).reset_index()
       
        fig_stacked = go.Figure()
        fig_stacked.add_trace(go.Bar(
            x=yearly_summary['Year'],
            y=yearly_summary['Principal_Payment'],
            name='Principal',
            marker_color='lightblue'
        ))
        fig_stacked.add_trace(go.Bar(
            x=yearly_summary['Year'],
            y=yearly_summary['Interest_Payment'],
            name='Interest',
            marker_color='lightcoral'
        ))
       
        fig_stacked.update_layout(
            title='Annual Principal vs Interest Payments',
            xaxis_title='Year',
            yaxis_title='Amount ($)',
            barmode='stack'
        )
        st.plotly_chart(fig_stacked, use_container_width=True)
   
    with col2:
        # Remaining balance over time
        fig_balance = go.Figure()
        fig_balance.add_trace(go.Scatter(
            x=yearly_summary['Year'],
            y=yearly_summary['Remaining_Balance'],
            mode='lines+markers',
            name='Remaining Balance',
            line=dict(color='green', width=3),
            marker=dict(size=6)
        ))
       
        fig_balance.update_layout(
            title='Remaining Loan Balance Over Time',
            xaxis_title='Year',
            yaxis_title='Remaining Balance ($)',
            yaxis_tickformat='$,.0f'
        )
        st.plotly_chart(fig_balance, use_container_width=True)


# Detailed amortization table
st.header("📋 Amortization Schedule")


# Display options
display_option = st.selectbox("Display Schedule", ["First 12 Months", "Yearly Summary", "Every 12th Payment", "Full Schedule"])


if display_option == "First 12 Months":
    display_df = amortization_df.head(12)
elif display_option == "Yearly Summary":
    display_df = yearly_summary.copy()
    display_df.columns = ['Year', 'Principal_Payment', 'Interest_Payment', 'Remaining_Balance']
elif display_option == "Every 12th Payment":
    display_df = amortization_df[amortization_df['Payment_Number'] % 12 == 0]
else:
    display_df = amortization_df


# Format the display DataFrame
if 'Year' not in display_df.columns:
    format_df = display_df.copy()
    monetary_columns = ['Monthly_Payment', 'Principal_Payment', 'Interest_Payment', 'Remaining_Balance', 'Cumulative_Principal', 'Cumulative_Interest']
    for col in monetary_columns:
        if col in format_df.columns:
            format_df[col] = format_df[col].apply(lambda x: f"${x:,.2f}")
else:
    format_df = display_df.copy()
    monetary_columns = ['Principal_Payment', 'Interest_Payment', 'Remaining_Balance']
    for col in monetary_columns:
        if col in format_df.columns:
            format_df[col] = format_df[col].apply(lambda x: f"${x:,.2f}")


st.dataframe(format_df, use_container_width=True)


# Comparison scenarios
st.header("🔄 Scenario Comparison")


st.subheader("Compare Different Scenarios")


col1, col2, col3 = st.columns(3)


with col1:
    st.write("**Current Scenario**")
    st.write(f"Rate: {interest_rate}%")
    st.write(f"Term: {loan_term_years} years")
    st.write(f"Monthly P&I: ${monthly_payment:,.2f}")
    if not amortization_df.empty:
        st.write(f"Total Interest: ${total_interest:,.2f}")


with col2:
    st.write("**15-Year Loan**")
    monthly_15 = calculate_monthly_payment(loan_amount, interest_rate, 15)
    schedule_15 = create_amortization_schedule(loan_amount, interest_rate, 15)
    total_interest_15 = schedule_15['Cumulative_Interest'].iloc[-1] if not schedule_15.empty else 0
    st.write(f"Rate: {interest_rate}%")
    st.write(f"Term: 15 years")
    st.write(f"Monthly P&I: ${monthly_15:,.2f}")
    st.write(f"Total Interest: ${total_interest_15:,.2f}")
    if not amortization_df.empty and not schedule_15.empty:
        savings = total_interest - total_interest_15
        st.write(f"**Interest Savings: ${savings:,.2f}**")


with col3:
    st.write("**Rate + 1%**")
    higher_rate = interest_rate + 1
    monthly_higher = calculate_monthly_payment(loan_amount, higher_rate, loan_term_years)
    schedule_higher = create_amortization_schedule(loan_amount, higher_rate, loan_term_years)
    total_interest_higher = schedule_higher['Cumulative_Interest'].iloc[-1] if not schedule_higher.empty else 0
    st.write(f"Rate: {higher_rate}%")
    st.write(f"Term: {loan_term_years} years")
    st.write(f"Monthly P&I: ${monthly_higher:,.2f}")
    st.write(f"Total Interest: ${total_interest_higher:,.2f}")
    if not amortization_df.empty and not schedule_higher.empty:
        extra_cost = total_interest_higher - total_interest
        st.write(f"**Extra Cost: ${extra_cost:,.2f}**")


# Extra payment calculator
st.header("💰 Extra Payment Calculator")


extra_payment = st.number_input("Additional Monthly Payment ($)", min_value=0, value=0, step=50)


if extra_payment > 0:
    # Calculate with extra payments
    def calculate_with_extra_payments(principal, annual_rate, years, extra_monthly):
        monthly_rate = annual_rate / 100 / 12
        monthly_payment = calculate_monthly_payment(principal, annual_rate, years)
        total_monthly = monthly_payment + extra_monthly
       
        remaining_balance = principal
        payment_count = 0
        total_interest = 0
       
        while remaining_balance > 0 and payment_count < years * 12:
            payment_count += 1
            interest_payment = remaining_balance * monthly_rate
            principal_payment = min(total_monthly - interest_payment, remaining_balance)
           
            remaining_balance -= principal_payment
            total_interest += interest_payment
           
            if remaining_balance <= 0:
                break
       
        return payment_count, total_interest
   
    payments_with_extra, interest_with_extra = calculate_with_extra_payments(loan_amount, interest_rate, loan_term_years, extra_payment)
   
    col1, col2 = st.columns(2)
   
    with col1:
        st.metric("Loan Term with Extra Payment", f"{payments_with_extra} payments ({payments_with_extra/12:.1f} years)")
        months_saved = (loan_term_years * 12) - payments_with_extra
        st.metric("Time Saved", f"{months_saved} months ({months_saved/12:.1f} years)")
   
    with col2:
        st.metric("Total Interest with Extra Payment", f"${interest_with_extra:,.2f}")
        if not amortization_df.empty:
            interest_saved = total_interest - interest_with_extra
            st.metric("Interest Saved", f"${interest_saved:,.2f}")


# Affordability calculator
st.header("💡 Affordability Guidelines")


col1, col2 = st.columns(2)


with col1:
    monthly_income = st.number_input("Gross Monthly Income ($)", min_value=0, value=8000, step=100)
    monthly_debts = st.number_input("Other Monthly Debt Payments ($)", min_value=0, value=500, step=50)


with col2:
    if monthly_income > 0:
        # 28% rule for housing
        max_housing_payment = monthly_income * 0.28
        # 36% rule for total debt
        max_total_debt = monthly_income * 0.36
        max_mortgage_with_debt = max_total_debt - monthly_debts
       
        st.write("**Affordability Guidelines:**")
        st.write(f"Max housing payment (28% rule): ${max_housing_payment:,.2f}")
        st.write(f"Max total debt (36% rule): ${max_total_debt:,.2f}")
        st.write(f"Max mortgage with existing debts: ${max_mortgage_with_debt:,.2f}")
       
        if total_monthly_payment > max_housing_payment:
            st.error(f"⚠️ Total payment exceeds 28% guideline by ${total_monthly_payment - max_housing_payment:,.2f}")
        else:
            st.success(f"✅ Within 28% guideline (${max_housing_payment - total_monthly_payment:,.2f} cushion)")


# Export functionality
st.header("💾 Export Data")


if not amortization_df.empty:
    col1, col2 = st.columns(2)
   
    with col1:
        # CSV download
        csv_data = amortization_df.to_csv(index=False)
        st.download_button(
            label="Download Full Amortization Schedule (CSV)",
            data=csv_data,
            file_name=f"amortization_schedule_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
   
    with col2:
        # Summary download
        summary_data = {
            'Loan Amount': [f"${loan_amount:,.2f}"],
            'Interest Rate': [f"{interest_rate}%"],
            'Loan Term': [f"{loan_term_years} years"],
            'Monthly P&I Payment': [f"${monthly_payment:,.2f}"],
            'Total Monthly Payment': [f"${total_monthly_payment:,.2f}"],
            'Total Interest': [f"${total_interest:,.2f}"],
            'Total Cost': [f"${loan_amount + total_interest:,.2f}"]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_csv = summary_df.to_csv(index=False)
       
        st.download_button(
            label="Download Loan Summary (CSV)",
            data=summary_csv,
            file_name=f"loan_summary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


# Footer with disclaimers
st.markdown("---")
st.info("""
**Disclaimer:** This calculator provides estimates for educational purposes only.
Actual loan terms, rates, and payments may vary based on lender requirements, credit score,
down payment, and other factors. Consult with a qualified mortgage professional for accurate quotes.


**Key Features:**
- Complete amortization schedule calculation
- PMI calculation for down payments < 20%
- Extra payment scenarios
- Affordability guidelines (28/36 rule)
- Multiple display options and export functionality
""")

