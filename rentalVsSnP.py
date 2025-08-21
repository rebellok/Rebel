import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Rental Property vs S&P 500 Comparison",
    page_icon="🏠📈",
    layout="wide"
)


st.title("🏠📈 Rental Property vs S&P 500 Investment Comparison")


# Sidebar for navigation
st.sidebar.header("Investment Comparison Calculator")


# Initial Investment
st.header("Initial Investment Details")
col1, col2 = st.columns(2)


with col1:
    st.subheader("💰 Initial Capital")
    total_investment = st.number_input("Total Available Investment Amount ($)",
                                     min_value=1000, value=100000, step=1000)
    st.caption(f"Total Investment: ${total_investment:,.2f}")


with col2:
    st.subheader("⏰ Investment Period")
    investment_years = st.number_input("Investment Period (years)",
                                     min_value=1, max_value=50, value=10, step=1)


# Rental Property Section
st.header("🏠 Rental Property Details")


col1, col2 = st.columns(2)


with col1:
    st.subheader("Property Purchase")
    property_price = st.number_input("Property Purchase Price ($)",
                                   min_value=1000, value=300000, step=5000)
    st.caption(f"Property Price: ${property_price:,.2f}")
   
    down_payment_pct = st.slider("Down Payment (%)",
                                min_value=5, max_value=100, value=20, step=5)
    down_payment = property_price * (down_payment_pct / 100)
    st.caption(f"Down Payment: ${down_payment:,.2f}")
   
    closing_costs = st.number_input("Closing Costs ($)",
                                  min_value=0, value=5000, step=500)
    st.caption(f"Closing Costs: ${closing_costs:,.2f}")
   
    renovation_costs = st.number_input("Initial Renovation/Repair Costs ($)",
                                     min_value=0, value=10000, step=1000)
    st.caption(f"Renovation Costs: ${renovation_costs:,.2f}")


with col2:
    st.subheader("Financing")
    mortgage_amount = property_price - down_payment
    st.write(f"**Mortgage Amount**: ${mortgage_amount:,.2f}")
   
    interest_rate = st.number_input("Mortgage Interest Rate (%)",
                                  min_value=0.0, max_value=20.0, value=6.5, step=0.1) / 100
   
    loan_term = st.number_input("Loan Term (years)",
                              min_value=5, max_value=40, value=30, step=5)
   
    # Calculate monthly mortgage payment
    if mortgage_amount > 0:
        monthly_rate = interest_rate / 12
        num_payments = loan_term * 12
        monthly_payment = mortgage_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    else:
        monthly_payment = 0
   
    st.write(f"**Monthly Mortgage Payment**: ${monthly_payment:,.2f}")


# Rental Income and Expenses
st.subheader("💵 Rental Income & Operating Expenses")


col1, col2, col3 = st.columns(3)


with col1:
    st.markdown("**Income**")
    monthly_rent = st.number_input("Monthly Rent ($)",
                                 min_value=0, value=2500, step=50)
    st.caption(f"Monthly Rent: ${monthly_rent:,.2f}")
   
    annual_rent_increase = st.number_input("Annual Rent Increase (%)",
                                         min_value=0.0, max_value=20.0, value=3.0, step=0.5) / 100
   
    vacancy_rate = st.number_input("Vacancy Rate (%)",
                                 min_value=0.0, max_value=50.0, value=5.0, step=1.0) / 100


with col2:
    st.markdown("**Operating Expenses**")
    property_tax_annual = st.number_input("Annual Property Tax ($)",
                                        min_value=0, value=4000, step=100)
    st.caption(f"Property Tax: ${property_tax_annual:,.2f}")
   
    insurance_annual = st.number_input("Annual Insurance ($)",
                                     min_value=0, value=1200, step=50)
    st.caption(f"Insurance: ${insurance_annual:,.2f}")
   
    maintenance_annual = st.number_input("Annual Maintenance/Repairs ($)",
                                       min_value=0, value=3000, step=100)
    st.caption(f"Maintenance: ${maintenance_annual:,.2f}")
   
    management_fee_pct = st.number_input("Property Management Fee (%)",
                                       min_value=0.0, max_value=20.0, value=8.0, step=0.5) / 100


with col3:
    st.markdown("**Other Costs**")
    hoa_annual = st.number_input("Annual HOA Fees ($)",
                               min_value=0, value=0, step=100)
    st.caption(f"HOA Fees: ${hoa_annual:,.2f}")
   
    other_expenses_annual = st.number_input("Other Annual Expenses ($)",
                                          min_value=0, value=1000, step=100)
    st.caption(f"Other Expenses: ${other_expenses_annual:,.2f}")
   
    property_appreciation = st.number_input("Annual Property Appreciation (%)",
                                          min_value=-10.0, max_value=20.0, value=3.5, step=0.5) / 100


# S&P 500 Section
st.header("📈 S&P 500 Investment Details")


col1, col2 = st.columns(2)


with col1:
    sp500_return = st.number_input("Expected Annual S&P 500 Return (%)",
                                 min_value=0.0, max_value=30.0, value=10.0, step=0.5) / 100
   
    sp500_volatility = st.number_input("S&P 500 Volatility (Standard Deviation %)",
                                     min_value=0.0, max_value=50.0, value=15.0, step=1.0) / 100


with col2:
    expense_ratio = st.number_input("Index Fund Expense Ratio (%)",
                                  min_value=0.0, max_value=2.0, value=0.05, step=0.01) / 100
   
    dividend_yield = st.number_input("S&P 500 Dividend Yield (%)",
                                   min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100


# Tax Considerations
st.header("💸 Tax Considerations")


col1, col2 = st.columns(2)


with col1:
    st.subheader("Rental Property Taxes")
    income_tax_rate = st.number_input("Income Tax Rate (%)",
                                    min_value=0.0, max_value=50.0, value=25.0, step=1.0) / 100
   
    depreciation_years = st.number_input("Depreciation Period (years)",
                                       min_value=10, max_value=50, value=27, step=1)
   
    capital_gains_tax_rate = st.number_input("Capital Gains Tax Rate (%)",
                                           min_value=0.0, max_value=50.0, value=15.0, step=1.0) / 100


with col2:
    st.subheader("S&P 500 Taxes")
    dividend_tax_rate = st.number_input("Dividend Tax Rate (%)",
                                      min_value=0.0, max_value=50.0, value=15.0, step=1.0) / 100
   
    stock_capital_gains_rate = st.number_input("Stock Capital Gains Tax Rate (%)",
                                             min_value=0.0, max_value=50.0, value=15.0, step=1.0) / 100


# Calculate button
if st.button("🔄 Calculate Investment Comparison", type="primary"):
   
    # Initial costs for rental property
    total_initial_rental_cost = down_payment + closing_costs + renovation_costs
   
    # Check if user has enough capital for rental property
    if total_initial_rental_cost > total_investment:
        st.error(f"❌ Insufficient capital for rental property. Required: ${total_initial_rental_cost:,.2f}, Available: ${total_investment:,.2f}")
        st.stop()
   
    # Remaining cash for S&P 500 if doing rental
    remaining_cash_rental = total_investment - total_initial_rental_cost
   
    # Full S&P 500 investment amount
    sp500_investment_amount = total_investment
   
    # Calculate year-by-year projections
    years = []
    rental_values = []
    rental_cash_flows = []
    rental_total_returns = []
    sp500_values = []
    sp500_full_values = []
   
    # Annual depreciation
    annual_depreciation = property_price / depreciation_years
   
    for year in range(1, investment_years + 1):
        years.append(year)
       
        # Rental Property Calculations
        current_property_value = property_price * ((1 + property_appreciation) ** year)
        current_monthly_rent = monthly_rent * ((1 + annual_rent_increase) ** year)
       
        # Annual rental income (after vacancy)
        annual_rental_income = current_monthly_rent * 12 * (1 - vacancy_rate)
       
        # Annual expenses
        management_fee = annual_rental_income * management_fee_pct
        total_annual_expenses = (property_tax_annual + insurance_annual + maintenance_annual +
                               hoa_annual + other_expenses_annual + management_fee +
                               (monthly_payment * 12))
       
        # Net operating income
        net_cash_flow = annual_rental_income - total_annual_expenses
       
        # Tax benefits from depreciation
        tax_savings = annual_depreciation * income_tax_rate
        net_cash_flow_after_tax = net_cash_flow + tax_savings
       
        # Remaining mortgage balance
        remaining_balance = mortgage_amount
        if mortgage_amount > 0:
            for i in range(year * 12):
                interest_payment = remaining_balance * (interest_rate / 12)
                principal_payment = monthly_payment - interest_payment
                remaining_balance -= principal_payment
       
        # Equity in property
        property_equity = current_property_value - remaining_balance
       
        # Total return from rental (equity + accumulated cash flows)
        total_rental_return = property_equity - down_payment
       
        rental_values.append(current_property_value)
        rental_cash_flows.append(net_cash_flow_after_tax)
        rental_total_returns.append(total_rental_return)
       
        # S&P 500 Calculations (remaining cash from rental scenario)
        sp500_value_partial = remaining_cash_rental * ((1 + sp500_return - expense_ratio) ** year)
        sp500_values.append(sp500_value_partial)
       
        # S&P 500 Full Investment
        sp500_value_full = sp500_investment_amount * ((1 + sp500_return - expense_ratio) ** year)
        sp500_full_values.append(sp500_value_full)
   
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Year': years,
        'Rental Property Value': rental_values,
        'Rental Annual Cash Flow': rental_cash_flows,
        'Rental Total Return': rental_total_returns,
        'S&P 500 (Remaining Cash)': sp500_values,
        'S&P 500 (Full Investment)': sp500_full_values
    })
   
    # Calculate total returns
    final_year = investment_years
   
    # Rental Property Final Numbers
    final_property_value = rental_values[-1]
    final_mortgage_balance = mortgage_amount
    if mortgage_amount > 0:
        for i in range(final_year * 12):
            interest_payment = final_mortgage_balance * (interest_rate / 12)
            principal_payment = monthly_payment - interest_payment
            final_mortgage_balance -= principal_payment
   
    final_property_equity = final_property_value - max(0, final_mortgage_balance)
    total_cash_flows = sum(rental_cash_flows)
   
    # Calculate taxes on sale
    capital_gain = final_property_value - property_price
    capital_gains_tax = capital_gain * capital_gains_tax_rate
   
    rental_final_value = final_property_equity + total_cash_flows - capital_gains_tax
   
    # S&P 500 Final Numbers
    sp500_partial_final = sp500_values[-1]
    sp500_full_final = sp500_full_values[-1]
   
    # Calculate taxes on S&P 500 sale
    sp500_full_gain = sp500_full_final - sp500_investment_amount
    sp500_full_tax = sp500_full_gain * stock_capital_gains_rate
    sp500_full_after_tax = sp500_full_final - sp500_full_tax
   
    sp500_partial_gain = sp500_partial_final - remaining_cash_rental
    sp500_partial_tax = sp500_partial_gain * stock_capital_gains_rate
    sp500_partial_after_tax = sp500_partial_final - sp500_partial_tax
   
    # Display Results
    st.header("📊 Investment Comparison Results")
   
    col1, col2, col3 = st.columns(3)
   
    with col1:
        st.subheader("🏠 Rental Property")
        st.metric("Final Property Value", f"${final_property_value:,.2f}")
        st.metric("Total Cash Flows", f"${total_cash_flows:,.2f}")
        st.metric("Final Net Worth", f"${rental_final_value:,.2f}")
        st.metric("Total Return", f"${rental_final_value - total_initial_rental_cost:,.2f}")
        rental_roi = ((rental_final_value - total_initial_rental_cost) / total_initial_rental_cost) * 100
        st.metric("ROI", f"{rental_roi:.1f}%")
   
    with col2:
        st.subheader("📈 S&P 500 (Full)")
        st.metric("Final Value", f"${sp500_full_final:,.2f}")
        st.metric("After-Tax Value", f"${sp500_full_after_tax:,.2f}")
        st.metric("Total Return", f"${sp500_full_after_tax - sp500_investment_amount:,.2f}")
        sp500_full_roi = ((sp500_full_after_tax - sp500_investment_amount) / sp500_investment_amount) * 100
        st.metric("ROI", f"{sp500_full_roi:.1f}%")
   
    with col3:
        st.subheader("📈 S&P 500 (Partial)")
        st.write("*Remaining cash after rental down payment*")
        st.metric("Final Value", f"${sp500_partial_final:,.2f}")
        st.metric("After-Tax Value", f"${sp500_partial_after_tax:,.2f}")
        st.metric("Total Return", f"${sp500_partial_after_tax - remaining_cash_rental:,.2f}")
        if remaining_cash_rental > 0:
            sp500_partial_roi = ((sp500_partial_after_tax - remaining_cash_rental) / remaining_cash_rental) * 100
            st.metric("ROI", f"{sp500_partial_roi:.1f}%")
        else:
            st.metric("ROI", "N/A")
   
    # Combined scenario (Rental + S&P 500 partial)
    combined_final_value = rental_final_value + sp500_partial_after_tax
    combined_return = combined_final_value - total_investment
    combined_roi = (combined_return / total_investment) * 100
   
    st.subheader("🔗 Combined Strategy (Rental + S&P 500)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Combined Final Value", f"${combined_final_value:,.2f}")
    with col2:
        st.metric("Combined Return", f"${combined_return:,.2f}")
    with col3:
        st.metric("Combined ROI", f"{combined_roi:.1f}%")
   
    # Winner determination
    st.subheader("🏆 Investment Winner")
    scenarios = {
        "Rental Property Only": rental_final_value,
        "S&P 500 Full Investment": sp500_full_after_tax,
        "Combined (Rental + S&P 500)": combined_final_value
    }
   
    winner = max(scenarios, key=scenarios.get)
    winner_value = scenarios[winner]
   
    if winner == "Rental Property Only":
        st.success(f"🏠 **Winner: {winner}** with ${winner_value:,.2f}")
    elif winner == "S&P 500 Full Investment":
        st.success(f"📈 **Winner: {winner}** with ${winner_value:,.2f}")
    else:
        st.success(f"🔗 **Winner: {winner}** with ${winner_value:,.2f}")
   
    # Visualization
    st.subheader("📈 Investment Growth Comparison")
   
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Investment Value Over Time', 'Annual Cash Flow (Rental Property)'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
   
    # Investment values
    fig.add_trace(
        go.Scatter(x=years, y=rental_values, name="Rental Property Value", line=dict(color="green")),
        row=1, col=1
    )
   
    fig.add_trace(
        go.Scatter(x=years, y=sp500_full_values, name="S&P 500 (Full)", line=dict(color="blue")),
        row=1, col=1
    )
   
    fig.add_trace(
        go.Scatter(x=years, y=sp500_values, name="S&P 500 (Partial)", line=dict(color="lightblue")),
        row=1, col=1
    )
   
    # Cash flows
    colors = ['green' if cf >= 0 else 'red' for cf in rental_cash_flows]
    fig.add_trace(
        go.Bar(x=years, y=rental_cash_flows, name="Rental Cash Flow", marker_color=colors),
        row=2, col=1
    )
   
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Investment Comparison Analysis"
    )
   
    fig.update_xaxes(title_text="Years", row=2, col=1)
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Annual Cash Flow ($)", row=2, col=1)
   
    st.plotly_chart(fig, use_container_width=True)
   
    # Detailed breakdown table
    st.subheader("📋 Year-by-Year Breakdown")
   
    # Format the dataframe for display
    display_df = results_df.copy()
    for col in display_df.columns:
        if col != 'Year':
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
   
    st.dataframe(display_df, use_container_width=True)
   
    # Key assumptions and disclaimers
    st.subheader("ℹ️ Key Assumptions & Disclaimers")
    st.info("""
    **Key Assumptions:**
    - Property appreciates at constant rate
    - Rent increases at constant rate
    - S&P 500 returns are constant (actual returns vary significantly)
    - No major repairs or capital improvements beyond annual maintenance
    - No periods of extended vacancy beyond the vacancy rate
    - Interest rates remain constant
    - Tax rates remain constant
   
    **Important Notes:**
    - Real estate provides diversification and leverage benefits not captured in simple returns
    - S&P 500 is more liquid than real estate
    - Real estate requires active management and time investment
    - This analysis doesn't include transaction costs for selling investments
    - Results are for illustration purposes only and should not be considered financial advice
    """)
   
    # Export option
    st.subheader("💾 Export Results")
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name=f"rental_vs_sp500_comparison_{investment_years}years.csv",
        mime="text/csv"
    )


# Input Summary
st.sidebar.subheader("📋 Input Summary")
if st.sidebar.button("Show Input Summary"):
    st.sidebar.write(f"**Investment Amount:** ${total_investment:,.2f}")
    st.sidebar.write(f"**Property Price:** ${property_price:,.2f}")
    st.sidebar.write(f"**Down Payment:** ${down_payment:,.2f}")
    st.sidebar.write(f"**Monthly Rent:** ${monthly_rent:,.2f}")
    st.sidebar.write(f"**S&P 500 Return:** {sp500_return*100:.1f}%")
    st.sidebar.write(f"**Investment Period:** {investment_years} years")
