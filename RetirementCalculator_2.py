import streamlit as st
import pandas as pd
import numpy as np
from datetime import date


st.set_page_config(
    page_title="Comprehensive Retirement Calculator",
    page_icon="👴👵",
    layout="wide"
)


st.title("👴👵 Comprehensive Retirement Calculator for Couples")


st.markdown("""
This calculator helps you project your retirement finances considering:
- Traditional, Roth, and Brokerage accounts
- Progressive tax rates
- Inflation and investment growth
- Roth conversion strategies
- Smart withdrawal sequencing
- Salary income during pre-retirement years
""")


# Define default values
current_year = date.today().year


# Create sidebar for inputs
st.sidebar.header("Personal Information")


col1, col2 = st.sidebar.columns(2)
with col1:
    person1_age = st.number_input("Person 1 Current Age", min_value=18, max_value=100, value=59)
    person1_retirement_age = st.number_input("Person 1 Retirement Age", min_value=person1_age, max_value=100, value=65)


with col2:
    person2_age = st.number_input("Person 2 Current Age", min_value=18, max_value=100, value=60)
    person2_retirement_age = st.number_input("Person 2 Retirement Age", min_value=person2_age, max_value=100, value=66)


# Calculate retirement start year
retirement_start_year = current_year + min(person1_retirement_age - person1_age, person2_retirement_age - person2_age)


st.sidebar.header("Salary Information")
enable_salary = st.sidebar.checkbox("Include Salary Income", value=True)


if enable_salary:
    annual_salary = st.sidebar.number_input("Annual Salary (Combined)", min_value=0, value=100000, step=5000)
    salary_increase_rate = st.sidebar.slider("Annual Salary Increase (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    salary_start_year = st.sidebar.number_input("Salary Start Year", min_value=current_year, value=current_year)
    salary_end_year = st.sidebar.number_input("Salary End Year", min_value=salary_start_year, value=retirement_start_year)
   
    st.sidebar.subheader("Salary Allocation")
    trad_contribution_pct = st.sidebar.slider("% to Traditional Accounts", min_value=0, max_value=100, value=10)
    roth_contribution_pct = st.sidebar.slider("% to Roth Accounts", min_value=0, max_value=100, value=5)
    brokerage_contribution_pct = st.sidebar.slider("% to Brokerage Account", min_value=0, max_value=100, value=5)
   
    # Calculate what's left for expenses
    remaining_pct = 100 - trad_contribution_pct - roth_contribution_pct - brokerage_contribution_pct
    st.sidebar.info(f"Remaining for expenses and taxes: {remaining_pct}%")
    st.sidebar.header("Current Savings")
traditional_savings = st.sidebar.number_input("Traditional IRA/401k Balance", min_value=0, value=250000, step=10000)
roth_savings = st.sidebar.number_input("Roth IRA/401k Balance", min_value=0, value=100000, step=10000)
brokerage_savings = st.sidebar.number_input("Brokerage Account Balance", min_value=0, value=40000, step=10000)


st.sidebar.header("Economic Assumptions")
inflation_rate = st.sidebar.slider("Annual Inflation Rate (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
traditional_growth_rate = st.sidebar.slider("Traditional Account Growth Rate (%)", min_value=0.0, max_value=15.0, value=6.0, step=0.1)
roth_growth_rate = st.sidebar.slider("Roth Account Growth Rate (%)", min_value=0.0, max_value=15.0, value=6.0, step=0.1)
brokerage_growth_rate = st.sidebar.slider("Brokerage Account Growth Rate (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)


st.sidebar.header("Retirement Planning")
retirement_end_year = st.sidebar.number_input("End of Retirement Planning Period",
                                             min_value=retirement_start_year,
                                             value=current_year + 35)


st.sidebar.header("Withdrawal Strategy")
annual_withdrawal_amount = st.sidebar.number_input("Annual Withdrawal Amount (Today's Dollars)",
                                                  min_value=0, value=60000, step=5000)
withdrawal_start_year = st.sidebar.number_input("Start Year for Withdrawals",
                                               min_value=current_year,
                                               value=retirement_start_year)
withdrawal_increase_with_inflation = st.sidebar.checkbox("Increase Withdrawals with Inflation", value=True)


st.sidebar.header("Roth Conversion Strategy")
roth_conversion_amount = st.sidebar.number_input("Annual Roth Conversion Amount",
                                                min_value=0, value=10000, step=1000)
roth_conversion_start_year = st.sidebar.number_input("Start Year for Roth Conversions",
                                                    min_value=current_year,
                                                    value=current_year)
roth_conversion_end_year = st.sidebar.number_input("End Year for Roth Conversions",
                                                  min_value=roth_conversion_start_year,
                                                  value=retirement_start_year)
# Define tax brackets (2023 MFJ)
def calculate_federal_tax(taxable_income):
    # 2023 Tax Brackets for Married Filing Jointly
    brackets = [
        (0, 22000, 0.10),
        (22000, 89450, 0.12),
        (89450, 190750, 0.22),
        (190750, 364200, 0.24),
        (364200, 462500, 0.32),
        (462500, 693750, 0.35),
        (693750, float('inf'), 0.37)
    ]
   
    # Standard deduction for MFJ in 2023
    standard_deduction = 27700
   
    # Apply standard deduction
    taxable_after_deduction = max(0, taxable_income - standard_deduction)
   
    tax = 0
    for lower, upper, rate in brackets:
        if taxable_after_deduction > lower:
            tax += (min(taxable_after_deduction, upper) - lower) * rate
    return tax


def calculate_ltcg_tax(ltcg_income):
    # 2023 Long Term Capital Gains Tax brackets for MFJ
    brackets = [
        (0, 89250, 0.00),
        (89250, 553850, 0.15),
        (553850, float('inf'), 0.20)
    ]
   
    tax = 0
    for lower, upper, rate in brackets:
        if ltcg_income > lower:
            tax += (min(ltcg_income, upper) - lower) * rate
    return tax


def calculate_retirement_projections():
    # Initialize data structure for projections
    years = list(range(current_year, retirement_end_year + 1))
    num_years = len(years)
   
    data = {
        "Year": years,
        "Age (Person 1)": [person1_age + (year - current_year) for year in years],
        "Age (Person 2)": [person2_age + (year - current_year) for year in years],
        "Salary Income": [0] * num_years,
        "Salary Tax": [0] * num_years,
        "Traditional Balance": [0] * num_years,
        "Roth Balance": [0] * num_years,
        "Brokerage Balance": [0] * num_years,
        "Roth Conversion": [0] * num_years,
        "Inflation-Adjusted Withdrawal": [0] * num_years,
        "Withdrawal from Traditional": [0] * num_years,
        "Withdrawal from Brokerage": [0] * num_years,
        "Withdrawal from Roth": [0] * num_years,
        "Taxes Paid": [0] * num_years,
        "Total Savings": [0] * num_years
    }
    # Initialize account balances
    data["Traditional Balance"][0] = traditional_savings
    data["Roth Balance"][0] = roth_savings
    data["Brokerage Balance"][0] = brokerage_savings
    data["Total Savings"][0] = traditional_savings + roth_savings + brokerage_savings
   
    # Process each year
    for i in range(1, num_years):
        year = years[i]
        prev_year_idx = i - 1
       
        # Copy previous balances as starting point
        trad_balance = data["Traditional Balance"][prev_year_idx]
        roth_balance = data["Roth Balance"][prev_year_idx]
        brokerage_balance = data["Brokerage Balance"][prev_year_idx]
       
        # Apply growth to each account
        trad_balance *= (1 + traditional_growth_rate / 100)
        roth_balance *= (1 + roth_growth_rate / 100)
        brokerage_balance *= (1 + brokerage_growth_rate / 100)
       
        # Process salary if enabled
        if enable_salary and salary_start_year <= year <= salary_end_year:
            # Calculate salary with annual increase
            years_of_increase = year - salary_start_year
            current_salary = annual_salary * (1 + salary_increase_rate / 100) ** years_of_increase
            data["Salary Income"][i] = current_salary
           
            # Calculate tax on salary
            # Assuming traditional contributions are tax-deductible
            taxable_salary = current_salary * (1 - trad_contribution_pct/100)
            salary_tax = calculate_federal_tax(taxable_salary)
            data["Salary Tax"][i] = salary_tax
           
            # Allocate salary to different accounts after tax
            trad_contribution = current_salary * (trad_contribution_pct / 100)
            roth_contribution = current_salary * (roth_contribution_pct / 100)
            brokerage_contribution = current_salary * (brokerage_contribution_pct / 100)
           
            # Add contributions to accounts
            trad_balance += trad_contribution
            roth_balance += roth_contribution
            brokerage_balance += (brokerage_contribution - salary_tax)  # Pay tax from brokerage contribution
           
            # Update taxes paid
            data["Taxes Paid"][i] += salary_tax
            # Calculate Roth conversions if applicable
        roth_conversion = 0
        if roth_conversion_start_year <= year <= roth_conversion_end_year and trad_balance > 0:
            roth_conversion = min(roth_conversion_amount, trad_balance)
            trad_balance -= roth_conversion
           
            # Pay taxes on Roth conversion from brokerage account
            conversion_tax = calculate_federal_tax(roth_conversion)
            brokerage_balance -= conversion_tax
            data["Taxes Paid"][i] += conversion_tax
           
            # Add converted amount to Roth
            roth_balance += roth_conversion
            data["Roth Conversion"][i] = roth_conversion
           
        # Calculate withdrawal amount for this year
        withdrawal_needed = 0
        if year >= withdrawal_start_year:
            # Apply inflation adjustment if selected
            if withdrawal_increase_with_inflation:
                inflation_factor = (1 + inflation_rate / 100) ** (year - current_year)
                withdrawal_needed = annual_withdrawal_amount * inflation_factor
            else:
                withdrawal_needed = annual_withdrawal_amount
               
            data["Inflation-Adjusted Withdrawal"][i] = withdrawal_needed
           
            # Implement withdrawal sequence: Traditional -> Brokerage -> Roth
            # 1. Withdraw from Traditional first
            trad_withdrawal = min(trad_balance, withdrawal_needed)
            trad_balance -= trad_withdrawal
            withdrawal_needed -= trad_withdrawal
            data["Withdrawal from Traditional"][i] = trad_withdrawal
           
            # Calculate tax on traditional withdrawal
            trad_tax = calculate_federal_tax(trad_withdrawal)
            brokerage_balance -= trad_tax  # Pay tax from brokerage
            data["Taxes Paid"][i] += trad_tax
           
            # 2. Withdraw from Brokerage next if needed
            if withdrawal_needed > 0:
                brokerage_withdrawal = min(brokerage_balance, withdrawal_needed)
               
                # Assume long term capital gains, with 50% of withdrawal being gains
                ltcg_amount = brokerage_withdrawal * 0.5  # Simplified assumption: 50% of withdrawal is capital gains
                ltcg_tax = calculate_ltcg_tax(ltcg_amount)
               
                brokerage_balance -= brokerage_withdrawal
                brokerage_balance -= ltcg_tax  # Pay LTCG tax from brokerage
                data["Taxes Paid"][i] += ltcg_tax
               
                withdrawal_needed -= brokerage_withdrawal
                data["Withdrawal from Brokerage"][i] = brokerage_withdrawal
                # 3. Withdraw from Roth last if still needed
            if withdrawal_needed > 0:
                roth_withdrawal = min(roth_balance, withdrawal_needed)
                roth_balance -= roth_withdrawal
                withdrawal_needed -= roth_withdrawal
                data["Withdrawal from Roth"][i] = roth_withdrawal
       
        # Update balances for this year
        data["Traditional Balance"][i] = trad_balance
        data["Roth Balance"][i] = roth_balance
        data["Brokerage Balance"][i] = brokerage_balance
        data["Total Savings"][i] = trad_balance + roth_balance + brokerage_balance
   
    return pd.DataFrame(data)


# Display the results
st.header("Retirement Projection")


# Calculate projections when user clicks the button
if st.button("Calculate Retirement Projection"):
    try:
        # Show loading spinner while calculating
        with st.spinner("Calculating your retirement projection..."):
            df = calculate_retirement_projections()
       
        # Highlight the first retirement year
        retirement_year_index = df[df["Year"] == retirement_start_year].index[0] if retirement_start_year in df["Year"].values else None
       
        # Format numbers as currency
        pd.options.display.float_format = "\${:,.0f}".format
       
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Detailed Projection", "Visual Charts", "Salary Impact"])
        with tab1:
            # Show full projection table
            float_cols = df.select_dtypes(include=['float', 'float64']).columns
            styled_df = df.style.format({col: "{:,.2f}" for col in float_cols})
            st.dataframe(
                styled_df,
                use_container_width=True
            )


            # Download option
            csv = df.to_csv(index=False, float_format="%.2f")
            st.download_button(
                "Download Projection as CSV",
                csv,
                "retirement_projection.csv",
                "text/csv",
                key="download-csv"
            )
           
        with tab2:
            st.subheader("Account Balances Over Time")
            chart_data = df[["Year", "Traditional Balance", "Roth Balance", "Brokerage Balance", "Total Savings"]]
            st.line_chart(chart_data.set_index("Year"))
           
            st.subheader("Annual Withdrawals")
            withdrawal_data = df[["Year", "Withdrawal from Traditional", "Withdrawal from Brokerage", "Withdrawal from Roth"]]
            st.line_chart(withdrawal_data.set_index("Year"))
           
            st.subheader("Taxes Paid")
            tax_data = df[["Year", "Taxes Paid"]]
            st.line_chart(tax_data.set_index("Year"))
           
        with tab3:
            if enable_salary:
                st.subheader("Salary Impact on Retirement")
                salary_data = df[["Year", "Salary Income", "Salary Tax"]]
                st.line_chart(salary_data.set_index("Year"))
               
                # Calculate total salary and contributions
                total_salary = df["Salary Income"].sum()
                total_salary_tax = df["Salary Tax"].sum()
               
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Salary Income", f"\${total_salary:,.2f}")
                with col2:
                    st.metric("Total Salary Tax", f"\${total_salary_tax:,.2f}")
                with col3:
                    effective_tax_rate = (total_salary_tax / total_salary * 100) if total_salary > 0 else 0
                    st.metric("Effective Tax Rate", f"{effective_tax_rate:.1f}%")
                   
                # Show contribution summary
                st.subheader("Contribution Summary")
               
                years_contributing = salary_end_year - salary_start_year + 1
                total_trad_contrib = total_salary * (trad_contribution_pct / 100)
                total_roth_contrib = total_salary * (roth_contribution_pct / 100)
                total_brokerage_contrib = total_salary * (brokerage_contribution_pct / 100)
               
                contrib_data = pd.DataFrame({
                    "Account Type": ["Traditional", "Roth", "Brokerage"],
                    "Total Contribution": [total_trad_contrib, total_roth_contrib, total_brokerage_contrib],
                    "Percentage": [trad_contribution_pct, roth_contribution_pct, brokerage_contribution_pct]
                })
               
                st.dataframe(contrib_data.style.format({
                    "Total Contribution": "\${:,.2f}",
                    "Percentage": "{:.1f}%"
                }), use_container_width=True)
            else:
                st.info("Salary component is disabled. Enable it in the sidebar to see salary impact.")


        # Key insights
        st.header("Key Insights")
       
        # Find out when money runs out (if it does)
        money_runs_out_idx = df[df["Total Savings"] <= 0].index.min()
        if pd.notna(money_runs_out_idx):
            money_runs_out_year = df.loc[money_runs_out_idx, "Year"]
            st.error(f"⚠️ Funds are projected to be depleted by {money_runs_out_year}")
        else:
            final_balance = df["Total Savings"].iloc[-1]
            st.success(f"✅ Your retirement savings are projected to last through {retirement_end_year} with \${final_balance:,.0f} remaining")
       
        # Total taxes paid in retirement
        total_tax = df["Taxes Paid"].sum()
        st.info(f"💰 Total estimated taxes over projection period: \${total_tax:,.0f}")
       
        # Total Roth conversions
        total_roth_conversion = df["Roth Conversion"].sum()
        st.info(f"🔄 Total Roth conversions: \${total_roth_conversion:,.0f}")
       
    except Exception as e:
        st.error(f"An error occurred during calculation: {e}")
else:
    st.info("Click the 'Calculate Retirement Projection' button to see results")


# Add some helpful explanations at the bottom
st.markdown("""
## About this Calculator


### Salary and Contributions
- Salary income can be modeled with annual increases
- Contributions to retirement accounts are calculated as percentages of salary
- Tax impact of salary and contributions is factored into projections


### Withdrawal Sequence
1. Traditional accounts are withdrawn first (taxed as ordinary income)
2. Brokerage accounts are withdrawn second (taxed at capital gains rates)
3. Roth accounts are withdrawn last (tax free)


### Tax Assumptions
- Federal taxes are calculated using 2023 tax brackets for Married Filing Jointly
- Traditional IRA/401k contributions reduce taxable income
- Standard deduction is applied to ordinary income
- Long-term capital gains are estimated assuming 50% of brokerage withdrawals are gains
- State and local taxes are not included


### Simplifications
- Required Minimum Distributions (RMDs) are not separately modeled
- Social Security benefits are not included (consider them part of your withdrawal amount)
- Tax brackets are not adjusted for inflation over time


Created with ❤️ using Streamlit
""")
###That's the complete code with the salary component added! The calculator now includes:


###Salary inputs with annual increases
###Allocation of salary to different account types
###Tax calculations on salary income
###Visualization of salary impact on retirement
###Updated insights with contribution summaries
###To run this enhanced retirement calculator:


###Save the entire code in a single Python file (e.g., retirement_calculator.py)
###Make sure you have the required packages installed: pip install streamlit pandas numpy
###Run it with: streamlit run retirement_calculator.py
###The calculator now provides a more comprehensive view of your retirement journey, from the working years through retirement, with detailed analysis of how your salary and contributions affect your long-term financial outlook.



