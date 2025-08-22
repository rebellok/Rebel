import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings


warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Australian Retirement Calculator",
    page_icon="🇦🇺💰",
    layout="wide"
)


st.title("🇦🇺💰 Comprehensive Australian Retirement Calculator")


# Sidebar for main settings
st.sidebar.header("Calculator Settings")


# Include spouse option
include_spouse = st.sidebar.checkbox("Include Spouse in Calculations", value=False)


# View mode
view_mode = st.sidebar.selectbox("View Mode", ["Nominal (Future Dollars)", "Real (Today's Purchasing Power)"])


# Inflation and growth assumptions
st.sidebar.subheader("Economic Assumptions")
inflation_rate = st.sidebar.number_input("Annual Inflation Rate (%)", min_value=0.0, max_value=20.0, value=2.5, step=0.1) / 100
investment_return = st.sidebar.number_input("Investment Return (%)", min_value=0.0, max_value=30.0, value=7.0, step=0.1) / 100
cola_rate = st.sidebar.number_input("Pension COLA Rate (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.1) / 100


# Main input sections
st.header("👤 Personal Information")


col1, col2 = st.columns(2)


with col1:
    st.subheader("Primary Person")
    age1 = st.number_input("Current Age", min_value=18, max_value=100, value=35, key="age1")
    retirement_age1 = st.number_input("Planned Retirement Age", min_value=50, max_value=85, value=67, key="ret_age1")
    life_expectancy1 = st.number_input("Life Expectancy", min_value=65, max_value=120, value=85, key="life_exp1")
   
    current_salary1 = st.number_input("Current Annual Salary ($)", min_value=0, value=80000, step=1000, key="salary1")
    st.caption(f"Current Salary: ${current_salary1:,.2f}")
   
    salary_growth1 = st.number_input("Annual Salary Growth (%)", min_value=0.0, max_value=20.0, value=3.0, step=0.1, key="growth1") / 100


with col2 if include_spouse else st.container():
    if include_spouse:
        st.subheader("Spouse")
        age2 = st.number_input("Spouse Current Age", min_value=18, max_value=100, value=33, key="age2")
        retirement_age2 = st.number_input("Spouse Planned Retirement Age", min_value=50, max_value=85, value=67, key="ret_age2")
        life_expectancy2 = st.number_input("Spouse Life Expectancy", min_value=65, max_value=120, value=87, key="life_exp2")
       
        current_salary2 = st.number_input("Spouse Current Annual Salary ($)", min_value=0, value=60000, step=1000, key="salary2")
        st.caption(f"Spouse Salary: ${current_salary2:,.2f}")
       
        salary_growth2 = st.number_input("Spouse Annual Salary Growth (%)", min_value=0.0, max_value=20.0, value=3.0, step=0.1, key="growth2") / 100
    else:
        age2 = 0
        retirement_age2 = 0
        life_expectancy2 = 0
        current_salary2 = 0
        salary_growth2 = 0


# Superannuation details
st.header("🏦 Superannuation Details")


col1, col2 = st.columns(2)


with col1:
    st.subheader("Primary Person Superannuation")
    super_balance1 = st.number_input("Current Super Balance ($)", min_value=0, value=150000, step=5000, key="super1")
    st.caption(f"Super Balance: ${super_balance1:,.2f}")
   
    super_contribution_rate1 = st.number_input("Super Contribution Rate (%)", min_value=9.5, max_value=50.0, value=11.0, step=0.5, key="super_rate1") / 100
   
    additional_super1 = st.number_input("Additional Annual Super Contributions ($)", min_value=0, value=5000, step=500, key="add_super1")
    st.caption(f"Additional Super: ${additional_super1:,.2f}")


with col2 if include_spouse else st.container():
    if include_spouse:
        st.subheader("Spouse Superannuation")
        super_balance2 = st.number_input("Spouse Current Super Balance ($)", min_value=0, value=100000, step=5000, key="super2")
        st.caption(f"Spouse Super Balance: ${super_balance2:,.2f}")
       
        super_contribution_rate2 = st.number_input("Spouse Super Contribution Rate (%)", min_value=9.5, max_value=50.0, value=11.0, step=0.5, key="super_rate2") / 100
       
        additional_super2 = st.number_input("Spouse Additional Annual Super Contributions ($)", min_value=0, value=3000, step=500, key="add_super2")
        st.caption(f"Spouse Additional Super: ${additional_super2:,.2f}")
    else:
        super_balance2 = 0
        super_contribution_rate2 = 0
        additional_super2 = 0


# Other investments
st.header("💼 Other Investments & Savings")


col1, col2 = st.columns(2)


with col1:
    other_savings = st.number_input("Current Non-Super Savings ($)", min_value=0, value=50000, step=1000)
    st.caption(f"Other Savings: ${other_savings:,.2f}")
   
    annual_savings = st.number_input("Annual Additional Savings ($)", min_value=0, value=10000, step=500)
    st.caption(f"Annual Savings: ${annual_savings:,.2f}")


with col2:
    property_value = st.number_input("Investment Property Value ($)", min_value=0, value=0, step=10000)
    st.caption(f"Property Value: ${property_value:,.2f}")
   
    property_growth = st.number_input("Property Growth Rate (%)", min_value=0.0, max_value=20.0, value=4.0, step=0.1) / 100


# Age Pension details
st.header("🏛️ Age Pension Information")


col1, col2 = st.columns(2)


with col1:
    st.subheader("Age Pension Rates (Current)")
    pension_single_max = st.number_input("Maximum Single Pension ($/fortnight)", min_value=0, value=1020, step=10)
    pension_couple_max = st.number_input("Maximum Couple Pension ($/fortnight each)", min_value=0, value=770, step=10)
   
    st.write(f"**Annual Maximum Pension:**")
    st.write(f"Single: ${pension_single_max * 26:,.2f}")
    st.write(f"Couple (each): ${pension_couple_max * 26:,.2f}")


with col2:
    st.subheader("Age Pension Thresholds")
    assets_threshold_single = st.number_input("Assets Threshold - Single ($)", min_value=0, value=301750, step=1000)
    assets_threshold_couple = st.number_input("Assets Threshold - Couple ($)", min_value=0, value=451500, step=1000)
   
    income_threshold_single = st.number_input("Income Threshold - Single ($/fortnight)", min_value=0, value=190, step=5)
    income_threshold_couple = st.number_input("Income Threshold - Couple ($/fortnight each)", min_value=0, value=336, step=5)


# Retirement expenses
st.header("💰 Retirement Expenses")


col1, col2 = st.columns(2)


with col1:
    annual_expenses_today = st.number_input("Desired Annual Retirement Expenses (today's dollars)", min_value=0, value=70000, step=1000)
    st.caption(f"Annual Expenses: ${annual_expenses_today:,.2f}")
   
    expenses_replacement_ratio = st.number_input("Expenses as % of Pre-Retirement Income", min_value=20, max_value=150, value=80, step=5) / 100


with col2:
    own_home = st.checkbox("Will own home outright in retirement", value=True)
    healthcare_annual = st.number_input("Additional Annual Healthcare Costs ($)", min_value=0, value=5000, step=500)
    st.caption(f"Healthcare Costs: ${healthcare_annual:,.2f}")


# Calculation functions
def calculate_age_pension(assets, income, is_couple=False):
    """Calculate Age Pension based on assets and income test"""
    if is_couple:
        max_pension = pension_couple_max * 26
        assets_threshold = assets_threshold_couple
        income_threshold = income_threshold_couple * 26
    else:
        max_pension = pension_single_max * 26
        assets_threshold = assets_threshold_single
        income_threshold = income_threshold_single * 26
   
    # Assets test - reduces pension by $3 per fortnight for every $1000 over threshold
    assets_reduction = max(0, (assets - assets_threshold) / 1000 * 3 * 26)
    pension_after_assets = max(0, max_pension - assets_reduction)
   
    # Income test - reduces pension by 50c for every dollar over threshold
    income_reduction = max(0, (income - income_threshold) * 0.5)
    pension_after_income = max(0, max_pension - income_reduction)
   
    # Take the lower of the two
    final_pension = min(pension_after_assets, pension_after_income)
   
    return final_pension


def simulate_retirement():
    """Main simulation function"""
    max_age = max(life_expectancy1, life_expectancy2 if include_spouse else 0)
    current_year = datetime.now().year
   
    results = []
   
    # Initialize balances
    super1_balance = super_balance1
    super2_balance = super_balance2
    other_balance = other_savings
    property_bal = property_value
   
    for year_offset in range(max_age - min(age1, age2 if include_spouse else age1) + 1):
        current_age1 = age1 + year_offset
        current_age2 = age2 + year_offset if include_spouse else 0
        year = current_year + year_offset
       
        # Check if people are still alive
        person1_alive = current_age1 <= life_expectancy1
        person2_alive = current_age2 <= life_expectancy2 if include_spouse else False
       
        if not person1_alive and not person2_alive:
            break
       
        # Calculate salaries (if still working)
        person1_working = current_age1 < retirement_age1 and person1_alive
        person2_working = current_age2 < retirement_age2 and person2_alive if include_spouse else False
       
        salary1_current = current_salary1 * ((1 + salary_growth1) ** year_offset) if person1_working else 0
        salary2_current = current_salary2 * ((1 + salary_growth2) ** year_offset) if person2_working else 0
       
        # Super contributions
        if person1_working:
            super_contrib1 = salary1_current * super_contribution_rate1 + additional_super1
            super1_balance += super_contrib1
       
        if person2_working:
            super_contrib2 = salary2_current * super_contribution_rate2 + additional_super2
            super2_balance += super_contrib2
       
        # Additional savings (only while working)
        if person1_working or person2_working:
            other_balance += annual_savings
       
        # Investment growth
        super1_balance *= (1 + investment_return)
        super2_balance *= (1 + investment_return)
        other_balance *= (1 + investment_return)
        property_bal *= (1 + property_growth)
       
        # Calculate retirement income and expenses
        person1_retired = current_age1 >= retirement_age1 and person1_alive
        person2_retired = current_age2 >= retirement_age2 and person2_alive if include_spouse else False
       
        super_income = 0
        other_investment_income = 0
       
        if person1_retired or person2_retired:
            # Calculate expenses (adjusted for inflation)
            annual_expenses = annual_expenses_today * ((1 + inflation_rate) ** year_offset)
            if not own_home:
                annual_expenses += 20000 * ((1 + inflation_rate) ** year_offset)  # Estimated rent
            annual_expenses += healthcare_annual * ((1 + inflation_rate) ** year_offset)
           
            # Determine if couple for pension calculation
            is_couple_for_pension = include_spouse and person1_alive and person2_alive
           
            # Calculate total assets for pension test
            total_assets = super1_balance + super2_balance + other_balance + property_bal
           
            # Estimate investment income for income test
            estimated_investment_income = (other_balance + property_bal) * 0.04  # Assume 4% income yield
           
            # Calculate Age Pension
            age_pension_annual = 0
            if (person1_retired and current_age1 >= 67) or (person2_retired and current_age2 >= 67):
                age_pension_annual = calculate_age_pension(total_assets, estimated_investment_income, is_couple_for_pension)
                if is_couple_for_pension and person1_retired and person2_retired:
                    age_pension_annual *= 2  # Both receive pension
           
            # Calculate required withdrawal from super and other investments
            required_income = max(0, annual_expenses - age_pension_annual)
           
            # Withdraw from investments (4% rule approximation)
            total_investment_balance = super1_balance + super2_balance + other_balance
            if total_investment_balance > 0:
                withdrawal_rate = min(0.05, required_income / total_investment_balance)  # Max 5% withdrawal
                super_withdrawal = (super1_balance + super2_balance) * withdrawal_rate
                other_withdrawal = other_balance * withdrawal_rate
               
                super1_balance -= super_withdrawal * (super1_balance / (super1_balance + super2_balance)) if super1_balance + super2_balance > 0 else 0
                super2_balance -= super_withdrawal * (super2_balance / (super1_balance + super2_balance)) if super1_balance + super2_balance > 0 else 0
                other_balance -= other_withdrawal
               
                super_income = super_withdrawal
                other_investment_income = other_withdrawal
        else:
            annual_expenses = 0
            age_pension_annual = 0
       
        # Store results
        results.append({
            'Year': year,
            'Age1': current_age1 if person1_alive else 0,
            'Age2': current_age2 if person2_alive else 0,
            'Person1_Alive': person1_alive,
            'Person2_Alive': person2_alive,
            'Salary1': salary1_current,
            'Salary2': salary2_current,
            'Super1_Balance': super1_balance,
            'Super2_Balance': super2_balance,
            'Other_Investments': other_balance,
            'Property_Value': property_bal,
            'Total_Assets': super1_balance + super2_balance + other_balance + property_bal,
            'Annual_Expenses': annual_expenses if person1_retired or person2_retired else 0,
            'Age_Pension': age_pension_annual,
            'Super_Income': super_income,
            'Other_Income': other_investment_income,
            'Total_Income': age_pension_annual + super_income + other_investment_income,
            'Net_Cash_Flow': (age_pension_annual + super_income + other_investment_income) - (annual_expenses if person1_retired or person2_retired else 0)
        })
   
    return pd.DataFrame(results)


# Calculate projections
if st.button("🔄 Calculate Retirement Projection", type="primary"):
   
    st.info("Calculating retirement projections...")
   
    # Run simulation
    df = simulate_retirement()
   
    if df.empty:
        st.error("Error in calculations. Please check your inputs.")
        st.stop()
   
    # Apply real vs nominal view
    if view_mode == "Real (Today's Purchasing Power)":
        monetary_columns = ['Salary1', 'Salary2', 'Super1_Balance', 'Super2_Balance', 'Other_Investments',
                          'Property_Value', 'Total_Assets', 'Annual_Expenses', 'Age_Pension', 'Super_Income',
                          'Other_Income', 'Total_Income', 'Net_Cash_Flow']
       
        base_year = df.iloc[0]['Year']
        for col in monetary_columns:
            if col in df.columns:
                years_from_base = df['Year'] - base_year
                df[col] = df[col] / ((1 + inflation_rate) ** years_from_base)
   
    st.success("Projection completed!")
   
    # Display key metrics
    st.header("📊 Key Retirement Metrics")
   
    retirement_start_row = df[(df['Age1'] >= retirement_age1) | (df['Age2'] >= retirement_age2)].head(1)
   
    if not retirement_start_row.empty:
        retirement_assets = retirement_start_row['Total_Assets'].iloc[0]
        col1, col2, col3, col4 = st.columns(4)
       
        with col1:
            st.metric("Assets at Retirement", f"${retirement_assets:,.0f}")
       
        with col2:
            final_assets = df['Total_Assets'].iloc[-1]
            st.metric("Final Assets", f"${final_assets:,.0f}")
       
        with col3:
            avg_pension = df[df['Age_Pension'] > 0]['Age_Pension'].mean()
            if not pd.isna(avg_pension):
                st.metric("Average Age Pension", f"${avg_pension:,.0f}")
            else:
                st.metric("Average Age Pension", "$0")
       
        with col4:
            years_of_pension = len(df[df['Age_Pension'] > 0])
            st.metric("Years Receiving Pension", f"{years_of_pension}")
   
    # Charts
    st.header("📈 Retirement Projections")
   
    # Asset growth chart
    fig_assets = go.Figure()
   
    fig_assets.add_trace(go.Scatter(x=df['Year'], y=df['Super1_Balance'], name='Person 1 Super', stackgroup='one'))
    if include_spouse:
        fig_assets.add_trace(go.Scatter(x=df['Year'], y=df['Super2_Balance'], name='Person 2 Super', stackgroup='one'))
    fig_assets.add_trace(go.Scatter(x=df['Year'], y=df['Other_Investments'], name='Other Investments', stackgroup='one'))
    if property_value > 0:
        fig_assets.add_trace(go.Scatter(x=df['Year'], y=df['Property_Value'], name='Property', stackgroup='one'))
   
    fig_assets.update_layout(
        title=f'Asset Growth Over Time ({view_mode})',
        xaxis_title='Year',
        yaxis_title='Value ($)',
        yaxis_tickformat='$,.0f'
    )
   
    st.plotly_chart(fig_assets, use_container_width=True)
   
    # Income vs Expenses in retirement
    retirement_df = df[(df['Age1'] >= retirement_age1) | (df['Age2'] >= retirement_age2)]
   
    if not retirement_df.empty:
        fig_income = go.Figure()
       
        fig_income.add_trace(go.Bar(x=retirement_df['Year'], y=retirement_df['Age_Pension'], name='Age Pension'))
        fig_income.add_trace(go.Bar(x=retirement_df['Year'], y=retirement_df['Super_Income'], name='Super Withdrawals'))
        fig_income.add_trace(go.Bar(x=retirement_df['Year'], y=retirement_df['Other_Income'], name='Other Investment Income'))
        fig_income.add_trace(go.Scatter(x=retirement_df['Year'], y=retirement_df['Annual_Expenses'],
                                      mode='lines', name='Annual Expenses', line=dict(color='red', width=3)))
       
        fig_income.update_layout(
            title=f'Retirement Income vs Expenses ({view_mode})',
            xaxis_title='Year',
            yaxis_title='Amount ($)',
            yaxis_tickformat='$,.0f',
            barmode='stack'
        )
       
        st.plotly_chart(fig_income, use_container_width=True)
   
    # Detailed year-by-year results
    st.header("📋 Year-by-Year Projections")
   
    # Format for display
    display_df = df.copy()
    monetary_cols = ['Salary1', 'Salary2', 'Super1_Balance', 'Super2_Balance', 'Other_Investments',
                    'Property_Value', 'Total_Assets', 'Annual_Expenses', 'Age_Pension', 'Total_Income']
   
    for col in monetary_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}" if x > 0 else "$0")
   
    # Select columns for display
    if include_spouse:
        display_columns = ['Year', 'Age1', 'Age2', 'Total_Assets', 'Age_Pension', 'Total_Income', 'Annual_Expenses']
        column_names = ['Year', 'Age 1', 'Age 2', 'Total Assets', 'Age Pension', 'Total Income', 'Annual Expenses']
    else:
        display_columns = ['Year', 'Age1', 'Total_Assets', 'Age_Pension', 'Total_Income', 'Annual_Expenses']
        column_names = ['Year', 'Age', 'Total Assets', 'Age Pension', 'Total Income', 'Annual Expenses']
   
    display_df_subset = display_df[display_columns]
    display_df_subset.columns = column_names
   
    st.dataframe(display_df_subset, use_container_width=True)
   
    # Age Pension analysis
    st.header("🏛️ Age Pension Analysis")
   
    pension_years = df[df['Age_Pension'] > 0]
    if not pension_years.empty:
        col1, col2 = st.columns(2)
       
        with col1:
            total_pension = pension_years['Age_Pension'].sum()
            st.metric("Total Age Pension Received", f"${total_pension:,.0f}")
           
            first_pension_year = pension_years['Year'].min()
            st.metric("First Year Receiving Pension", f"{first_pension_year}")
       
        with col2:
            avg_annual_pension = pension_years['Age_Pension'].mean()
            st.metric("Average Annual Pension", f"${avg_annual_pension:,.0f}")
           
            pension_duration = len(pension_years)
            st.metric("Years Receiving Pension", f"{pension_duration}")
    else:
        st.info("Based on your projections, you may not be eligible for the Age Pension due to the assets and income tests.")
   
    # Export functionality
    st.header("💾 Export Results")
   
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="Download Full Projection (CSV)",
        data=csv_data,
        file_name=f"australian_retirement_projection_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


# Information section
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About This Calculator")
st.sidebar.info("""
This Australian retirement calculator includes:


**Superannuation:**
- Compulsory super contributions
- Additional voluntary contributions
- Investment growth projections


**Age Pension:**
- Assets and income testing
- Current pension rates and thresholds
- COLA adjustments


**Other Features:**
- Spouse calculations
- Investment properties
- Real vs nominal projections
- Comprehensive expense planning


**Note:** This is for planning purposes only.
Consult a financial advisor for personalized advice.
""")


st.sidebar.markdown("---")
st.sidebar.caption("⚠️ This tool is for educational purposes only. Not financial advice.")


# Key assumptions footer
st.markdown("---")
st.subheader("📋 Key Assumptions & Disclaimers")


col1, col2 = st.columns(2)


with col1:
    st.info("""
    **Key Assumptions:**
    - Superannuation grows at specified investment return rate
    - Age Pension rates and thresholds remain constant in real terms
    - Salary growth continues until retirement
    - Investment returns are consistent (actual returns will vary)
    - Age Pension eligibility based on current rules
    """)


with col2:
    st.info("""
    **Important Notes:**
    - Age Pension rules may change over time
    - Superannuation rules and contribution caps may change
    - Investment returns will fluctuate
    - Healthcare costs may be higher than estimated
    - Consider seeking professional financial advice
    - Tax implications not fully modeled
    """)


st.warning("""
**Disclaimer:** This calculator provides estimates for educational and planning purposes only.
Actual retirement outcomes will depend on many factors including investment performance,
changes to government policy, inflation, and personal circumstances.
Always consult with a qualified financial advisor for personalized retirement planning advice.
""")

