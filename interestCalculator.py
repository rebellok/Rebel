import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math

# Page configuration
st.set_page_config(
    page_title="Interest Calculator",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Interest Calculator")

# Create tabs
tab1, tab2 = st.tabs(["Basic Calculator", "Advanced Calculator"])

def calculate_compound_frequency(frequency):
    """Convert frequency string to number of times per year"""
    freq_map = {
        "Annual": 1,
        "Semiannually": 2,
        "Quarterly": 4,
        "Monthly": 12,
        "Weekly": 52,
        "Bimonthly": 6,
        "Daily": 365
    }
    return freq_map.get(frequency, 12)

def calculate_basic_interest(principal, rate, compound_freq, time_years, monthly_deposit, deposit_timing):
    """Calculate basic compound interest with regular deposits"""
    n = compound_freq
    r = rate / 100
    
    # Convert to monthly calculations for deposits
    months = int(time_years * 12)
    monthly_rate = r / 12
    
    balance = principal
    total_deposits = 0
    
    for month in range(months):
        # Add deposit at beginning or end of month
        if deposit_timing == "Beginning of month":
            balance += monthly_deposit
            total_deposits += monthly_deposit
        
        # Apply compound interest (monthly for simplicity)
        balance = balance * (1 + monthly_rate)
        
        if deposit_timing == "End of month":
            balance += monthly_deposit
            total_deposits += monthly_deposit
    
    total_invested = principal + total_deposits
    total_interest = balance - total_invested
    
    # Calculate effective annual rate
    effective_rate = ((balance / principal) ** (1 / time_years) - 1) * 100 if time_years > 0 else 0
    
    return balance, total_invested, total_interest, total_deposits, effective_rate

def calculate_advanced_interest(principal, rate, compound_freq, time_years, monthly_deposit, deposit_timing,
                              limit_deposits, deposit_from_year, deposit_to_year, withdrawal_freq, withdrawal_type, 
                              withdrawal_amount, limit_withdrawals, withdrawal_from_year, withdrawal_to_year):
    """Calculate advanced compound interest with deposits and withdrawals"""
    months = int(time_years * 12)
    monthly_rate = rate / 100 / 12
    
    # Initialize tracking variables
    balance_history = []
    balance = principal
    total_deposits = monthly_deposit * 12 if monthly_deposit > 0 else 0  # Initial estimate
    total_withdrawals = 0
    monthly_data = []
    
    # Withdrawal frequency mapping
    withdrawal_freq_map = {
        "Monthly": 1,
        "Quarterly": 3,
        "Semiannually": 6,
        "Bimonthly": 2,
        "Annually": 12
    }
    
    withdrawal_interval = withdrawal_freq_map.get(withdrawal_freq, 12)
    total_deposits = 0
    
    for month in range(1, months + 1):
        month_start_balance = balance
        month_deposits = 0
        month_withdrawals = 0
        
        # Check if we should add deposits
        current_year = (month - 1) // 12 + 1
        if not limit_deposits or (deposit_from_year <= current_year <= deposit_to_year):
            if deposit_timing == "Beginning of month":
                balance += monthly_deposit
                month_deposits = monthly_deposit
                total_deposits += monthly_deposit
        
        # Apply monthly compound interest
        interest_this_month = balance * monthly_rate
        balance += interest_this_month
        
        # Add deposit at end of month if applicable
        if not limit_deposits or (deposit_from_year <= current_year <= deposit_to_year):
            if deposit_timing == "End of month":
                balance += monthly_deposit
                month_deposits = monthly_deposit
                total_deposits += monthly_deposit
        
        # Handle withdrawals
        if month % withdrawal_interval == 0:  # Time for withdrawal
            if not limit_withdrawals or (withdrawal_from_year <= current_year <= withdrawal_to_year):
                if withdrawal_type == "Fixed value":
                    withdrawal = min(withdrawal_amount, balance)
                elif withdrawal_type == "Balance percentage":
                    withdrawal = balance * (withdrawal_amount / 100)
                elif withdrawal_type == "Interest percentage":
                    withdrawal = interest_this_month * (withdrawal_amount / 100)
                
                balance -= withdrawal
                month_withdrawals = withdrawal
                total_withdrawals += withdrawal
        
        # Store monthly data
        monthly_data.append({
            'month': month,
            'year': current_year,
            'starting_balance': month_start_balance,
            'ending_balance': balance,
            'deposits_this_month': month_deposits,
            'withdrawals_this_month': month_withdrawals,
            'interest_this_month': interest_this_month,
            'total_deposits_so_far': total_deposits,
            'total_withdrawals_so_far': total_withdrawals
        })
        
        balance_history.append(balance)
    
    total_invested = principal + total_deposits
    total_interest = balance - total_invested + total_withdrawals
    effective_rate = ((balance / principal) ** (1 / time_years) - 1) * 100 if time_years > 0 and principal > 0 else 0
    
    return balance, total_invested, total_interest, total_deposits, total_withdrawals, effective_rate, monthly_data

# Basic Calculator Tab
with tab1:
    st.header("Basic Interest Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        initial_value = st.number_input("Initial Investment ($)", min_value=0.0, value=1000.0, step=100.0)
        interest_rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, value=5.0, step=0.1)
        compound_frequency = st.selectbox("Compounding Frequency", 
                                        ["Annual", "Semiannually", "Quarterly", "Monthly", "Weekly", "Bimonthly", "Daily"],
                                        index=3)
    
    with col2:
        duration_unit = st.selectbox("Duration Unit", ["Years", "Months"])
        if duration_unit == "Years":
            duration = st.number_input("Duration (Years)", min_value=0.1, value=5.0, step=0.1)
            duration_years = duration
        else:
            duration = st.number_input("Duration (Months)", min_value=1, value=60, step=1)
            duration_years = duration / 12
        
        monthly_deposit = st.number_input("Monthly Deposit ($)", min_value=0.0, value=100.0, step=10.0)
        deposit_timing = st.selectbox("Deposit Timing", ["Beginning of month", "End of month"])
    
    if st.button("Calculate", key="basic_calc"):
        ending_balance, total_invested, total_interest, total_deposits, effective_rate = calculate_basic_interest(
            initial_value, interest_rate, calculate_compound_frequency(compound_frequency), 
            duration_years, monthly_deposit, deposit_timing
        )
        
        st.subheader("Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ending Balance", f"${ending_balance:,.2f}")
            st.metric("Total Invested", f"${total_invested:,.2f}")
        
        with col2:
            st.metric("Total Interest Earned", f"${total_interest:,.2f}")
            st.metric("Total Deposits", f"${total_deposits:,.2f}")
        
        with col3:
            st.metric("Effective Annual Rate", f"{effective_rate:.2f}%")

# Advanced Calculator Tab
with tab2:
    st.header("Advanced Interest Calculator")
    
    # Input sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Parameters")
        adv_initial_value = st.number_input("Initial Investment ($)", min_value=0.0, value=1000.0, step=100.0, key="adv_initial")
        adv_interest_rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, value=5.0, step=0.1, key="adv_rate")
        
        st.subheader("Deposits")
        adv_monthly_deposit = st.number_input("Monthly Deposit ($)", min_value=0.0, value=100.0, step=10.0, key="adv_deposit")
        adv_deposit_timing = st.selectbox("Deposit Timing", ["Beginning of month", "End of month"], key="adv_timing")
        
        limit_deposits = st.checkbox("Limit deposits to a period")
        if limit_deposits:
            col_dep1, col_dep2 = st.columns(2)
            with col_dep1:
                deposit_from_year = st.number_input("From year", min_value=1, max_value=100, value=1, key="dep_from")
            with col_dep2:
                deposit_to_year = st.number_input("To year", min_value=1, max_value=100, value=10, key="dep_to")
        else:
            deposit_from_year = 1
            deposit_to_year = 100
    
    with col2:
        st.subheader("Withdrawals")
        withdrawal_freq = st.selectbox("Withdrawal Frequency", ["Monthly", "Quarterly", "Semiannually", "Bimonthly", "Annually"])
        withdrawal_type = st.selectbox("Withdrawal Type", ["Fixed value", "Balance percentage", "Interest percentage"])
        
        if withdrawal_type == "Fixed value":
            withdrawal_amount = st.number_input("Withdrawal Amount ($)", min_value=0.0, value=50.0, step=10.0)
        else:
            withdrawal_amount = st.number_input("Withdrawal Percentage (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        
        limit_withdrawals = st.checkbox("Limit withdrawals to a period")
        if limit_withdrawals:
            col_with1, col_with2 = st.columns(2)
            with col_with1:
                withdrawal_from_year = st.number_input("From year", min_value=1, max_value=100, value=1, key="with_from")
            with col_with2:
                withdrawal_to_year = st.number_input("To year", min_value=1, max_value=100, value=15, key="with_to")
        else:
            withdrawal_from_year = 1
            withdrawal_to_year = 100
    
    if st.button("Calculate Advanced", key="adv_calc"):
        # Determine the actual calculation period based on user inputs
        max_period_years = max(deposit_to_year if limit_deposits else 10, 
                              withdrawal_to_year if limit_withdrawals else 10)
        
        ending_balance, total_invested, total_interest, total_deposits, total_withdrawals, effective_rate, monthly_data = calculate_advanced_interest(
            adv_initial_value, adv_interest_rate, 12, max_period_years, adv_monthly_deposit, adv_deposit_timing,
            limit_deposits, deposit_from_year, deposit_to_year, withdrawal_freq, withdrawal_type, withdrawal_amount,
            limit_withdrawals, withdrawal_from_year, withdrawal_to_year
        )
        
        # Summary Section
        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Ending Balance", f"${ending_balance:,.2f}")
            st.metric("Total Invested", f"${total_invested:,.2f}")
        
        with col2:
            st.metric("Total Interest Earned", f"${total_interest:,.2f}")
            st.metric("Total Deposits", f"${total_deposits:,.2f}")
        
        with col3:
            st.metric("Total Withdrawals", f"${total_withdrawals:,.2f}")
            st.metric("Effective Annual Rate", f"{effective_rate:.2f}%")
        
        # Disclaimers
        st.info("⚠️ **Disclaimers**: This calculator provides estimates based on the inputs provided. Actual results may vary due to market conditions, fees, taxes, and other factors. Past performance does not guarantee future results. Consult with a financial advisor for personalized advice.")
        
        # Graph Section
        st.subheader("Balance Evolution")
        
        # Create DataFrame for plotting
        df_monthly = pd.DataFrame(monthly_data)
        df_monthly['cumulative_invested'] = adv_initial_value + df_monthly['total_deposits_so_far']
        
        # Group by year for yearly view
        df_yearly = df_monthly.groupby('year').agg({
            'ending_balance': 'last',
            'cumulative_invested': 'last',
            'total_deposits_so_far': 'last',
            'total_withdrawals_so_far': 'last'
        }).reset_index()
        
        # Dropdown for time period
        time_period = st.selectbox("View by:", ["By Year", "By Month"])
        
        if time_period == "By Year":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_yearly['year'], y=df_yearly['ending_balance'], 
                                   mode='lines+markers', name='Balance', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df_yearly['year'], y=df_yearly['cumulative_invested'], 
                                   mode='lines+markers', name='Total Invested', line=dict(color='green')))
            fig.update_layout(title="Balance Evolution Over Time", xaxis_title="Years", yaxis_title="Amount ($)")
        else:
            months_x = [i/12 for i in range(1, len(monthly_data) + 1)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=months_x, y=df_monthly['ending_balance'], 
                                   mode='lines', name='Balance', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=months_x, y=df_monthly['cumulative_invested'], 
                                   mode='lines', name='Total Invested', line=dict(color='green')))
            fig.update_layout(title="Balance Evolution Over Time", xaxis_title="Years", yaxis_title="Amount ($)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Breakdown Section
        st.subheader("Breakdown")
        breakdown_view = st.selectbox("Breakdown by:", ["By Year", "By Month"], key="breakdown")
        
        if breakdown_view == "By Year":
            # Initial value display
            st.markdown("### 📊 Year-by-Year Breakdown")
            
            # Initial Investment Card
            st.markdown("""
            <div style="background-color: #e8f4fd; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #1f77b4;">
                <h4 style="margin: 0; color: #1f77b4;">🏦 Initial Investment</h4>
                <h3 style="margin: 5px 0; color: #1f77b4;">${:,.2f}</h3>
            </div>
            """.format(adv_initial_value), unsafe_allow_html=True)
            
            # Yearly breakdown with enhanced styling
            for year in df_yearly['year']:
                year_data = df_monthly[df_monthly['year'] == year]
                year_start_balance = year_data.iloc[0]['starting_balance'] if len(year_data) > 0 else 0
                year_end_balance = year_data.iloc[-1]['ending_balance'] if len(year_data) > 0 else 0
                year_deposits = year_data['deposits_this_month'].sum()
                year_withdrawals = year_data['withdrawals_this_month'].sum()
                year_interest = year_data['interest_this_month'].sum()
                total_deposits_so_far = year_data.iloc[-1]['total_deposits_so_far'] if len(year_data) > 0 else 0
                total_withdrawals_so_far = year_data.iloc[-1]['total_withdrawals_so_far'] if len(year_data) > 0 else 0
                balance_change = year_end_balance - year_start_balance
                
                # Determine card color based on balance change
                if balance_change > 0:
                    card_color = "#d4edda"
                    border_color = "#28a745"
                    icon = "📈"
                elif balance_change < 0:
                    card_color = "#f8d7da"
                    border_color = "#dc3545"
                    icon = "📉"
                else:
                    card_color = "#fff3cd"
                    border_color = "#ffc107"
                    icon = "📊"
                
                # Year card with enhanced styling
                st.markdown(f"""
                <div style="background-color: {card_color}; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid {border_color};">
                    <h4 style="margin: 0; color: {border_color};">{icon} Year {year}</h4>
                    <h3 style="margin: 5px 0; color: {border_color};">Balance: ${year_end_balance:,.2f}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Expandable details for each year
                with st.expander(f"📋 Detailed View - Year {year}", expanded=False):
                    # Create three columns for better organization
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**💰 Balance Information**")
                        st.metric("Starting Balance", f"${year_start_balance:,.2f}")
                        st.metric("Ending Balance", f"${year_end_balance:,.2f}")
                        st.metric("Balance Change", f"${balance_change:,.2f}", 
                                delta=f"{((balance_change/year_start_balance)*100):.1f}%" if year_start_balance > 0 else "N/A")
                    
                    with col2:
                        st.markdown("**📥 Deposits & Contributions**")
                        st.metric("Deposits This Year", f"${year_deposits:,.2f}")
                        st.metric("Total Deposits So Far", f"${total_deposits_so_far:,.2f}")
                        st.metric("Total Contributions", f"${adv_initial_value + total_deposits_so_far:,.2f}")
                    
                    with col3:
                        st.markdown("**📤 Withdrawals & Interest**")
                        st.metric("Withdrawals This Year", f"${year_withdrawals:,.2f}")
                        st.metric("Total Withdrawals So Far", f"${total_withdrawals_so_far:,.2f}")
                        st.metric("Interest Earned This Year", f"${year_interest:,.2f}")
                    
                    # Additional insights
                    st.markdown("---")
                    st.markdown("**📊 Year Summary**")
                    
                    insight_col1, insight_col2 = st.columns(2)
                    with insight_col1:
                        net_flow = year_deposits - year_withdrawals
                        st.write(f"**Net Cash Flow:** ${net_flow:,.2f}")
                        roi_this_year = (year_interest / year_start_balance * 100) if year_start_balance > 0 else 0
                        st.write(f"**Return on Investment:** {roi_this_year:.2f}%")
                    
                    with insight_col2:
                        total_growth = year_end_balance - adv_initial_value
                        st.write(f"**Total Growth from Start:** ${total_growth:,.2f}")
                        if year_start_balance > 0:
                            growth_rate = ((year_end_balance / adv_initial_value) ** (1/year) - 1) * 100
                            st.write(f"**Annualized Growth Rate:** {growth_rate:.2f}%")
        
        else:  # By Month view
            # Initial value row
            with st.expander("Initial Investment", expanded=True):
                st.write(f"**Balance**: ${adv_initial_value:,.2f}")
            
            # Monthly breakdown
            for _, month_row in df_monthly.iterrows():
                month_display = f"Year {month_row['year']}, Month {((month_row['month'] - 1) % 12) + 1}"
                with st.expander(month_display):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Balance**: ${month_row['ending_balance']:,.2f}")
                    with col2:
                        with st.expander("Interest Details"):
                            st.write(f"Starting Balance: ${month_row['starting_balance']:,.2f}")
                            st.write(f"Ending Balance: ${month_row['ending_balance']:,.2f}")
                            st.write(f"All Deposits So Far: ${month_row['total_deposits_so_far']:,.2f}")
                            st.write(f"Deposits This Month: ${month_row['deposits_this_month']:,.2f}")
                            st.write(f"All Withdrawals So Far: ${month_row['total_withdrawals_so_far']:,.2f}")
                            st.write(f"Withdrawals This Month: ${month_row['withdrawals_this_month']:,.2f}")
                            st.write(f"Interest This Month: ${month_row['interest_this_month']:,.2f}")

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: Use the advanced calculator for more detailed planning with withdrawals and time-limited deposits/withdrawals.")