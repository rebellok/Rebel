#User: generate python code using streamlit for UI, I am looking to build a Tax estimator, ask all the questions relavant in order to estimate taxes for a year, use tax rates of 2024 where taxes will be filed in 2025
#Model: # Tax Estimator Application using Streamlit for 2024 Tax Year


#Below is a Python application using Streamlit to create a tax estimation tool for the 2024 tax year (filed in 2025). This application will gather relevant information from the user and estimate their federal income tax liability.


import streamlit as st
import pandas as pd
import numpy as np


def main():
    st.title("2024 Tax Year Estimator")
    st.subheader("For taxes to be filed in 2025")
   
    st.info("This tool provides an estimate of your federal income tax liability for the 2024 tax year. This is not professional tax advice.")
   
    # Sidebar for navigation
    nav_selection = st.sidebar.radio(
        "Navigation",
        ["Personal Information", "Income", "Deductions & Credits", "Tax Estimate"]
    )
   
    # Initialize session state variables if they don't exist
    if 'filing_status' not in st.session_state:
        st.session_state.filing_status = 'single'
    if 'dependents' not in st.session_state:
        st.session_state.dependents = 0
    if 'wages' not in st.session_state:
        st.session_state.wages = 0.0
    if 'interest' not in st.session_state:
        st.session_state.interest = 0.0
    if 'dividends' not in st.session_state:
        st.session_state.dividends = 0.0
    if 'capital_gains' not in st.session_state:
        st.session_state.capital_gains = 0.0
    if 'self_employment' not in st.session_state:
        st.session_state.self_employment = 0.0
    if 'other_income' not in st.session_state:
        st.session_state.other_income = 0.0
    if 'itemized_deductions' not in st.session_state:
        st.session_state.itemized_deductions = 0.0
    if 'retirement_contributions' not in st.session_state:
        st.session_state.retirement_contributions = 0.0
    if 'student_loan_interest' not in st.session_state:
        st.session_state.student_loan_interest = 0.0
    if 'charitable_contributions' not in st.session_state:
        st.session_state.charitable_contributions = 0.0
    if 'mortgage_interest' not in st.session_state:
        st.session_state.mortgage_interest = 0.0
    if 'medical_expenses' not in st.session_state:
        st.session_state.medical_expenses = 0.0
    if 'state_local_taxes' not in st.session_state:
        st.session_state.state_local_taxes = 0.0
    if 'tax_credits' not in st.session_state:
        st.session_state.tax_credits = 0.0
    if 'tax_withheld' not in st.session_state:
        st.session_state.tax_withheld = 0.0
    if 'qualified_dividends' not in st.session_state:
        st.session_state.qualified_dividends = 0.0
    if 'long_term_capital_gains' not in st.session_state:
        st.session_state.long_term_capital_gains = 0.0
    if 'use_standard_deduction' not in st.session_state:
        st.session_state.use_standard_deduction = True
    if 'child_tax_credit' not in st.session_state:
        st.session_state.child_tax_credit = 0.0
    if 'dependent_care_credit' not in st.session_state:
        st.session_state.dependent_care_credit = 0.0


    if nav_selection == "Personal Information":
        display_personal_info()
    elif nav_selection == "Income":
        display_income_section()
    elif nav_selection == "Deductions & Credits":
        display_deductions_credits()
    elif nav_selection == "Tax Estimate":
        display_tax_estimate()


def display_personal_info():
    st.header("Personal Information")
    st.session_state.filing_status = st.selectbox(
        "Filing Status",
        ["single", "married_joint", "married_separate", "head_of_household", "qualifying_widow"],
        index=["single", "married_joint", "married_separate", "head_of_household", "qualifying_widow"].index(st.session_state.filing_status)
    )
    st.session_state.dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=20,
        value=st.session_state.dependents,
        step=1
    )
    # Display standard deduction info based on filing status
    standard_deduction = get_standard_deduction(st.session_state.filing_status)
    st.info(f"Your standard deduction for 2024 will be ${standard_deduction:,.2f}")


def display_income_section():
    st.header("Income Information")
    st.session_state.wages = st.number_input(
        "W-2 Wages",
        min_value=0.0,
        value=st.session_state.wages,
        format="%.2f"
    )
    st.session_state.interest = st.number_input(
        "Interest Income",
        min_value=0.0,
        value=st.session_state.interest,
        format="%.2f"
    )
    st.session_state.dividends = st.number_input(
        "Total Dividends",
        min_value=0.0,
        value=st.session_state.dividends,
        format="%.2f"
    )
    st.session_state.qualified_dividends = st.number_input(
        "Qualified Dividends (portion of total dividends)",
        min_value=0.0,
        max_value=st.session_state.dividends,
        value=min(st.session_state.qualified_dividends, st.session_state.dividends),
        format="%.2f"
    )
    st.session_state.capital_gains = st.number_input(
        "Total Capital Gains/Losses",
        value=st.session_state.capital_gains,
        format="%.2f"
    )
    st.session_state.long_term_capital_gains = st.number_input(
        "Long-Term Capital Gains (held over 1 year)",
        min_value=0.0,
        value=st.session_state.long_term_capital_gains,
        format="%.2f"
    )
    st.session_state.self_employment = st.number_input(
        "Self-Employment Income",
        min_value=0.0,
        value=st.session_state.self_employment,
        format="%.2f"
    )
    st.session_state.other_income = st.number_input(
        "Other Income",
        min_value=0.0,
        value=st.session_state.other_income,
        format="%.2f"
    )
    total_income = calculate_total_income()
    st.subheader(f"Total Gross Income: ${total_income:,.2f}")


def calculate_total_income():
    return (st.session_state.wages + st.session_state.interest +
            st.session_state.dividends + st.session_state.capital_gains +
            st.session_state.self_employment + st.session_state.other_income)


def get_standard_deduction(filing_status):
    # 2024 Standard Deduction amounts
    deductions = {
        "single": 14600,
        "married_joint": 29200,
        "married_separate": 14600,
        "head_of_household": 21900,
        "qualifying_widow": 29200
    }
    return deductions.get(filing_status, 12950)  # Default to single if not found


def calculate_taxable_income():
    total_income = calculate_total_income()
    # Adjustments to income
    adjustments = (st.session_state.retirement_contributions +
                  st.session_state.student_loan_interest)
    adjusted_gross_income = total_income - adjustments
    # Determine whether to use standard or itemized deduction
    standard_deduction = get_standard_deduction(st.session_state.filing_status)
    itemized_deductions = (st.session_state.charitable_contributions +
                          st.session_state.mortgage_interest +
                          st.session_state.state_local_taxes +
                          st.session_state.medical_expenses)
    # Cap on state and local tax deductions
    if st.session_state.state_local_taxes > 10000:
        itemized_deductions -= (st.session_state.state_local_taxes - 10000)
    # Medical expenses must exceed 7.5% of AGI to be deductible
    medical_threshold = adjusted_gross_income * 0.075
    if st.session_state.medical_expenses <= medical_threshold:
        itemized_deductions -= st.session_state.medical_expenses
    else:
        itemized_deductions -= medical_threshold
    # Use the larger of standard or itemized deductions
    if st.session_state.use_standard_deduction or standard_deduction > itemized_deductions:
        deduction = standard_deduction
    else:
        deduction = itemized_deductions
    taxable_income = max(0, adjusted_gross_income - deduction)
    return taxable_income, adjusted_gross_income, deduction


def display_deductions_credits():
    st.header("Deductions and Credits")
    # Above-the-line deductions
    st.subheader("Adjustments to Income")
    # Retirement contributions
    st.session_state.retirement_contributions = st.number_input(
        "Traditional IRA/401(k) Contributions",
        min_value=0.0,
        value=st.session_state.retirement_contributions,
        format="%.2f"
    )
    # Student loan interest
    st.session_state.student_loan_interest = st.number_input(
        "Student Loan Interest",
        min_value=0.0,
        max_value=2500.0,  # 2024 limit
        value=min(st.session_state.student_loan_interest, 2500.0),
        format="%.2f"
    )
    # Itemized deductions section
    st.subheader("Deductions")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.use_standard_deduction = st.checkbox(
            "Use Standard Deduction",
            value=st.session_state.use_standard_deduction
        )
    # Show the standard deduction amount
    standard_deduction = get_standard_deduction(st.session_state.filing_status)
    with col2:
        st.write(f"Standard Deduction: ${standard_deduction:,.2f}")
    # Only show itemized deduction inputs if not using standard deduction
    if not st.session_state.use_standard_deduction:
        st.write("Itemized Deductions:")
        st.session_state.charitable_contributions = st.number_input(
            "Charitable Contributions",
            min_value=0.0,
            value=st.session_state.charitable_contributions,
            format="%.2f"
        )
        st.session_state.mortgage_interest = st.number_input(
            "Mortgage Interest",
            min_value=0.0,
            value=st.session_state.mortgage_interest,
            format="%.2f"
        )
        st.session_state.state_local_taxes = st.number_input(
            "State/Local Taxes Paid (capped at $10,000)",
            min_value=0.0,
            value=st.session_state.state_local_taxes,
            format="%.2f"
        )
        st.session_state.medical_expenses = st.number_input(
            "Medical Expenses (only amount exceeding 7.5% of AGI is deductible)",
            min_value=0.0,
            value=st.session_state.medical_expenses,
            format="%.2f"
        )
        # Calculate and display total itemized deductions
        total_itemized = (st.session_state.charitable_contributions +
                          st.session_state.mortgage_interest +
                          min(st.session_state.state_local_taxes, 10000))
        # Calculate the medical expense deduction
        agi = calculate_total_income() - st.session_state.retirement_contributions - st.session_state.student_loan_interest
        medical_threshold = agi * 0.075
        medical_deduction = max(0, st.session_state.medical_expenses - medical_threshold)
        total_itemized += medical_deduction
        st.write(f"Total Itemized Deductions: ${total_itemized:,.2f}")
        if total_itemized < standard_deduction:
            st.warning(f"Your itemized deductions (${total_itemized:,.2f}) are less than your standard deduction (${standard_deduction:,.2f}). You may want to use the standard deduction instead.")
    # Tax Credits section
    st.subheader("Tax Credits")
    if st.session_state.dependents > 0:
        st.write(f"You have {st.session_state.dependents} dependent(s)")
        child_count = st.number_input(
            "How many dependents are qualifying children under 17?",
            min_value=0,
            max_value=st.session_state.dependents,
            value=min(st.session_state.dependents, 0),
            step=1
        )
        if child_count > 0:
            st.session_state.child_tax_credit = child_count * 2000  # $2,000 per qualifying child for 2024
            st.write(f"Estimated Child Tax Credit: ${st.session_state.child_tax_credit:,.2f}")
    st.session_state.dependent_care_credit = st.number_input(
        "Child and Dependent Care Credit",
        min_value=0.0,
        value=st.session_state.dependent_care_credit,
        format="%.2f"
    )
    # Other tax credits
    st.session_state.tax_credits = st.number_input(
        "Other Tax Credits",
        min_value=0.0,
        value=st.session_state.tax_credits,
        format="%.2f"
    )
    # Tax withholding
    st.subheader("Tax Payments")
    st.session_state.tax_withheld = st.number_input(
        "Federal Income Tax Withheld",
        min_value=0.0,
        value=st.session_state.tax_withheld,
        format="%.2f"
    )


def calculate_income_tax(taxable_income):
    # 2024 Federal income tax brackets
    brackets = {
        "single": [
            (0, 11600, 0.10),
            (11600, 47150, 0.12),
            (47150, 100525, 0.22),
            (100525, 191950, 0.24),
            (191950, 243725, 0.32),
            (243725, 609350, 0.35),
            (609350, float('inf'), 0.37)
        ],
        "married_joint": [
            (0, 23200, 0.10),
            (23200, 94300, 0.12),
            (94300, 201050, 0.22),
            (201050, 383900, 0.24),
            (383900, 487450, 0.32),
            (487450, 731200, 0.35),
            (731200, float('inf'), 0.37)
        ],
        "married_separate": [
            (0, 11600, 0.10),
            (11600, 47150, 0.12),
            (47150, 100525, 0.22),
            (100525, 191950, 0.24),
            (191950, 243725, 0.32),
            (243725, 365600, 0.35),
            (365600, float('inf'), 0.37)
        ],
        "head_of_household": [
            (0, 16550, 0.10),
            (16550, 63100, 0.12),
            (63100, 100500, 0.22),
            (100500, 191950, 0.24),
            (191950, 243700, 0.32),
            (243700, 609350, 0.35),
            (609350, float('inf'), 0.37)
        ],
        "qualifying_widow": [
            (0, 23200, 0.10),
            (23200, 94300, 0.12),
            (94300, 201050, 0.22),
            (201050, 383900, 0.24),
            (383900, 487450, 0.32),
            (487450, 731200, 0.35),
            (731200, float('inf'), 0.37)
        ]
    }
    filing_status = st.session_state.filing_status
    if filing_status not in brackets:
        filing_status = "single"  # Default to single if status not found
    tax = 0
    for lower, upper, rate in brackets[filing_status]:
        if taxable_income > lower:
            tax += (min(taxable_income, upper) - lower) * rate
            if taxable_income <= upper:
                break
    return tax


def calculate_capital_gains_tax(adjusted_gross_income):
    # 2024 Capital gains tax brackets
    brackets = {
        "single": [
            (0, 44625, 0.00),
            (44625, 492300, 0.15),
            (492300, float('inf'), 0.20)
        ],
        "married_joint": [
            (0, 89250, 0.00),
            (89250, 553850, 0.15),
            (553850, float('inf'), 0.20)
        ],
        "married_separate": [
            (0, 44625, 0.00),
            (44625, 276900, 0.15),
            (276900, float('inf'), 0.20)
        ],
        "head_of_household": [
            (0, 59750, 0.00),
            (59750, 523050, 0.15),
            (523050, float('inf'), 0.20)
        ],
        "qualifying_widow": [
            (0, 89250, 0.00),
            (89250, 553850, 0.15),
            (553850, float('inf'), 0.20)
        ]
    }
    filing_status = st.session_state.filing_status
    if filing_status not in brackets:
        filing_status = "single"  # Default to single if status not found
    # Calculate tax on qualified dividends and long-term capital gains
    qualified_investment_income = (st.session_state.qualified_dividends +
                                  st.session_state.long_term_capital_gains)
    if qualified_investment_income <= 0:
        return 0
    tax = 0
    for lower, upper, rate in brackets[filing_status]:
        if adjusted_gross_income > lower:
            bracket_income = min(adjusted_gross_income, upper) - lower
            applicable_income = min(qualified_investment_income, bracket_income)
            if applicable_income > 0:
                tax += applicable_income * rate
                qualified_investment_income -= applicable_income
            if qualified_investment_income <= 0 or adjusted_gross_income <= upper:
                break
    return tax


def calculate_self_employment_tax():
    se_income = st.session_state.self_employment
    if se_income <= 0:
        return 0
    # 2024 self-employment tax rates
    # Social Security: 12.4% on first $168,600
    # Medicare: 2.9% on all self-employment income
    ss_wage_base = 168600
    ss_tax_rate = 0.124
    medicare_tax_rate = 0.029
    ss_tax = min(se_income, ss_wage_base) * ss_tax_rate
    medicare_tax = se_income * medicare_tax_rate
    return ss_tax + medicare_tax


def display_tax_estimate():
    st.header("2024 Tax Estimate")
    taxable_income, adjusted_gross_income, deduction = calculate_taxable_income()
    # Calculate regular income tax
    regular_tax = calculate_income_tax(taxable_income)
    # Calculate capital gains and qualified dividends tax
    capital_gains_tax = calculate_capital_gains_tax(adjusted_gross_income)
    # Calculate self-employment tax if applicable
    self_employment_tax = calculate_self_employment_tax()
    # Calculate total tax
    total_tax = regular_tax + capital_gains_tax + self_employment_tax
    # Subtract tax credits
    total_credits = (st.session_state.child_tax_credit +
                    st.session_state.dependent_care_credit +
                    st.session_state.tax_credits)
    tax_after_credits = max(0, total_tax - total_credits)
    # Calculate amount owed or refund
    amount_due = tax_after_credits - st.session_state.tax_withheld
    # Display the results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Income Summary")
        st.write(f"Total Gross Income: ${calculate_total_income():,.2f}")
        st.write(f"Adjustments to Income: ${(st.session_state.retirement_contributions + st.session_state.student_loan_interest):,.2f}")
        st.write(f"Adjusted Gross Income (AGI): ${adjusted_gross_income:,.2f}")
        st.write(f"Deduction Used: ${deduction:,.2f}")
        st.write(f"Taxable Income: ${taxable_income:,.2f}")
    with col2:
        st.subheader("Tax Summary")
        st.write(f"Regular Income Tax: ${regular_tax:,.2f}")
        if capital_gains_tax > 0:
            st.write(f"Capital Gains/Qualified Dividends Tax: ${capital_gains_tax:,.2f}")
        if self_employment_tax > 0:
            st.write(f"Self-Employment Tax: ${self_employment_tax:,.2f}")
        st.write(f"Total Tax Before Credits: ${total_tax:,.2f}")
        st.write(f"Tax Credits: ${total_credits:,.2f}")
        st.write(f"Tax After Credits: ${tax_after_credits:,.2f}")
        st.write(f"Federal Tax Withheld: ${st.session_state.tax_withheld:,.2f}")
    st.markdown("---")
    if amount_due > 0:
        st.error(f"### Estimated Amount You Owe: ${amount_due:,.2f}")
    else:
        st.success(f"### Estimated Refund: ${abs(amount_due):,.2f}")
    st.markdown("---")
    st.caption("""
    **Disclaimer**: This is an estimate based on the information provided and known tax rates for 2024.
    Actual tax liability may differ due to changes in tax law, eligibility for additional deductions or credits,
    or other factors. Please consult a tax professional for advice specific to your situation.
    """)


def display_insights():
    taxable_income, adjusted_gross_income, deduction = calculate_taxable_income()
    st.header("Tax Insights")
    # Check if user is using standard deduction when itemized might be better
    standard_deduction = get_standard_deduction(st.session_state.filing_status)
    itemized_total = (st.session_state.charitable_contributions +
                     st.session_state.mortgage_interest +
                     min(st.session_state.state_local_taxes, 10000))
    # Medical expense calculation
    medical_threshold = adjusted_gross_income * 0.075
    medical_deduction = max(0, st.session_state.medical_expenses - medical_threshold)
    itemized_total += medical_deduction
    if st.session_state.use_standard_deduction and itemized_total > standard_deduction:
        st.warning(f"You might save ${itemized_total - standard_deduction:,.2f} by itemizing deductions instead of taking the standard deduction.")
    # Retirement contribution insights
    if st.session_state.wages > 0 and st.session_state.retirement_contributions == 0:
        st.info("Contributing to a traditional retirement account could reduce your taxable income.")
    # Self-employment tax insight
    if st.session_state.self_employment > 0:
        st.info("As a self-employed individual, you may be eligible for additional deductions such as health insurance premiums and business expenses.")
    # Child tax credit insight
    if st.session_state.dependents > 0 and st.session_state.child_tax_credit == 0:
        st.info("You have dependents. If any are qualifying children under 17, you might be eligible for the Child Tax Credit.")
    # Tax bracket insight
    tax_brackets = {
        "single": [
            (0, 11600, "10%"),
            (11600, 47150, "12%"),
            (47150, 100525, "22%"),
            (100525, 191950, "24%"),
            (191950, 243725, "32%"),
            (243725, 609350, "35%"),
            (609350, float('inf'), "37%")
        ],
    }
    # Get tax bracket based on filing status
    brackets = tax_brackets.get(st.session_state.filing_status, tax_brackets["single"])
    # Determine current tax bracket
    current_bracket = brackets[-1][2]  # Default to highest
    for lower, upper, rate in brackets:
        if taxable_income <= upper:
            current_bracket = rate
            break
    st.write(f"Based on your taxable income, you are in the {current_bracket} marginal tax bracket.")






# Example usage:
# balances = {'salary': 5000, 'brokerage': 2000, 'traditional': 1000, 'roth': 500}
# total_deduction = 7000
# new_balances = apply_tax_deduction(total_deduction, balances)
# print(new_balances)  # Deduction is applied in order, no negative


if __name__ == "__main__":
    main()


## Complete Application


#To use this application, save the entire code to a file named `tax_estimator.py` and run it using:


# streamlit run tax_estimator.py


#This application includes:
#1. Personal information collection (filing status, dependents)
#2. Income collection (wages, interest, dividends, capital gains, self-employment)
#3. Deductions and credits calculation
#4. Tax estimation based on 2024 tax brackets
#5. Special handling for qualified dividends and long-term capital gains
#6. Self-employment tax calculation
#7. Tax insights and recommendations
#8. A summary of estimated tax liability or refund


#Please note that this is an educational tool and not a substitute for professional tax advice. The 2024 tax rates used are based on projections and may change before the 2025 filing season.

