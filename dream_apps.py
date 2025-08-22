import streamlit as st




# Page configuration
st.set_page_config(
    page_title="My Apps Launcher",
    page_icon="🚀",
    layout="wide"
)




# Custom CSS for better styling
st.markdown("""
    <style>
        .app-card {
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #ddd;
            margin: 10px 0;
            background-color: #f8f9fa;
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
        .app-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .app-title {
            color: #0066cc;
            font-size: 24px;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .app-description {
            color: #666;
            margin-bottom: 15px;
            line-height: 1.5;
        }
        .app-features {
            color: #555;
            font-size: 14px;
            margin-bottom: 15px;
        }
        .launch-btn {
            display: inline-block;
            padding: 10px 20px;
            background-color: #0066cc;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .launch-btn:hover {
            background-color: #0052a3;
        }
        .section-header {
            color: #0066cc;
            font-size: 28px;
            margin-top: 30px;
            margin-bottom: 20px;
            font-weight: bold;
            border-bottom: 2px solid #0066cc;
            padding-bottom: 10px;
        }
        .canadian-flag {
            display: inline-block;
            margin-right: 8px;
        }
        .indian-flag {
            display: inline-block;
            margin-right: 8px;
        }
        .australian-flag {
            display: inline-block;
            margin-right: 8px;
        }
    </style>
""", unsafe_allow_html=True)




def main():
    # Header
    st.title("🎯 My Applications Hub")
    st.markdown("### Welcome to my collection of web applications!")
    st.markdown("Explore powerful financial tools designed to help you make informed decisions about your money.")
    st.markdown("---")




    # Retirement Planning Section (TOP PRIORITY)
    st.markdown('<div class="section-header">🏖️ Retirement Planning Tools</div>', unsafe_allow_html=True)




    col1, col2 = st.columns(2)




    with col1:
        # US Retirement Calculator App
        st.markdown("""
            <div class="app-card">
                <div class="app-title">🇺🇸 US Retirement Calculator</div>
                <div class="app-description">
                    Plan your financial future with this comprehensive US retirement planning tool.
                    Calculate savings projections and retirement income scenarios for US markets.
                </div>
                <div class="app-features">
                    <strong>Features:</strong><br>
                    • US-focused savings projections<br>
                    • 401(k) & IRA calculations<br>
                    • Social Security integration<br>
                    • Goal planning & scenarios
                </div>
                <a href="https://czsnconjpbqmhpvhjncydg.streamlit.app/" target="_blank" class="launch-btn">
                    🚀 Launch App
                </a>
            </div>
        """, unsafe_allow_html=True)




        # Indian Retirement Calculator App
        st.markdown("""
            <div class="app-card">
                <div class="app-title"><span class="indian-flag">🇮🇳</span>India Retirement Calculator</div>
                <div class="app-description">
                    Comprehensive retirement planning for Indians with EPF, NPS, PPF, and tax-efficient strategies.
                    Includes inflation modeling and Indian tax considerations.
                </div>
                <div class="app-features">
                    <strong>Features:</strong><br>
                    • EPF/NPS/PPF calculations<br>
                    • Indian tax implications<br>
                    • Inflation-adjusted planning<br>
                    • Withdrawal optimization<br>
                    • Single/couple planning
                </div>
                <a href="https://zdgvadyanxoke46j24pmvu.streamlit.app/" target="_blank" class="launch-btn">
                    🚀 Launch App
                </a>
            </div>
        """, unsafe_allow_html=True)




    with col2:
        # Canadian Retirement Calculator App
        st.markdown("""
            <div class="app-card">
                <div class="app-title"><span class="canadian-flag">🇨🇦</span>Canadian Retirement Planner</div>
                <div class="app-description">
                    Comprehensive Canadian retirement planning with RRSP, TFSA, CPP, OAS, and GIS modeling.
                    Includes provincial tax calculations and government benefit optimization.
                </div>
                <div class="app-features">
                    <strong>Features:</strong><br>
                    • RRSP/TFSA/RRIF modeling<br>
                    • CPP & OAS deferral strategies<br>
                    • GIS & OAS clawback calculations<br>
                    • Provincial tax integration
                </div>
                <a href="https://ahnnrccgdzahqlslorty5d.streamlit.app/" target="_blank" class="launch-btn">
                    🚀 Launch App
                </a>
            </div>
        """, unsafe_allow_html=True)




        # Australian Retirement Calculator App
        st.markdown("""
            <div class="app-card">
                <div class="app-title"><span class="australian-flag">🇦🇺</span>Australia Retirement Calculator</div>
                <div class="app-description">
                    Comprehensive Australian retirement planning with superannuation, Age Pension, and investment strategies.
                    Includes assets/income testing and spouse planning capabilities.
                </div>
                <div class="app-features">
                    <strong>Features:</strong><br>
                    • Superannuation projections<br>
                    • Age Pension eligibility testing<br>
                    • Investment property modeling<br>
                    • Spouse retirement planning<br>
                    • Real vs nominal projections
                </div>
                <a href="https://hrrudb5khytu6eknn4g7vs.streamlit.app/" target="_blank" class="launch-btn">
                    🚀 Launch App
                </a>
            </div>
        """, unsafe_allow_html=True)




    # Real Estate & Mortgage Tools Section
    st.markdown('<div class="section-header">🏠 Real Estate & Mortgage Tools</div>', unsafe_allow_html=True)




    col3, col4 = st.columns(2)




    with col3:
        # Mortgage Calculator App
        st.markdown("""
            <div class="app-card">
                <div class="app-title">🏠 Mortgage Calculator</div>
                <div class="app-description">
                    Comprehensive mortgage calculator with amortization schedules, payment breakdowns,
                    and refinancing analysis. Perfect for home buyers and refinance decisions.
                </div>
                <div class="app-features">
                    <strong>Features:</strong><br>
                    • Monthly payment calculations<br>
                    • Amortization schedules<br>
                    • Extra payment scenarios<br>
                    • Refinancing analysis<br>
                    • Interest vs principal breakdown
                </div>
                <a href="https://zaa8dg8owb9crbtms6km6k.streamlit.app/" target="_blank" class="launch-btn">
                    🚀 Launch App
                </a>
            </div>
        """, unsafe_allow_html=True)




    with col4:
        # Rental vs S&P 500 Comparison App
        st.markdown("""
            <div class="app-card">
                <div class="app-title">🏘️ Rental vs S&P 500 Comparison</div>
                <div class="app-description">
                    Compare the long-term returns of investing in rental property versus investing
                    the same amount in S&P 500 index funds. Includes all costs and tax implications.
                </div>
                <div class="app-features">
                    <strong>Features:</strong><br>
                    • Total return comparison<br>
                    • Cash flow analysis<br>
                    • Tax implications modeling<br>
                    • Maintenance & vacancy costs<br>
                    • Risk-adjusted returns
                </div>
                <a href="https://mw56rpzbgawmatwxkhe39a.streamlit.app/" target="_blank" class="launch-btn">
                    🚀 Launch App
                </a>
            </div>
        """, unsafe_allow_html=True)




    # Tax & Financial Planning Section
    st.markdown('<div class="section-header">🧾 Tax & Financial Planning</div>', unsafe_allow_html=True)




    col5, col6 = st.columns(2)




    with col5:
        # Tax Estimator App
        st.markdown("""
            <div class="app-card">
                <div class="app-title">🧾 US Tax Estimator</div>
                <div class="app-description">
                    Estimate your US federal income taxes for 2024 tax year with this comprehensive calculator.
                    Get accurate tax projections and planning insights.
                </div>
                <div class="app-features">
                    <strong>Features:</strong><br>
                    • 2024 tax brackets<br>
                    • Deductions & credits<br>
                    • Multiple filing status<br>
                    • Refund/owe estimates
                </div>
                <a href="https://nmsusr7dcvpdczly5utg6v.streamlit.app/" target="_blank" class="launch-btn">
                    🚀 Launch App
                </a>
            </div>
        """, unsafe_allow_html=True)




    with col6:
        # Interest Calculator App
        st.markdown("""
            <div class="app-card">
                <div class="app-title">💹 Interest Calculator</div>
                <div class="app-description">
                    Advanced compound interest calculator with deposits, withdrawals, and comprehensive planning features.
                    Perfect for savings goals and investment projections.
                </div>
                <div class="app-features">
                    <strong>Features:</strong><br>
                    • Basic & advanced modes<br>
                    • Custom deposit/withdrawal periods<br>
                    • Interactive charts & breakdowns<br>
                    • Multiple compounding frequencies
                </div>
                <a href="https://ixcfeveoappfxmpbr4akpk7.streamlit.app/" target="_blank" class="launch-btn">
                    🚀 Launch App
                </a>
            </div>
        """, unsafe_allow_html=True)




    # Stock Market & Investment Tools Section
    st.markdown('<div class="section-header">📊 Stock Market & Investment Tools</div>', unsafe_allow_html=True)




    col7, col8 = st.columns(2)




    with col7:
        # Portfolio Dashboard App
        st.markdown("""
            <div class="app-card">
                <div class="app-title">📊 Portfolio Dashboard</div>
                <div class="app-description">
                    A comprehensive dashboard for tracking and analyzing investment portfolios.
                    Monitor your investments with real-time data and advanced analytics.
                </div>
                <div class="app-features">
                    <strong>Features:</strong><br>
                    • Performance tracking<br>
                    • Investment predictions<br>
                    • Detailed analytics<br>
                    • Risk assessment
                </div>
                <a href="https://ole63tukdbl5fgamucyqbw.streamlit.app/" target="_blank" class="launch-btn">
                    🚀 Launch App
                </a>
            </div>
        """, unsafe_allow_html=True)




    with col8:
        # Unusual Options Activity App
        st.markdown("""
            <div class="app-card">
                <div class="app-title">🎯 Unusual Options Activity</div>
                <div class="app-description">
                    Track and analyze unusual options activity in US markets. Identify potential market movements
                    through high-volume options trades and institutional flow patterns.
                </div>
                <div class="app-features">
                    <strong>Features:</strong><br>
                    • Real-time options flow detection<br>
                    • Unusual activity scoring<br>
                    • Volume & premium analysis<br>
                    • Interactive filtering & alerts<br>
                    • Export capabilities
                </div>
                <a href="https://2z8ubnzpx44fpqvsuwywi2.streamlit.app/" target="_blank" class="launch-btn">
                    🚀 Launch App
                </a>
            </div>
        """, unsafe_allow_html=True)




    # Quick stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
   
    with col1:
        st.metric("Total Apps", "10", "2")
   
    with col2:
        st.metric("Countries Supported", "US + CA + IN + AU", "🌍")
   
    with col3:
        st.metric("Categories", "4", "1")
   
    with col4:
        st.metric("Latest Addition", "Options Flow & AU Retirement", "🇦🇺📊")




    # App categories section
    st.markdown("---")
    st.subheader("🗂️ App Categories")
   
    categories_col1, categories_col2, categories_col3, categories_col4 = st.columns(4)
   
    with categories_col1:
        st.markdown("""
        **🏖️ Retirement Planning**
        - US Retirement Calculator - 401(k), IRA, Social Security
        - Canadian Retirement Planner - RRSP, TFSA, CPP, OAS, GIS
        - India Retirement Calculator - EPF, NPS, PPF, tax optimization
        - Australia Retirement Calculator - Superannuation & Age Pension
        """)
   
    with categories_col2:
        st.markdown("""
        **🏠 Real Estate & Mortgages**
        - Mortgage Calculator - Payment schedules & analysis
        - Rental vs S&P 500 - Investment comparison tool
        """)




    with categories_col3:
        st.markdown("""
        **🧾 Tax & Financial Planning**
        - US Tax Estimator - Federal tax calculations
        - Interest Calculator - Compound interest with advanced features
        """)




    with categories_col4:
        st.markdown("""
        **📊 Stock Market & Investments**
        - Portfolio Dashboard - Track investments & performance
        - Unusual Options Activity - Options flow intelligence
        """)




    # Regional Focus section
    st.markdown("---")
    st.subheader("🌎 Regional Coverage")
   
    region_col1, region_col2, region_col3, region_col4 = st.columns(4)
   
    with region_col1:
        st.markdown("""
        **🇺🇸 United States Tools:**
        - US Tax Estimator (Federal taxes)
        - US Retirement Calculator (401k, IRA, Social Security)
        - Unusual Options Activity (US markets)
        - Portfolio Dashboard (US markets focused)
        """)
   
    with region_col2:
        st.markdown("""
        **🇨🇦 Canadian Tools:**
        - Canadian Retirement Planner (RRSP, TFSA, CPP, OAS)
        - Provincial tax calculations included
        - GIS and benefit optimization features
        """)




    with region_col3:
        st.markdown("""
        **🇮🇳 Indian Tools:**
        - India Retirement Calculator (EPF, NPS, PPF)
        - Indian tax system integration
        - Inflation-adjusted expense planning
        - Single & couple retirement scenarios
        """)




    with region_col4:
        st.markdown("""
        **🇦🇺 Australian Tools:**
        - Australia Retirement Calculator (Superannuation)
        - Age Pension assets & income testing
        - Investment property modeling
        - Spouse retirement planning
        """)




    # Special Features section
    st.markdown("---")
    st.subheader("⭐ Special Features by Region")
   
    features_col1, features_col2, features_col3, features_col4 = st.columns(4)
   
    with features_col1:
        st.markdown("""
        **🇺🇸 US-Specific Features:**
        - Social Security benefit calculations
        - Traditional vs Roth IRA comparisons
        - Federal tax bracket optimization
        - 401(k) employer matching scenarios
        - US mortgage market analysis
        - S&P 500 historical performance data
        - Real-time US options market data
        """)
   
    with features_col2:
        st.markdown("""
        **🇨🇦 Canada-Specific Features:**
        - CPP & OAS deferral strategies
        - RRSP vs TFSA optimization
        - Provincial tax variations
        - GIS clawback minimization
        """)
   
    with features_col3:
        st.markdown("""
        **🇮🇳 India-Specific Features:**
        - EPF withdrawal tax implications
        - NPS tax-free vs annuity options
        - PPF 15-year lock-in modeling
        - Medical inflation considerations
        """)




    with features_col4:
        st.markdown("""
        **🇦🇺 Australia-Specific Features:**
        - Superannuation contribution caps
        - Age Pension assets & income testing
        - Real vs nominal dollar projections
        - Investment property impact on pension
        """)




    # Usage tips section
    st.markdown("---")
    st.subheader("💡 Tips for Best Experience")
   
    tips_col1, tips_col2 = st.columns(2)
   
    with tips_col1:
        st.markdown("""
        **Getting Started:**
        - Each app opens in a new tab for easy navigation
        - Bookmark your favorites for quick access
        - Apps are optimized for both desktop and mobile
        - Try different scenarios to explore possibilities
        - Use retirement calculators first for long-term planning
        """)
   
    with tips_col2:
        st.markdown("""
        **Data Security:**
        - All calculations are performed client-side
        - No personal data is stored on servers
        - Apps are secure and privacy-focused
        - Your financial information stays private
        - Real-time market data is anonymized
        """)




    # Footer
    st.markdown("---")
    st.markdown("""
        ### 🌟 About This Hub
        This is a central hub for accessing my collection of financial web applications.
        Each application is designed to solve specific financial problems and provide valuable insights for different regions and financial situations.
       
        **📄 Updates:** Check back regularly for new applications and feature updates!
       
        **🤝 Feedback:** Your feedback helps improve these tools. Feel free to reach out with suggestions.
       
        **🌍 Expanding Coverage:** Now covering US, Canada, India, and Australia with specialized tools for retirement planning, real estate, investment analysis, and options trading!
    """)
   
    # Last updated info
    st.markdown("---")
    st.caption("Last updated: December 2024 | Apps are continuously maintained and improved | Now featuring Australian retirement planning and US options flow intelligence!")




if __name__ == "__main__":
    main()





