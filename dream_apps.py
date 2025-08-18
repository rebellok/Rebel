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
    </style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.title("🎯 My Applications Hub")
    st.markdown("### Welcome to my collection of web applications!")
    st.markdown("Explore powerful financial tools designed to help you make informed decisions about your money.")
    st.markdown("---")

    # Portfolio & Investment Tools Section
    st.markdown('<div class="section-header">📊 Portfolio & Investment Tools</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
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

    with col2:
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

    # Retirement Planning Section
    st.markdown('<div class="section-header">🏖️ Retirement Planning Tools</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
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

    with col4:
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
        # Placeholder for future app
        st.markdown("""
            <div class="app-card" style="opacity: 0.7;">
                <div class="app-title">🔮 Coming Soon</div>
                <div class="app-description">
                    More financial planning tools are in development. Stay tuned for additional calculators 
                    and planning utilities to expand your financial toolkit.
                </div>
                <div class="app-features">
                    <strong>Planned Features:</strong><br>
                    • Mortgage calculators<br>
                    • Debt payoff planners<br>
                    • Investment comparisons<br>
                    • Budget trackers
                </div>
                <div style="padding: 10px 20px; background-color: #ccc; color: #666; border-radius: 5px; font-weight: bold; text-align: center;">
                    🚧 In Development
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Quick stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Apps", "5", "1")
    
    with col2:
        st.metric("Countries Supported", "US + Canada", "🌎")
    
    with col3:
        st.metric("Categories", "3", "")
    
    with col4:
        st.metric("Latest Addition", "Canadian Retirement", "🇨🇦")

    # App categories section
    st.markdown("---")
    st.subheader("🏷️ App Categories")
    
    categories_col1, categories_col2, categories_col3 = st.columns(3)
    
    with categories_col1:
        st.markdown("""
        **💼 Investment & Portfolio**
        - Portfolio Dashboard - Track investments
        - Interest Calculator - Compound interest with advanced features
        """)
    
    with categories_col2:
        st.markdown("""
        **🏖️ Retirement Planning**
        - US Retirement Calculator - 401(k), IRA, Social Security
        - Canadian Retirement Planner - RRSP, TFSA, CPP, OAS, GIS
        """)

    with categories_col3:
        st.markdown("""
        **🧾 Tax & Planning**
        - US Tax Estimator - Federal tax calculations
        - More tools coming soon!
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
        """)
    
    with tips_col2:
        st.markdown("""
        **Data Security:**
        - All calculations are performed client-side
        - No personal data is stored on servers
        - Apps are secure and privacy-focused
        - Your financial information stays private
        """)

    # Regional Focus section
    st.markdown("---")
    st.subheader("🌎 Regional Coverage")
    
    region_col1, region_col2 = st.columns(2)
    
    with region_col1:
        st.markdown("""
        **🇺🇸 United States Tools:**
        - US Tax Estimator (Federal taxes)
        - US Retirement Calculator (401k, IRA, Social Security)
        - Portfolio Dashboard (US markets focused)
        """)
    
    with region_col2:
        st.markdown("""
        **🇨🇦 Canadian Tools:**
        - Canadian Retirement Planner (RRSP, TFSA, CPP, OAS)
        - Provincial tax calculations included
        - GIS and benefit optimization features
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
        ### 🌟 About This Hub
        This is a central hub for accessing my collection of financial web applications. 
        Each application is designed to solve specific financial problems and provide valuable insights for different regions and financial situations.
        
        **📄 Updates:** Check back regularly for new applications and feature updates!
        
        **🤝 Feedback:** Your feedback helps improve these tools. Feel free to reach out with suggestions.
        
        **🌍 Expanding Coverage:** Working to add more regional-specific tools and calculators.
    """)
    
    # Last updated info
    st.markdown("---")
    st.caption("Last updated: December 2024 | Apps are continuously maintained and improved | Now featuring Canadian retirement planning tools!")

if __name__ == "__main__":
    main()