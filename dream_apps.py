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
    </style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.title("🎯 My Applications Hub")
    st.markdown("### Welcome to my collection of web applications!")
    st.markdown("Explore powerful financial tools designed to help you make informed decisions about your money.")
    st.markdown("---")

    # Apps section
    st.subheader("📱 Available Applications")

    # Create layout for apps - using 3 columns now to accommodate the third app
    col1, col2, col3 = st.columns(3)

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
        # Retirement Calculator App
        st.markdown("""
            <div class="app-card">
                <div class="app-title">💰 Retirement Calculator</div>
                <div class="app-description">
                    Plan your financial future with this comprehensive retirement planning tool.
                    Calculate savings projections and retirement income scenarios.
                </div>
                <div class="app-features">
                    <strong>Features:</strong><br>
                    • Savings projections<br>
                    • Required contributions<br>
                    • Income scenarios<br>
                    • Goal planning
                </div>
                <a href="https://czsnconjpbqmhpvhjncydg.streamlit.app/" target="_blank" class="launch-btn">
                    🚀 Launch App
                </a>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        # Tax Estimator App
        st.markdown("""
            <div class="app-card">
                <div class="app-title">🧾 Tax Estimator</div>
                <div class="app-description">
                    Estimate your federal income taxes for 2024 tax year with this comprehensive calculator.
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

    # Additional information section
    st.markdown("---")
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Apps", "3", "1")
    
    with col2:
        st.metric("Categories", "Financial Tools", "")
    
    with col3:
        st.metric("Latest Addition", "Tax Estimator", "")

    # App categories section
    st.markdown("---")
    st.subheader("🏷️ App Categories")
    
    categories_col1, categories_col2 = st.columns(2)
    
    with categories_col1:
        st.markdown("""
        **💼 Investment & Portfolio Management**
        - Portfolio Dashboard - Track and analyze your investments
        """)
    
    with categories_col2:
        st.markdown("""
        **📈 Financial Planning**
        - Retirement Calculator - Plan your retirement savings
        - Tax Estimator - Calculate your annual tax liability
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
        """)
    
    with tips_col2:
        st.markdown("""
        **Data Security:**
        - All calculations are performed client-side
        - No personal data is stored on servers
        - Apps are secure and privacy-focused
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
        ### 🌟 About This Hub
        This is a central hub for accessing my collection of financial web applications. 
        Each application is designed to solve specific financial problems and provide valuable insights to users.
        
        **🔄 Updates:** Check back regularly for new applications and feature updates!
        
        **🤝 Feedback:** Your feedback helps improve these tools. Feel free to reach out with suggestions.
    """)
    
    # Last updated info
    st.markdown("---")
    st.caption("Last updated: December 2024 | Apps are continuously maintained and improved")

if __name__ == "__main__":
    main()