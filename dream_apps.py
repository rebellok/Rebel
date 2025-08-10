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
        }
        .app-title {
            color: #0066cc;
            font-size: 24px;
            margin-bottom: 10px;
        }
        .app-description {
            color: #666;
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.title("🎯 My Applications Hub")
    st.markdown("### Welcome to my collection of web applications!")
    st.markdown("---")

    # Apps section
    st.subheader("📱 Available Applications")

    # Create columns for apps
    col1, col2 = st.columns(2)

    with col1:
        # Portfolio Dashboard App
        st.markdown("""
            <div class="app-card">
                <div class="app-title">📊 Portfolio Dashboard</div>
                <div class="app-description">
                    A comprehensive dashboard for tracking and analyzing investment portfolios.
                    Features include performance tracking, predictions, and detailed analytics.
                </div>
                <a href="https://e5uovfuhcfumf9tudf8xbz.streamlit.app/" target="_blank">
                    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit">
                </a>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        # Placeholder for future app
        st.markdown("""
            <div class="app-card">
                <div class="app-title">🔜 Coming Soon</div>
                <div class="app-description">
                    More exciting applications are under development. 
                    Stay tuned for updates!
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
        ### 🌟 About
        This is a central hub for accessing my various web applications. 
        Each application is designed to solve specific problems and provide value to users.
        Check back regularly for new additions!
    """)

if __name__ == "__main__":
    main()