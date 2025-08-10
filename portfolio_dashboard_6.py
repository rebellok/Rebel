import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1e88e5;
    }
    .positive {
        color: #4caf50;
    }
    .negative {
        color: #f44336;
    }
    .big-font {
        font-size: 2rem !important;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data(data):
    """Load and process portfolio data"""
    try:
        # Use the DataFrame directly instead of loading from file
        df = data.copy()
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Convert string currency values to numeric
        currency_columns = ['Current Value', 'Last Price', "Today's Gain/Loss Dollar", "Total Gain/Loss Dollar"]
        for col in currency_columns:
            if col in df.columns:
                # Remove currency symbols and convert to numeric
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').str.replace('+', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert quantity to numeric
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        
        # Convert percentage columns
        if 'Percent Of Account' in df.columns:
            df['Percent Of Account'] = df['Percent Of Account'].astype(str).str.replace('%', '')
            df['Percent Of Account'] = pd.to_numeric(df['Percent Of Account'], errors='coerce')
        
        # Fix the Type column aggregation to handle NaN values
        def safe_join_types(x):
            if 'Type' not in df.columns:
                return 'N/A'
            # Convert to string and filter out NaN values
            unique_types = x.dropna().astype(str).unique()
            return ', '.join(unique_types) if len(unique_types) > 0 else 'N/A'
        
        # Group by symbol and aggregate
        grouped_df = df.groupby('Symbol').agg({
            'Current Value': 'sum',
            'Quantity': 'sum',
            'Last Price': 'first',  # Use first price as reference
            "Today's Gain/Loss Dollar": 'sum',
            "Total Gain/Loss Dollar": 'sum',
            'Description': 'first',
            'Type': safe_join_types
        }).reset_index()
        
        # Calculate additional metrics
        total_portfolio_value = grouped_df['Current Value'].sum()
        grouped_df['Percent of Portfolio'] = (grouped_df['Current Value'] / total_portfolio_value * 100).round(2)
        
        # Calculate total return percentage
        grouped_df['Cost Basis'] = grouped_df['Current Value'] - grouped_df["Total Gain/Loss Dollar"]
        grouped_df['Total Return %'] = np.where(
            grouped_df['Cost Basis'] != 0,
            (grouped_df["Total Gain/Loss Dollar"] / grouped_df['Cost Basis']) * 100,
            0
        ).round(2)
        
        # Sort by current value (descending)
        grouped_df = grouped_df.sort_values('Current Value', ascending=False)
        
        return grouped_df, df
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def format_currency(value):
    """Format numbers as currency"""
    if pd.isna(value):
        return "$0.00"
    
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:.0f}K"
    else:
        return f"${value:,.2f}"

def format_percentage(value):
    """Format percentage with + or - sign"""
    if pd.isna(value):
        return "0.00%"
    return f"{value:+.2f}%"

def create_portfolio_pie_chart(df):
    """Create portfolio allocation pie chart"""
    top_10 = df.head(10)
    others_value = df.iloc[10:]['Current Value'].sum() if len(df) > 10 else 0
    
    if others_value > 0:
        pie_data = pd.concat([
            top_10[['Symbol', 'Current Value']],
            pd.DataFrame({'Symbol': ['Others'], 'Current Value': [others_value]})
        ])
    else:
        pie_data = top_10[['Symbol', 'Current Value']]
    
    # Calculate percentages
    pie_data['Percentage'] = (pie_data['Current Value'] / pie_data['Current Value'].sum() * 100).round(1)
    
    fig = px.pie(
        pie_data, 
        values='Current Value', 
        names='Symbol',
        title="Portfolio Allocation (Top 10 + Others)",
        color_discrete_sequence=px.colors.qualitative.Set3,
        hover_data=['Percentage']
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        showlegend=True,
        height=500,
        font=dict(size=12),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def create_holdings_bar_chart(df):
    """Create holdings bar chart"""
    top_15 = df.head(15)
    
    fig = go.Figure()
    
    # Add current value bars
    fig.add_trace(go.Bar(
        x=top_15['Symbol'],
        y=top_15['Current Value'],
        name='Current Value',
        marker_color='#1f77b4',
        hovertemplate='<b>%{x}</b><br>Current Value: $%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Top 15 Holdings by Value",
        xaxis_title="Symbol",
        yaxis_title="Value ($)",
        height=500,
        showlegend=False,
        xaxis_tickangle=-45
    )
    
    return fig

def create_performance_chart(df):
    """Create performance comparison chart"""
    top_10 = df.head(10)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Current Value bars
    fig.add_trace(
        go.Bar(
            x=top_10['Symbol'],
            y=top_10['Current Value'],
            name='Current Value',
            marker_color='#1f77b4',
            hovertemplate='<b>%{x}</b><br>Current Value: $%{y:,.2f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Total Gain/Loss line
    colors = ['green' if x >= 0 else 'red' for x in top_10["Total Gain/Loss Dollar"]]
    fig.add_trace(
        go.Scatter(
            x=top_10['Symbol'],
            y=top_10["Total Gain/Loss Dollar"],
            mode='lines+markers',
            name='Total Gain/Loss',
            line=dict(color='orange', width=3),
            marker=dict(size=8, color=colors),
            hovertemplate='<b>%{x}</b><br>Gain/Loss: $%{y:,.2f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Symbol", tickangle=-45)
    fig.update_yaxes(title_text="Current Value ($)", secondary_y=False)
    fig.update_yaxes(title_text="Gain/Loss ($)", secondary_y=True)
    
    fig.update_layout(
        title="Performance Overview - Top 10 Holdings",
        height=500,
        hovermode='x'
    )
    
    return fig

def create_gain_loss_chart(df):
    """Create gain/loss waterfall-style chart"""
    top_10 = df.head(10)
    
    # Separate winners and losers
    winners = top_10[top_10["Total Gain/Loss Dollar"] > 0]
    losers = top_10[top_10["Total Gain/Loss Dollar"] < 0]
    
    fig = go.Figure()
    
    if not winners.empty:
        fig.add_trace(go.Bar(
            x=winners['Symbol'],
            y=winners["Total Gain/Loss Dollar"],
            name='Gains',
            marker_color='green',
            hovertemplate='<b>%{x}</b><br>Gain: +$%{y:,.2f}<extra></extra>'
        ))
    
    if not losers.empty:
        fig.add_trace(go.Bar(
            x=losers['Symbol'],
            y=losers["Total Gain/Loss Dollar"],
            name='Losses',
            marker_color='red',
            hovertemplate='<b>%{x}</b><br>Loss: $%{y:,.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Gains vs Losses - Top 10 Holdings",
        xaxis_title="Symbol",
        yaxis_title="Gain/Loss ($)",
        height=400,
        xaxis_tickangle=-45,
        showlegend=True
    )
    
    return fig

def calculate_predictions(grouped_df):
    """Calculate predicted prices and values for future years"""
    current_year = 2025
    prediction_years = [2025, 2026, 2027, 2028, 2029, 2030]
    
    predictions = []
    
    for _, row in grouped_df.iterrows():
        symbol = row['Symbol']
        current_price = row['Last Price']
        current_value = row['Current Value']
        quantity = row['Quantity']
        total_return_pct = row['Total Return %'] if pd.notna(row['Total Return %']) else 0
        
        # Calculate predicted growth rate based on current performance
        # This is a simplified model - in reality you'd want to use more sophisticated methods
        base_growth_rate = 0.07  # 7% baseline market growth
        
        # Adjust growth rate based on current performance
        if total_return_pct > 20:
            growth_rate = 0.12  # High performers get 12% growth
        elif total_return_pct > 0:
            growth_rate = 0.10  # Moderate performers get 10% growth
        elif total_return_pct > -10:
            growth_rate = 0.05  # Slight underperformers get 5% growth
        else:
            growth_rate = 0.03  # Poor performers get 3% growth
        
        # Add some sector-based adjustments (simplified)
        symbol_upper = symbol.upper()
        if any(tech in symbol_upper for tech in ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META']):
            growth_rate += 0.02  # Tech premium
        elif any(util in symbol_upper for util in ['XLU', 'UTIL', 'PG', 'JNJ', 'KO']):
            growth_rate -= 0.02  # Conservative utilities/consumer staples
        
        # Cap growth rates at reasonable levels
        growth_rate = min(max(growth_rate, 0.01), 0.15)  # Between 1% and 15%
        
        row_predictions = {'Symbol': symbol, 'Current Price': current_price, 'Current Value': current_value}
        
        for year in prediction_years:
            years_ahead = year - current_year
            
            # Add some volatility/uncertainty - longer predictions are less reliable
            uncertainty_factor = 1 + (years_ahead * 0.02 * np.random.uniform(-1, 1))
            adjusted_growth = growth_rate * uncertainty_factor
            
            predicted_price = current_price * ((1 + adjusted_growth) ** years_ahead)
            predicted_value = quantity * predicted_price
            
            row_predictions[f'{year} Price'] = predicted_price
            row_predictions[f'{year} Value'] = predicted_value
            row_predictions[f'{year} Growth'] = ((predicted_price / current_price - 1) * 100)
        
        predictions.append(row_predictions)
    
    return pd.DataFrame(predictions)

# Initialize session states
if 'predictions_slider' not in st.session_state:
    st.session_state.predictions_slider = 15

def main():
    # Header
    st.title("📊 Portfolio Dashboard")
    st.markdown("### Professional Portfolio Analysis & Performance Tracking")
    st.markdown("---")
    
    # File input section
    st.markdown("#### 📁 Data Source")
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Default filename with current date
        current_date = datetime.now()
        default_filename = f"Portfolio_Positions_{current_date.strftime('%b-%d-%Y')}.csv"
        filename = st.text_input(
            "Default CSV Filename",
            value=default_filename,
            help="Default filename using current date"
        )

    with col2:
        uploaded_file = st.file_uploader(
            "Or Upload/Drop CSV File",
            type=['csv'],
            help="Drag & drop or browse for your portfolio CSV file"
        )

    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        load_button = st.button("📊 Load Data", type="primary")

    # Handle file loading logic
    if load_button:
        try:
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {uploaded_file.name}")
            else:
                df = pd.read_csv(filename)
                st.success(f"✅ Successfully loaded {filename}")

            grouped_df, raw_df = load_and_process_data(df)

            if grouped_df.empty:
                st.error("❌ No data available. Please check your CSV file format.")
                st.info("💡 Make sure the file contains the required columns.")
                return

            # Store in session state
            st.session_state['grouped_df'] = grouped_df
            st.session_state['raw_df'] = raw_df
            st.session_state['data_loaded'] = True

        except FileNotFoundError:
            st.error(f"❌ File not found: {filename}")
            st.info("Please check if the file exists in the correct location.")
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted.")

    # Check if data is loaded in session state
    elif st.session_state.get('data_loaded', False):
        grouped_df = st.session_state['grouped_df']
        raw_df = st.session_state['raw_df']

    # If data is loaded, show the dashboard tabs
    if st.session_state.get('data_loaded', False):
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "💼 Holdings",
            "📈 Performance",
            "💰 Gains/Losses",
            "🔮 Predictions"
        ])

        # Overview Tab
        with tab1:
            st.subheader("Portfolio Overview")
            
            # Calculate key metrics
            total_value = grouped_df['Current Value'].sum()
            total_gain_loss = grouped_df['Total Gain/Loss Dollar'].sum()
            today_gain_loss = grouped_df["Today's Gain/Loss Dollar"].sum()
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Portfolio Value",
                    format_currency(total_value),
                    format_currency(today_gain_loss)
                )
            
            with col2:
                st.metric(
                    "Total Gain/Loss",
                    format_currency(total_gain_loss),
                    format_percentage((total_gain_loss / (total_value - total_gain_loss)) * 100)
                )
            
            with col3:
                st.metric(
                    "Today's Gain/Loss",
                    format_currency(today_gain_loss),
                    format_percentage((today_gain_loss / total_value) * 100)
                )
            
            # Display charts
            st.plotly_chart(create_portfolio_pie_chart(grouped_df), use_container_width=True)
            st.plotly_chart(create_holdings_bar_chart(grouped_df), use_container_width=True)

        # Holdings Tab
        with tab2:
            st.subheader("Portfolio Holdings")
            
            # Display holdings table
            st.dataframe(
                grouped_df[[
                    'Symbol', 'Description', 'Type', 'Quantity', 'Last Price',
                    'Current Value', 'Percent of Portfolio', 'Total Return %'
                ]].style.format({
                    'Last Price': '${:,.2f}',
                    'Current Value': '${:,.2f}',
                    'Percent of Portfolio': '{:.2f}%',
                    'Total Return %': '{:+.2f}%'
                }),
                use_container_width=True,
                height=600
            )

        # Performance Tab
        with tab3:
            st.subheader("Performance Analysis")
            st.plotly_chart(create_performance_chart(grouped_df), use_container_width=True)

        # Gains/Losses Tab
        with tab4:
            st.subheader("Gains and Losses")
            st.plotly_chart(create_gain_loss_chart(grouped_df), use_container_width=True)

        # Predictions Tab
        with tab5:
            st.subheader("🔮 Future Predictions (2025-2030)")
            
            # Calculate predictions
            with st.spinner("Calculating future predictions..."):
                predictions_df = calculate_predictions(grouped_df)

            # Display options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                view_type = st.selectbox(
                    "View Type",
                    options=["Prices", "Values", "Growth %"],
                    help="Choose what to display in the prediction table"
                )
            
            with col2:
                top_n = st.slider(
                    "Show Top N Holdings",
                    min_value=5,
                    max_value=min(50, len(predictions_df)),
                    value=15,
                    help="Number of holdings to display"
                )
            
            with col3:
                show_charts = st.checkbox("Show Prediction Charts", value=True)
                
            # Filter and display predictions based on selected view
            top_predictions = predictions_df.head(top_n)
            
            # Create display dataframe based on selected view
            if view_type == "Prices":
                display_cols = ['Symbol', 'Current Price', '2025 Price', '2026 Price', 
                              '2027 Price', '2028 Price', '2029 Price', '2030 Price']
                format_func = lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00"
                title_suffix = "Predicted Stock Prices"
            
            elif view_type == "Values":
                display_cols = ['Symbol', 'Current Value', '2025 Value', '2026 Value',
                              '2027 Value', '2028 Value', '2029 Value', '2030 Value']
                format_func = lambda x: format_currency(x) if pd.notna(x) else "$0.00"
                title_suffix = "Predicted Position Values"
            
            else:  # Growth %
                display_cols = ['Symbol'] + [f'{year} Growth' for year in range(2025, 2031)]
                format_func = lambda x: f"{x:+.1f}%" if pd.notna(x) else "0.0%"
                title_suffix = "Predicted Growth Percentages"

            # Format the display dataframe
            display_df = top_predictions[display_cols].copy()
            for col in display_cols[1:]:
                display_df[col] = display_df[col].apply(format_func)

            st.markdown(f"#### 📊 {title_suffix} - Top {top_n} Holdings")
            st.dataframe(display_df, use_container_width=True)

    # Show instructions only if no data is loaded
    else:
        st.info("""
        👋 **Welcome to Portfolio Dashboard!**
        
        To get started:
        1. Enter your CSV filename above (default uses today's date)
        2. Click the "📊 Load Data" button
        3. Explore your portfolio analytics across the different tabs
        
        **Expected CSV Format:**
        Your CSV should contain columns like:
        - Symbol, Description, Quantity, Last Price, Current Value
        - Today's Gain/Loss Dollar, Total Gain/Loss Dollar
        - Type (optional), Percent Of Account (optional)
        """)
        
        st.markdown("---")
        st.markdown("### 🔧 Sample File Format")
        sample_data = {
            'Symbol': ['AAPL', 'GOOGL', 'MSFT'],
            'Description': ['Apple Inc.', 'Alphabet Inc.', 'Microsoft Corporation'],
            'Quantity': [100, 50, 75],
            'Last Price': [150.00, 2500.00, 300.00],
            'Current Value': [15000.00, 125000.00, 22500.00],
            "Today's Gain/Loss Dollar": [200.00, -500.00, 300.00],
            "Total Gain/Loss Dollar": [2000.00, 15000.00, -1500.00]
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

if __name__ == "__main__":
    main()