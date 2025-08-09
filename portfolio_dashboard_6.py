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
                # Use uploaded file
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {uploaded_file.name}")
            else:
                # Use default filename
                df = pd.read_csv(filename)
                st.success(f"✅ Successfully loaded {filename}")
            
            # Process the data
            grouped_df, raw_df = load_and_process_data(df)
            
            if grouped_df.empty:
                st.error("❌ No data available. Please check your CSV file format.")
                st.info("💡 Make sure the file contains the required columns.")
                return
                
            # Calculate key metrics
            total_value = grouped_df['Current Value'].sum()
            total_gain_loss = grouped_df["Total Gain/Loss Dollar"].sum()
            today_gain_loss = grouped_df["Today's Gain/Loss Dollar"].sum()
            
            # Calculate return percentages
            total_cost_basis = total_value - total_gain_loss
            total_return_pct = (total_gain_loss / total_cost_basis) * 100 if total_cost_basis > 0 else 0
            today_return_pct = (today_gain_loss / total_value) * 100 if total_value > 0 else 0
            
            # Success message
            st.success(f"✅ Successfully loaded {len(grouped_df)} positions from {filename}")
            st.markdown("---")
            
            # Sidebar with portfolio summary
            with st.sidebar:
                st.title("📈 Portfolio Summary")
                st.markdown("---")
                
                st.metric("📊 Total Positions", len(grouped_df))
                st.metric("🎯 Unique Symbols", grouped_df['Symbol'].nunique())
                
                if not raw_df.empty and 'Type' in raw_df.columns:
                    st.metric("💼 Account Types", raw_df['Type'].nunique())
                
                st.markdown("---")
                st.markdown("#### 🏆 Top 3 Holdings")
                for i, (_, row) in enumerate(grouped_df.head(3).iterrows(), 1):
                    st.markdown(f"**{i}. {row['Symbol']}**")
                    st.markdown(f"   💰 {format_currency(row['Current Value'])}")
                    st.markdown(f"   📊 {row['Percent of Portfolio']:.1f}%")
                
                st.markdown("---")
                st.markdown("#### ⚙️ Dashboard Info")
                st.info("💡 Use the tabs below to explore different views of your portfolio data.")
                st.markdown(f"📁 **Data Source:** {filename}")
            
            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "💰 Total Portfolio Value",
                    format_currency(total_value),
                    help="Total market value of all holdings"
                )
            
            with col2:
                delta_color = "normal" if total_gain_loss >= 0 else "inverse"
                st.metric(
                    "📈 Total Gain/Loss",
                    format_currency(total_gain_loss),
                    f"{total_return_pct:+.2f}%",
                    delta_color=delta_color,
                    help="Total unrealized gain/loss across all positions"
                )
            
            with col3:
                today_delta_color = "normal" if today_gain_loss >= 0 else "inverse"
                st.metric(
                    "📅 Today's Change",
                    format_currency(today_gain_loss),
                    f"{today_return_pct:+.2f}%",
                    delta_color=today_delta_color,
                    help="Today's change in portfolio value"
                )
            
            with col4:
                if not grouped_df.empty:
                    largest_holding = grouped_df.iloc[0]
                    st.metric(
                        "🏆 Largest Position",
                        largest_holding['Symbol'],
                        f"{largest_holding['Percent of Portfolio']:.1f}%",
                        help=f"Value: {format_currency(largest_holding['Current Value'])}"
                    )
            
            st.markdown("---")
            
            # Main content tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📋 Holdings", "🎯 Performance", "📈 Analytics", "🔮 Predictions"])
            
            with tab1:
                st.subheader("📊 Portfolio Overview")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Portfolio allocation pie chart
                    if len(grouped_df) > 0:
                        fig_pie = create_portfolio_pie_chart(grouped_df)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.warning("No data available for pie chart")
                
                with col2:
                    # Top holdings summary
                    st.markdown("#### 🔝 Top Holdings Summary")
                    top_holdings = grouped_df.head(10)[['Symbol', 'Current Value', 'Percent of Portfolio', "Total Gain/Loss Dollar"]]
                    
                    # Create a formatted display
                    for _, row in top_holdings.iterrows():
                        col_a, col_b, col_c = st.columns([2, 2, 2])
                        
                        with col_a:
                            st.markdown(f"**{row['Symbol']}**")
                        
                        with col_b:
                            st.markdown(f"{format_currency(row['Current Value'])}")
                            st.markdown(f"*{row['Percent of Portfolio']:.1f}% of portfolio*")
                        
                        with col_c:
                            gain_loss = row["Total Gain/Loss Dollar"]
                            if pd.notna(gain_loss):
                                color = "🟢" if gain_loss >= 0 else "🔴"
                                st.markdown(f"{color} {format_currency(gain_loss)}")
                            else:
                                st.markdown("➖ No data")
                        
                        st.markdown("---")
                
                # Additional overview metrics
                st.subheader("📈 Quick Stats")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_position_size = grouped_df['Current Value'].mean()
                    st.metric("Average Position", format_currency(avg_position_size))
                
                with col2:
                    median_position_size = grouped_df['Current Value'].median()
                    st.metric("Median Position", format_currency(median_position_size))
                
                with col3:
                    winners_count = len(grouped_df[grouped_df["Total Gain/Loss Dollar"] > 0])
                    st.metric("Winning Positions", winners_count)
                
                with col4:
                    losers_count = len(grouped_df[grouped_df["Total Gain/Loss Dollar"] < 0])
                    st.metric("Losing Positions", losers_count)
            
            with tab2:
                st.subheader("📋 Complete Holdings Analysis")
                
                # Add filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    min_value = st.number_input("Minimum Value ($)", min_value=0, value=0, step=1000)
                
                with col2:
                    if not raw_df.empty and 'Type' in raw_df.columns:
                        account_types = st.multiselect(
                            "Account Types",
                            options=raw_df['Type'].unique().tolist(),
                            default=raw_df['Type'].unique().tolist()
                        )
                    else:
                        account_types = []
                
                with col3:
                    show_only_gains = st.checkbox("Show only profitable positions")
                
                # Filter data based on selections
                filtered_df = grouped_df[grouped_df['Current Value'] >= min_value].copy()
                
                if show_only_gains:
                    filtered_df = filtered_df[filtered_df["Total Gain/Loss Dollar"] > 0]
                
                # Display metrics for filtered data
                if not filtered_df.empty:
                    st.markdown(f"**Showing {len(filtered_df)} positions** (Total value: {format_currency(filtered_df['Current Value'].sum())})")
                    
                    # Format data for display
                    display_df = filtered_df.copy()
                    
                    # Format columns for better display
                    display_df['Current Value'] = display_df['Current Value'].apply(format_currency)
                    display_df['Last Price'] = display_df['Last Price'].apply(format_currency)
                    display_df['Quantity'] = display_df['Quantity'].apply(lambda x: f"{x:.2f}")
                    display_df['Percent of Portfolio'] = display_df['Percent of Portfolio'].apply(lambda x: f"{x:.2f}%")
                    display_df["Today's Gain/Loss Dollar"] = display_df["Today's Gain/Loss Dollar"].apply(
                        lambda x: f"${x:+,.2f}" if pd.notna(x) else "$0.00"
                    )
                    display_df["Total Gain/Loss Dollar"] = display_df["Total Gain/Loss Dollar"].apply(
                        lambda x: f"${x:+,.2f}" if pd.notna(x) else "$0.00"
                    )
                    display_df['Total Return %'] = display_df['Total Return %'].apply(
                        lambda x: f"{x:+.2f}%" if pd.notna(x) else "0.00%"
                    )
                    
                    # Select columns to display
                    columns_to_show = ['Symbol', 'Description', 'Quantity', 'Last Price', 'Current Value', 
                                     'Percent of Portfolio', "Today's Gain/Loss Dollar", "Total Gain/Loss Dollar", 'Total Return %']
                    
                    st.dataframe(
                        display_df[columns_to_show],
                        column_config={
                            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                            "Description": st.column_config.TextColumn("Company", width="medium"),
                            "Quantity": st.column_config.TextColumn("Shares", width="small"),
                            "Last Price": st.column_config.TextColumn("Price", width="small"),
                            "Current Value": st.column_config.TextColumn("Value", width="medium"),
                            "Percent of Portfolio": st.column_config.TextColumn("% Portfolio", width="small"),
                            "Today's Gain/Loss Dollar": st.column_config.TextColumn("Today G/L", width="small"),
                            "Total Gain/Loss Dollar": st.column_config.TextColumn("Total G/L", width="small"),
                            "Total Return %": st.column_config.TextColumn("Return %", width="small")
                        },
                        hide_index=True,
                        use_container_width=True,
                        height=600
                    )
                else:
                    st.warning("No positions match your filter criteria.")
            
            with tab3:
                st.subheader("🎯 Performance Analysis")
                
                # Performance charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Holdings value bar chart
                    if len(grouped_df) > 0:
                        fig_bar = create_holdings_bar_chart(grouped_df)
                        st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    # Gain/Loss chart
                    if len(grouped_df) > 0:
                        fig_gain_loss = create_gain_loss_chart(grouped_df)
                        st.plotly_chart(fig_gain_loss, use_container_width=True)
                
                # Performance comparison chart (full width)
                if len(grouped_df) > 0:
                    fig_perf = create_performance_chart(grouped_df)
                    st.plotly_chart(fig_perf, use_container_width=True)
                
                # Winners and Losers analysis
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🚀 Top Performers")
                    winners = grouped_df[grouped_df["Total Gain/Loss Dollar"] > 0].head(5)
                    if not winners.empty:
                        for _, row in winners.iterrows():
                            return_pct = row['Total Return %'] if pd.notna(row['Total Return %']) else 0
                            st.success(f"**{row['Symbol']}**: +{format_currency(row['Total Gain/Loss Dollar'])} ({return_pct:+.2f}%)")
                    else:
                        st.info("No profitable positions found")
                
                with col2:
                    st.subheader("📉 Need Attention")
                    losers = grouped_df[grouped_df["Total Gain/Loss Dollar"] < 0].head(5)
                    if not losers.empty:
                        for _, row in losers.iterrows():
                            return_pct = row['Total Return %'] if pd.notna(row['Total Return %']) else 0
                            st.error(f"**{row['Symbol']}**: {format_currency(row['Total Gain/Loss Dollar'])} ({return_pct:+.2f}%)")
                    else:
                        st.info("No losing positions found")
            
            with tab4:
                st.subheader("📈 Advanced Portfolio Analytics")
                
                # Portfolio concentration analysis
                st.markdown("#### 🎯 Portfolio Concentration")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    top_5_concentration = grouped_df.head(5)['Percent of Portfolio'].sum()
                    st.metric("Top 5 Concentration", f"{top_5_concentration:.1f}%")
                
                with col2:
                    top_10_concentration = grouped_df.head(10)['Percent of Portfolio'].sum()
                    st.metric("Top 10 Concentration", f"{top_10_concentration:.1f}%")
                
                with col3:
                    # Herfindahl Index (measure of concentration)
                    position_weights = grouped_df['Percent of Portfolio'] / 100
                    herfindahl_index = (position_weights ** 2).sum()
                    st.metric("Herfindahl Index", f"{herfindahl_index:.3f}", help="Lower values indicate better diversification")
                
                # Risk analysis
                st.markdown("#### ⚠️ Risk Analysis")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    positions_over_5pct = len(grouped_df[grouped_df['Percent of Portfolio'] > 5])
                    st.metric("Positions > 5%", positions_over_5pct)
                
                with col2:
                    positions_over_10pct = len(grouped_df[grouped_df['Percent of Portfolio'] > 10])
                    st.metric("Positions > 10%", positions_over_10pct)
                
                with col3:
                    avg_position_size = grouped_df['Current Value'].mean()
                    st.metric("Avg Position Size", format_currency(avg_position_size))
                
                with col4:
                    portfolio_volatility = grouped_df['Percent of Portfolio'].std()
                    st.metric("Position Size Std Dev", f"{portfolio_volatility:.2f}%")
                
                # Performance distribution
                st.markdown("#### 📊 Performance Distribution")
                
                # Create performance histogram
                returns_data = grouped_df[grouped_df['Total Return %'].notna()]['Total Return %']
                if not returns_data.empty:
                    fig_hist = px.histogram(
                        x=returns_data,
                        nbins=20,
                        title="Distribution of Returns (%)",
                        labels={'x': 'Return (%)', 'y': 'Number of Positions'}
                    )
                    fig_hist.update_layout(height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Summary statistics
                st.markdown("#### 📋 Portfolio Statistics Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Value Statistics:**")
                    st.write(f"• Total Portfolio Value: {format_currency(total_value)}")
                    st.write(f"• Average Position: {format_currency(grouped_df['Current Value'].mean())}")
                    st.write(f"• Median Position: {format_currency(grouped_df['Current Value'].median())}")
                    st.write(f"• Largest Position: {format_currency(grouped_df['Current Value'].max())}")
                    st.write(f"• Smallest Position: {format_currency(grouped_df['Current Value'].min())}")
                
                with col2:
                    st.markdown("**Performance Statistics:**")
                    profitable_positions = len(grouped_df[grouped_df["Total Gain/Loss Dollar"] > 0])
                    total_positions = len(grouped_df)
                    win_rate = (profitable_positions / total_positions) * 100 if total_positions > 0 else 0
                    
                    st.write(f"• Win Rate: {win_rate:.1f}% ({profitable_positions}/{total_positions})")
                    st.write(f"• Total Gain/Loss: {format_currency(total_gain_loss)}")
                    st.write(f"• Total Return: {total_return_pct:+.2f}%")
                    st.write(f"• Today's P&L: {format_currency(today_gain_loss)}")
                    st.write(f"• Best Performer: {grouped_df.loc[grouped_df['Total Gain/Loss Dollar'].idxmax(), 'Symbol'] if not grouped_df.empty else 'N/A'}")
            
            with tab5:
                st.subheader("🔮 Future Predictions (2025-2030)")
                
                st.markdown("""
                <div style="background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                    <strong>⚠️ Disclaimer:</strong> These predictions are estimates based on historical performance and general market assumptions. 
                    They should not be used as financial advice. Actual future performance may vary significantly.
                </div>
                """, unsafe_allow_html=True)
                
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
                        value=min(15, len(predictions_df)),
                        help="Number of holdings to display"
                    )
                
                with col3:
                    show_charts = st.checkbox("Show Prediction Charts", value=True)
                
                # Filter to top N holdings by current value
                top_predictions = predictions_df.head(top_n)
                
                # Create display dataframe based on selected view
                if view_type == "Prices":
                    display_cols = ['Symbol', 'Current Price', '2025 Price', '2026 Price', '2027 Price', '2028 Price', '2029 Price', '2030 Price']
                    format_func = lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00"
                    title_suffix = "Predicted Stock Prices"
                    
                elif view_type == "Values":
                    display_cols = ['Symbol', 'Current Value', '2025 Value', '2026 Value', '2027 Value', '2028 Value', '2029 Value', '2030 Value']
                    format_func = lambda x: format_currency(x) if pd.notna(x) else "$0.00"
                    title_suffix = "Predicted Position Values"
                    
                else:  # Growth %
                    display_cols = ['Symbol'] + [f'{year} Growth' for year in [2025, 2026, 2027, 2028, 2029, 2030]]
                    format_func = lambda x: f"{x:+.1f}%" if pd.notna(x) else "0.0%"
                    title_suffix = "Predicted Growth Percentages"
                
                # Format the display dataframe
                display_df = top_predictions[display_cols].copy()
                
                # Apply formatting to all columns except Symbol
                for col in display_cols[1:]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(format_func)
                
                st.markdown(f"#### 📊 {title_suffix} - Top {top_n} Holdings")
                
                # Create column configuration for better display
                column_config = {}
                for col in display_cols:
                    if col == 'Symbol':
                        column_config[col] = st.column_config.TextColumn("Symbol", width="medium")
                    elif 'Current' in col:
                        column_config[col] = st.column_config.TextColumn("Current", width="medium")
                    else:
                        year = col.split()[0] if ' ' in col else col[:4]
                        column_config[col] = st.column_config.TextColumn(year, width="small")
                
                st.dataframe(
                    display_df,
                    column_config=column_config,
                    hide_index=True,
                    use_container_width=True,
                    height=600
                )
                
                # Show prediction charts if requested
                if show_charts:
                    st.markdown("---")
                    st.markdown("#### 📈 Prediction Visualizations")
                    
                    # Portfolio value over time chart
                    years = [2025, 2026, 2027, 2028, 2029, 2030]
                    total_values = []
                    
                    for year in years:
                        year_total = predictions_df[f'{year} Value'].sum()
                        total_values.append(year_total)
                    
                    fig_timeline = go.Figure()
                    fig_timeline.add_trace(go.Scatter(
                        x=years,
                        y=total_values,
                        mode='lines+markers',
                        name='Total Portfolio Value',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=8),
                        hovertemplate='<b>%{x}</b><br>Portfolio Value: $%{y:,.2f}<extra></extra>'
                    ))
                    
                    fig_timeline.update_layout(
                        title="Predicted Total Portfolio Value Over Time",
                        xaxis_title="Year",
                        yaxis_title="Portfolio Value ($)",
                        height=400,
                        xaxis=dict(tickmode='linear', tick0=2025, dtick=1)
                    )
                    
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Top holdings comparison chart
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Current vs 2030 comparison for top 10
                        top_10_pred = predictions_df.head(10)
                        
                        fig_comparison = go.Figure()
                        fig_comparison.add_trace(go.Bar(
                            x=top_10_pred['Symbol'],
                            y=top_10_pred['Current Value'],
                            name='Current Value',
                            marker_color='#1f77b4',
                            hovertemplate='<b>%{x}</b><br>Current: $%{y:,.2f}<extra></extra>'
                        ))
                        fig_comparison.add_trace(go.Bar(
                            x=top_10_pred['Symbol'],
                            y=top_10_pred['2030 Value'],
                            name='2030 Predicted',
                            marker_color='#ff7f0e',
                            hovertemplate='<b>%{x}</b><br>2030: $%{y:,.2f}<extra></extra>'
                        ))
                        
                        fig_comparison.update_layout(
                            title="Current vs 2030 Predicted Values",
                            xaxis_title="Symbol",
                            yaxis_title="Value ($)",
                            height=400,
                            xaxis_tickangle=-45,
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    with col2:
                        # Growth rate distribution
                        growth_2030 = predictions_df['2030 Growth'].dropna()
                        
                        fig_growth_dist = px.histogram(
                            x=growth_2030,
                            nbins=20,
                            title="Distribution of Predicted 2030 Growth Rates",
                            labels={'x': '2030 Growth Rate (%)', 'y': 'Number of Holdings'}
                        )
                        fig_growth_dist.update_layout(height=400)
                        
                        st.plotly_chart(fig_growth_dist, use_container_width=True)
                    
                    # Summary statistics
                    st.markdown("#### 📋 Prediction Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**2030 Predictions:**")
                        total_current = predictions_df['Current Value'].sum()
                        total_2030 = predictions_df['2030 Value'].sum()
                        total_growth = ((total_2030 / total_current - 1) * 100) if total_current > 0 else 0
                        
                        st.write(f"• Current Portfolio: {format_currency(total_current)}")
                        st.write(f"• 2030 Predicted: {format_currency(total_2030)}")
                        st.write(f"• Total Growth: {total_growth:+.1f}%")
                        st.write(f"• Annual Avg: {(total_growth/5):+.1f}%")
                    
                    with col2:
                        st.markdown("**Best Predicted Performers:**")
                        best_performers = predictions_df.nlargest(3, '2030 Growth')
                        for _, row in best_performers.iterrows():
                            st.write(f"• {row['Symbol']}: {row['2030 Growth']:+.1f}%")
                    
                    with col3:
                        st.markdown("**Risk Assessment:**")
                        high_growth_count = len(predictions_df[predictions_df['2030 Growth'] > 100])
                        moderate_growth_count = len(predictions_df[(predictions_df['2030 Growth'] > 50) & (predictions_df['2030 Growth'] <= 100)])
                        conservative_count = len(predictions_df[predictions_df['2030 Growth'] <= 50])
                        
                        st.write(f"• High Growth (>100%): {high_growth_count}")
                        st.write(f"• Moderate (50-100%): {moderate_growth_count}")
                        st.write(f"• Conservative (<50%): {conservative_count}")
        
        except FileNotFoundError:
            st.error(f"❌ File not found: {filename}")
            st.info("Please check if the file exists in the correct location.")
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted.")
    else:
        # Show instructions when no file is loaded
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