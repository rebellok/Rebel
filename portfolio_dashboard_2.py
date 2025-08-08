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
def load_and_process_data(filename):
    """Load and process portfolio data"""
    try:
        # Load your CSV file
        df = pd.read_csv(filename)
        
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
        
    except FileNotFoundError:
        st.error(f"Portfolio CSV file '{filename}' not found. Please ensure the file is in the same directory as this script.")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
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

def main():
    # Header
    st.title("📊 Portfolio Dashboard")
    st.markdown("### Professional Portfolio Analysis & Performance Tracking")
    st.markdown("---")
    
    # Generate default filename with current date
    current_date = datetime.now()
    default_filename = f"Portfolio_Positions_{current_date.strftime('%b-%d-%Y')}.csv"
    
    # File input section
    st.markdown("#### 📁 Data Source")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        filename = st.text_input(
            "CSV Filename",
            value=default_filename,
            help="Enter the name of your portfolio CSV file"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        load_button = st.button("📊 Load Data", type="primary")
    
    # Only proceed if user clicks load button or if filename is provided
    if load_button or filename:
        # Load data
        with st.spinner(f"Loading portfolio data from {filename}..."):
            grouped_df, raw_df = load_and_process_data(filename)
    
        if grouped_df.empty:
            st.error("❌ No data available. Please check your CSV file and try again.")
            st.info(f"💡 Make sure '{filename}' is in the same directory as this script.")
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
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📋 Holdings", "🎯 Performance", "📈 Analytics"])
    
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
        # Continue with remaining tabs (all content gets indented to be within the load_button conditional)
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