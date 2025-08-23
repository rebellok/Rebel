import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Options Flow Intelligence",
    page_icon="🎯📈",
    layout="wide"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .flow-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-high { background-color: #dc3545; color: white; }
    .alert-medium { background-color: #fd7e14; color: white; }
    .alert-low { background-color: #28a745; color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="flow-header"><h1>🎯📈 Options Flow Intelligence Center</h1><p>Real-time unusual options activity detection via Yahoo Finance (FREE)</p></div>', unsafe_allow_html=True)

class YahooOptionsScanner:
    def __init__(self):
        # Popular symbols to scan for options activity
        self.popular_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 
            'BRK-B', 'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'MA', 'CVX', 
            'ABBV', 'PFE', 'KO', 'BAC', 'PEP', 'TMO', 'COST', 'AVGO',
            'SPY', 'QQQ', 'IWM', 'VTI', 'GLD', 'TLT', 'EEM', 'XLF', 'DIA'
        ]
        
        # ETFs and popular options symbols
        self.etf_symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'EEM', 'GLD', 'SLV', 'TLT', 'HYG']
        
        # High volume options symbols
        self.high_vol_symbols = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'SPY', 'QQQ', 'META', 'AMZN', 'MSFT']
    
    def get_options_data_for_symbol(self, symbol):
        """Get options data for a specific symbol using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current stock info
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            
            if current_price == 0:
                # Fallback to fast_info for current price
                try:
                    current_price = ticker.fast_info.get('lastPrice', 0)
                except:
                    current_price = 100  # Default fallback
            
            # Get options expiration dates
            exp_dates = ticker.options
            if not exp_dates:
                return []
            
            all_options = []
            
            # Process first few expiration dates to avoid timeout
            for exp_date in exp_dates[:4]:  # Limit to first 4 expirations
                try:
                    # Get options chain for this expiration
                    options_chain = ticker.option_chain(exp_date)
                    
                    # Process calls
                    if not options_chain.calls.empty:
                        calls = options_chain.calls.copy()
                        calls['option_type'] = 'Call'
                        calls['expiration'] = exp_date
                        calls['symbol'] = symbol
                        calls['underlying_price'] = current_price
                        all_options.append(calls)
                    
                    # Process puts
                    if not options_chain.puts.empty:
                        puts = options_chain.puts.copy()
                        puts['option_type'] = 'Put'
                        puts['expiration'] = exp_date
                        puts['symbol'] = symbol
                        puts['underlying_price'] = current_price
                        all_options.append(puts)
                        
                except Exception as e:
                    continue
            
            return all_options
            
        except Exception as e:
            st.warning(f"Failed to get data for {symbol}: {str(e)}")
            return []
    
    def calculate_unusual_score(self, volume, open_interest, last_price, strike, underlying_price, option_type):
        """Calculate unusual activity score"""
        if not all([volume, last_price]) or volume < 1:
            return 0
        
        # Basic components
        vol_score = min(volume / 100, 50) * 0.4
        
        # Volume to Open Interest ratio
        vol_oi_ratio = volume / max(open_interest, 1) if open_interest > 0 else volume
        vol_oi_score = min(vol_oi_ratio * 5, 30) * 0.3
        
        # Premium value component
        premium_value = volume * last_price * 100
        premium_score = min(premium_value / 50000, 25) * 0.2
        
        # Moneyness component (closer to ATM gets higher score)
        moneyness_score = 0
        if underlying_price > 0 and strike > 0:
            if option_type == 'Call':
                ratio = strike / underlying_price
            else:  # Put
                ratio = underlying_price / strike
            
            # Score based on how close to ATM (1.0)
            distance_from_atm = abs(1.0 - ratio)
            moneyness_score = max(5 - (distance_from_atm * 20), 0)
        
        total_score = vol_score + vol_oi_score + premium_score + moneyness_score
        
        return round(total_score, 1)
    
    def classify_moneyness(self, strike, underlying_price, option_type, threshold=0.05):
        """Classify option moneyness"""
        if underlying_price <= 0 or strike <= 0:
            return "Unknown"
        
        if option_type == 'Call':
            ratio = strike / underlying_price
            if ratio > (1 + threshold):
                return "OTM"
            elif ratio < (1 - threshold):
                return "ITM"
            else:
                return "ATM"
        else:  # Put
            ratio = underlying_price / strike
            if ratio > (1 + threshold):
                return "OTM"
            elif ratio < (1 - threshold):
                return "ITM"
            else:
                return "ATM"
    
    def process_options_data(self, raw_options_data):
        """Process raw options data into standardized format"""
        processed_data = []
        
        for options_df in raw_options_data:
            if options_df.empty:
                continue
                
            for _, row in options_df.iterrows():
                try:
                    volume = row.get('volume', 0) or 0
                    open_interest = row.get('openInterest', 0) or 0
                    last_price = row.get('lastPrice', 0) or 0
                    strike = row.get('strike', 0) or 0
                    
                    # Skip if no meaningful data
                    if volume < 1 or last_price <= 0:
                        continue
                    
                    symbol = row.get('symbol', '')
                    option_type = row.get('option_type', '')
                    expiration = row.get('expiration', '')
                    underlying_price = row.get('underlying_price', 0)
                    
                    # Calculate metrics
                    premium_value = volume * last_price * 100
                    vol_oi_ratio = volume / max(open_interest, 1) if open_interest > 0 else volume
                    
                    # Calculate unusual score
                    unusual_score = self.calculate_unusual_score(
                        volume, open_interest, last_price, strike, 
                        underlying_price, option_type
                    )
                    
                    # Classify moneyness
                    moneyness = self.classify_moneyness(strike, underlying_price, option_type)
                    
                    processed_option = {
                        'symbol': symbol,
                        'option_type': option_type,
                        'strike': strike,
                        'expiration': expiration,
                        'volume': int(volume),
                        'price': round(last_price, 2),
                        'open_interest': int(open_interest),
                        'premium_value': round(premium_value, 2),
                        'vol_oi_ratio': round(vol_oi_ratio, 2),
                        'unusual_score': unusual_score,
                        'bid': row.get('bid', last_price * 0.95) or last_price * 0.95,
                        'ask': row.get('ask', last_price * 1.05) or last_price * 1.05,
                        'implied_vol': row.get('impliedVolatility', 0) or 0,
                        'moneyness': moneyness,
                        'underlying_price': underlying_price,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    }
                    
                    processed_data.append(processed_option)
                    
                except Exception as e:
                    continue
        
        if processed_data:
            df = pd.DataFrame(processed_data)
            return df.sort_values('unusual_score', ascending=False)
        else:
            return pd.DataFrame()
    
    def scan_unusual_options_activity(self, symbols_to_scan, max_workers=5):
        """Scan for unusual options activity across multiple symbols using threading"""
        if not symbols_to_scan:
            symbols_to_scan = self.popular_symbols[:12]  # Limit to prevent timeout
        
        all_options_data = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def fetch_symbol_data(symbol):
            return symbol, self.get_options_data_for_symbol(symbol)
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(fetch_symbol_data, symbol): symbol 
                for symbol in symbols_to_scan
            }
            
            completed = 0
            total = len(symbols_to_scan)
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                completed += 1
                symbol = future_to_symbol[future]
                
                # Update progress
                progress_bar.progress(completed / total)
                status_text.text(f"Scanned {symbol}... ({completed}/{total})")
                
                try:
                    symbol, symbol_options = future.result()
                    if symbol_options:
                        all_options_data.extend(symbol_options)
                except Exception as e:
                    st.warning(f"Error processing {symbol}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        
        # Process the collected data
        if all_options_data:
            return self.process_options_data(all_options_data)
        else:
            return pd.DataFrame()
    
    def get_demo_data(self):
        """Generate demo data when needed"""
        st.info("📊 Generating demo data for testing")
        
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'SPY', 'QQQ', 'IWM']
        option_types = ['Call', 'Put']
        moneyness_options = ['ITM', 'ATM', 'OTM']
        
        data = []
        for i in range(50):
            symbol = np.random.choice(symbols)
            option_type = np.random.choice(option_types)
            
            # Generate realistic strikes based on symbol
            strike_ranges = {
                'AAPL': (150, 220), 'MSFT': (300, 450), 'GOOGL': (2400, 2800),
                'TSLA': (150, 350), 'NVDA': (400, 600), 'META': (200, 400),
                'AMZN': (140, 200), 'SPY': (400, 550), 'QQQ': (350, 480), 'IWM': (180, 240)
            }
            
            strike_min, strike_max = strike_ranges.get(symbol, (100, 200))
            strike = np.random.randint(strike_min, strike_max)
            underlying_price = strike + np.random.randint(-30, 31)
            
            volume = int(np.random.exponential(1000) + 50)  # Exponential distribution for realistic volume
            price = round(np.random.uniform(0.05, 30.0), 2)
            open_interest = int(np.random.exponential(2000) + 10)
            
            # Calculate derived metrics
            premium_value = volume * price * 100
            vol_oi_ratio = round(volume / max(open_interest, 1), 2)
            
            # Unusual score calculation
            score = self.calculate_unusual_score(volume, open_interest, price, strike, underlying_price, option_type)
            
            # Expiration dates
            exp_date = datetime.now() + timedelta(days=int(np.random.choice([7, 14, 21, 28, 35, 42, 49])))
            
            # Classify moneyness
            moneyness = self.classify_moneyness(strike, underlying_price, option_type)
            
            data.append({
                'symbol': symbol,
                'option_type': option_type,
                'strike': strike,
                'expiration': exp_date.strftime('%Y-%m-%d'),
                'volume': volume,
                'price': price,
                'open_interest': open_interest,
                'premium_value': premium_value,
                'vol_oi_ratio': vol_oi_ratio,
                'unusual_score': score,
                'bid': round(price * 0.95, 2),
                'ask': round(price * 1.05, 2),
                'implied_vol': round(np.random.uniform(0.15, 0.80), 3),
                'moneyness': moneyness,
                'underlying_price': underlying_price,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
        
        return pd.DataFrame(data).sort_values('unusual_score', ascending=False)

# Initialize scanner
@st.cache_resource
def get_scanner():
    return YahooOptionsScanner()

scanner = get_scanner()

# Sidebar controls
st.sidebar.title("🔧 Scanner Controls")

# Data source selection
st.sidebar.subheader("📊 Data Source")
data_mode = st.sidebar.radio(
    "Mode",
    ["Yahoo Finance (Live)", "Demo Data"],
    help="Yahoo Finance provides free, real-time options data"
)

# Symbol selection
st.sidebar.subheader("🎯 Symbol Selection")
symbol_preset = st.sidebar.selectbox(
    "Symbol Preset",
    ["Popular Stocks", "High Volume Options", "ETFs", "Custom List"],
    help="Choose a preset list or enter custom symbols"
)

custom_symbols = []
if symbol_preset == "Custom List":
    symbol_input = st.sidebar.text_area(
        "Enter Symbols (one per line)",
        placeholder="AAPL\nTSLA\nNVDA\nSPY\nQQQ",
        help="Enter stock symbols, one per line"
    )
    if symbol_input:
        custom_symbols = [s.strip().upper() for s in symbol_input.split('\n') if s.strip()]

# Get symbols based on preset
if symbol_preset == "Popular Stocks":
    selected_symbols = scanner.popular_symbols[:15]
elif symbol_preset == "High Volume Options":
    selected_symbols = scanner.high_vol_symbols
elif symbol_preset == "ETFs":
    selected_symbols = scanner.etf_symbols
elif symbol_preset == "Custom List":
    selected_symbols = custom_symbols[:20]  # Limit to 20 to prevent timeout
else:
    selected_symbols = scanner.popular_symbols[:15]

# Display selected symbols
if selected_symbols:
    st.sidebar.info(f"📈 Scanning: {', '.join(selected_symbols[:5])}{'...' if len(selected_symbols) > 5 else ''}")

# Filtering options
st.sidebar.subheader("🎯 Filters")

volume_threshold = st.sidebar.slider(
    "Minimum Volume",
    min_value=1,
    max_value=5000,
    value=50,
    step=10
)

premium_threshold = st.sidebar.slider(
    "Minimum Premium ($)",
    min_value=500,
    max_value=100000,
    value=2500,
    step=500
)

score_threshold = st.sidebar.slider(
    "Unusual Score Threshold",
    min_value=0.0,
    max_value=50.0,
    value=3.0,
    step=0.5
)

option_filter = st.sidebar.selectbox(
    "Option Type",
    ["All", "Calls Only", "Puts Only"]
)

moneyness_filter = st.sidebar.multiselect(
    "Moneyness",
    ["ITM", "ATM", "OTM"],
    default=["ITM", "ATM", "OTM"]
)

# Expiration filter
exp_filter = st.sidebar.selectbox(
    "Expiration Filter",
    ["All", "This Week (≤7 days)", "This Month (≤30 days)", "Next Quarter (≤90 days)"]
)

# Auto-refresh controls
refresh_interval = st.sidebar.selectbox(
    "Refresh Interval",
    ["Manual", "2 minutes", "5 minutes", "10 minutes"],
    index=0
)

manual_refresh = st.sidebar.button("🔄 Scan Now", type="primary")

# Data fetching logic
@st.cache_data(ttl=120, show_spinner=False)  # Cache for 2 minutes
def fetch_options_data(mode, symbols_list, timestamp):
    """Fetch options data with caching"""
    if mode == "Demo Data":
        return scanner.get_demo_data()
    else:
        return scanner.scan_unusual_options_activity(symbols_list)

# Auto-refresh logic
if refresh_interval != "Manual":
    interval_map = {"2 minutes": 120, "5 minutes": 300, "10 minutes": 600}
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    current_time = time.time()
    if current_time - st.session_state.last_refresh >= interval_map[refresh_interval]:
        st.session_state.last_refresh = current_time
        st.rerun()

# Fetch data
current_timestamp = int(time.time())
if manual_refresh or 'options_data' not in st.session_state:
    with st.spinner("🔍 Scanning unusual options activity..."):
        options_data = fetch_options_data(data_mode, selected_symbols, current_timestamp)
        st.session_state.options_data = options_data
        st.session_state.last_update = datetime.now()
else:
    options_data = st.session_state.options_data

# Main content
if options_data is not None and not options_data.empty:
    
    # Apply filters
    filtered_data = options_data.copy()
    
    # Volume filter
    filtered_data = filtered_data[filtered_data['volume'] >= volume_threshold]
    
    # Premium filter
    filtered_data = filtered_data[filtered_data['premium_value'] >= premium_threshold]
    
    # Score filter
    filtered_data = filtered_data[filtered_data['unusual_score'] >= score_threshold]
    
    # Option type filter
    if option_filter == "Calls Only":
        filtered_data = filtered_data[filtered_data['option_type'] == 'Call']
    elif option_filter == "Puts Only":
        filtered_data = filtered_data[filtered_data['option_type'] == 'Put']
    
    # Moneyness filter
    if moneyness_filter:
        filtered_data = filtered_data[filtered_data['moneyness'].isin(moneyness_filter)]
    
    # Expiration filter
    if exp_filter != "All":
        today = datetime.now().date()
        if exp_filter == "This Week (≤7 days)":
            cutoff = today + timedelta(days=7)
        elif exp_filter == "This Month (≤30 days)":
            cutoff = today + timedelta(days=30)
        elif exp_filter == "Next Quarter (≤90 days)":
            cutoff = today + timedelta(days=90)
        
        filtered_data['exp_date'] = pd.to_datetime(filtered_data['expiration']).dt.date
        filtered_data = filtered_data[filtered_data['exp_date'] <= cutoff]
        filtered_data = filtered_data.drop('exp_date', axis=1)
    
    # Display status
    col1, col2, col3 = st.columns(3)
    with col1:
        data_source = "Yahoo Finance (FREE)" if data_mode == "Yahoo Finance (Live)" else "Demo Mode"
        st.info(f"📊 **Data Source:** {data_source}")
    with col2:
        last_update = st.session_state.get('last_update', datetime.now())
        st.info(f"🕐 **Last Update:** {last_update.strftime('%H:%M:%S')}")
    with col3:
        st.info(f"📈 **Contracts Found:** {len(filtered_data)}")
    
    if not filtered_data.empty:
        
        # Key metrics
        st.subheader("📊 Market Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_volume = filtered_data['volume'].sum()
            st.metric("📢 Total Volume", f"{total_volume:,.0f}")
        
        with col2:
            total_premium = filtered_data['premium_value'].sum()
            st.metric("💰 Total Premium", f"${total_premium:,.0f}")
        
        with col3:
            avg_score = filtered_data['unusual_score'].mean()
            st.metric("⭐ Avg Score", f"{avg_score:.1f}")
        
        with col4:
            call_count = len(filtered_data[filtered_data['option_type'] == 'Call'])
            put_count = len(filtered_data[filtered_data['option_type'] == 'Put'])
            call_put_ratio = call_count / max(put_count, 1)
            st.metric("📊 Call/Put Ratio", f"{call_put_ratio:.2f}")
        
        with col5:
            high_vol_oi = len(filtered_data[filtered_data['vol_oi_ratio'] > 2.0])
            st.metric("🚨 High Vol/OI", f"{high_vol_oi}")
        
        # Top unusual activity table
        st.subheader("🎯 Top Unusual Options Activity")
        
        # Format display
        display_data = filtered_data.head(25).copy()
        display_data['Volume'] = display_data['volume'].apply(lambda x: f"{x:,.0f}")
        display_data['Premium'] = display_data['premium_value'].apply(lambda x: f"${x:,.0f}")
        display_data['Price'] = display_data['price'].apply(lambda x: f"${x:.2f}")
        display_data['Strike'] = display_data['strike'].apply(lambda x: f"${x:.0f}")
        display_data['Vol/OI'] = display_data['vol_oi_ratio'].apply(lambda x: f"{x:.1f}")
        display_data['Score'] = display_data['unusual_score'].apply(lambda x: f"{x:.1f}")
        display_data['IV'] = display_data['implied_vol'].apply(lambda x: f"{x:.0%}" if x > 0 else "N/A")
        
        # Select columns for display
        display_cols = ['symbol', 'option_type', 'Strike', 'expiration', 'Volume', 'Price', 
                       'Premium', 'Vol/OI', 'Score', 'IV', 'moneyness']
        column_names = ['Symbol', 'Type', 'Strike', 'Exp', 'Volume', 'Last', 'Premium', 
                       'Vol/OI', 'Score', 'IV', 'Money']
        
        final_display = display_data[display_cols].copy()
        final_display.columns = column_names
        
        # Color-code by score
        def highlight_scores(row):
            try:
                score = float(row['Score'])
                if score >= 25:
                    return ['background-color: #ffcdd2'] * len(row)  # Red
                elif score >= 15:
                    return ['background-color: #ffe0b2'] * len(row)  # Orange
                elif score >= 8:
                    return ['background-color: #f3e5f5'] * len(row)  # Purple
                else:
                    return ['background-color: #e8f5e8'] * len(row)  # Green
            except:
                return [''] * len(row)
        
        st.dataframe(
            final_display.style.apply(highlight_scores, axis=1),
            use_container_width=True,
            height=600
        )
        
        # Charts section
        st.subheader("📈 Analytics Dashboard")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Volume Analysis", "💰 Premium Flow", "📈 Score Analysis", "🎯 Activity Heatmap"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Volume by symbol
                symbol_volume = filtered_data.groupby('symbol')['volume'].sum().sort_values(ascending=False).head(12)
                fig1 = px.bar(
                    x=symbol_volume.index,
                    y=symbol_volume.values,
                    title="🏆 Top Symbols by Volume",
                    color=symbol_volume.values,
                    color_continuous_scale='viridis',
                    labels={'x': 'Symbol', 'y': 'Total Volume'}
                )
                fig1.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Call vs Put volume distribution
                type_volume = filtered_data.groupby('option_type')['volume'].sum()
                fig2 = px.pie(
                    values=type_volume.values,
                    names=type_volume.index,
                    title="📞 Call vs Put Volume Distribution",
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4']
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            # Premium flow scatter plot
            fig3 = px.scatter(
                filtered_data.head(100),  # Limit for performance
                x='volume',
                y='premium_value',
                color='unusual_score',
                size='unusual_score',
                hover_data=['symbol', 'option_type', 'strike', 'moneyness'],
                title="💰 Volume vs Premium Flow (Size & Color = Unusual Score)",
                color_continuous_scale='plasma',
                labels={'volume': 'Volume', 'premium_value': 'Premium Value ($)'}
            )
            fig3.update_layout(height=500)
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distribution
                fig4 = px.histogram(
                    filtered_data,
                    x='unusual_score',
                    nbins=25,
                    title="📈 Unusual Score Distribution",
                    color_discrete_sequence=['#8E44AD'],
                    labels={'unusual_score': 'Unusual Score', 'count': 'Number of Contracts'}
                )
                fig4.update_layout(height=400)
                st.plotly_chart(fig4, use_container_width=True)
            
            with col2:
                # Volume/OI ratio distribution
                fig5 = px.box(
                    filtered_data,
                    y='vol_oi_ratio',
                    x='option_type',
                    title="📊 Vol/OI Ratio by Option Type",
                    color='option_type',
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4']
                )
                fig5.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig5, use_container_width=True)
        
        with tab4:
            # Activity heatmap by moneyness and option type
            if 'moneyness' in filtered_data.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Volume heatmap
                    pivot_data = filtered_data.groupby(['moneyness', 'option_type'])['volume'].sum().reset_index()
                    pivot_table = pivot_data.pivot(index='moneyness', columns='option_type', values='volume').fillna(0)
                    
                    fig6 = px.imshow(
                        pivot_table.values,
                        x=pivot_table.columns,
                        y=pivot_table.index,
                        color_continuous_scale='viridis',
                        title="🔥 Volume Heatmap by Moneyness",
                        aspect="auto"
                    )
                    fig6.update_layout(height=400)
                    st.plotly_chart(fig6, use_container_width=True)
                
                with col2:
                    # Premium heatmap
                    pivot_premium = filtered_data.groupby(['moneyness', 'option_type'])['premium_value'].sum().reset_index()
                    pivot_premium_table = pivot_premium.pivot(index='moneyness', columns='option_type', values='premium_value').fillna(0)
                    
                    fig7 = px.imshow(
                        pivot_premium_table.values,
                        x=pivot_premium_table.columns,
                        y=pivot_premium_table.index,
                        color_continuous_scale='plasma',
                        title="💰 Premium Heatmap by Moneyness",
                        aspect="auto"
                    )
                    fig7.update_layout(height=400)
                    st.plotly_chart(fig7, use_container_width=True)
        
        # Alert system for high unusual activity
        st.subheader("🚨 Activity Alerts")
        
        # High score alerts
        high_score_alerts = filtered_data[filtered_data['unusual_score'] >= 20]
        if not high_score_alerts.empty:
            st.error(f"🔴 **HIGH ACTIVITY**: {len(high_score_alerts)} contracts with unusual scores ≥ 20")
            for _, alert in high_score_alerts.head(3).iterrows():
                st.write(f"• **{alert['symbol']} {alert['strike']}{alert['option_type'][0]}** - Score: {alert['unusual_score']:.1f}, Volume: {alert['volume']:,.0f}")
        
        # High volume alerts  
        high_volume_alerts = filtered_data[filtered_data['volume'] >= 2000]
        if not high_volume_alerts.empty:
            st.warning(f"🟡 **HIGH VOLUME**: {len(high_volume_alerts)} contracts with volume ≥ 2,000")
        
        # High Vol/OI alerts
        high_vol_oi_alerts = filtered_data[filtered_data['vol_oi_ratio'] >= 5]
        if not high_vol_oi_alerts.empty:
            st.info(f"🔵 **HIGH VOL/OI**: {len(high_vol_oi_alerts)} contracts with Vol/OI ratio ≥ 5.0")
        
        # Export functionality
        st.subheader("💾 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Dataset (CSV)",
                data=csv_data,
                file_name=f"unusual_options_yahoo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Top 10 summary
            top_10 = filtered_data.head(10)[['symbol', 'option_type', 'strike', 'expiration', 'volume', 'premium_value', 'unusual_score']]
            top_10_csv = top_10.to_csv(index=False)
            st.download_button(
                label="📥 Download Top 10 (CSV)",
                data=top_10_csv,
                file_name=f"top_10_unusual_yahoo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            # High alerts only
            alerts_data = filtered_data[filtered_data['unusual_score'] >= 15]
            if not alerts_data.empty:
                alerts_csv = alerts_data.to_csv(index=False)
                st.download_button(
                    label="🚨 Download Alerts (CSV)",
                    data=alerts_csv,
                    file_name=f"high_activity_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    else:
        st.warning("🔍 No options contracts match your current filters.")
        
        # Show suggestion for filter adjustment
        original_count = len(options_data)
        st.info(f"💡 Found {original_count} total contracts before filtering. Try adjusting your thresholds:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"• Lower volume threshold (currently {volume_threshold})")
        with col2:
            st.write(f"• Lower premium threshold (currently ${premium_threshold:,.0f})")
        with col3:
            st.write(f"• Lower score threshold (currently {score_threshold})")

else:
    st.error("❌ Unable to fetch options data.")
    
    if data_mode == "Yahoo Finance (Live)":
        st.info("""
        **Possible solutions:**
        
        1. **Check internet connection**
        2. **Try demo mode** to test the interface
        3. **Reduce symbol list** if scanning too many symbols
        4. **Yahoo Finance might be temporarily unavailable**
        
        **Install yfinance if needed:**
        ```bash
        pip install yfinance
        ```
        """)

# Information Panel
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Yahoo Finance Info")
st.sidebar.markdown("""
**100% FREE - No API Key Required!**

✅ **Features:**
- Real-time options chains
- Volume & open interest data  
- Implied volatility
- Complete options data
- No rate limits
- No signup required

⚡ **Data Quality:**
- 15-minute delayed (still very current)
- Direct from Yahoo Finance
- Same data used by major platforms
- Reliable and comprehensive

📊 **Coverage:**
- All US stocks with options
- Major ETFs (SPY, QQQ, IWM, etc.)
- Popular options symbols
- Multiple expirations

🛠️ **Requirements:**
- Python package: `yfinance`
- Internet connection
- That's it!
""")

# Performance tips
st.sidebar.markdown("---") 
st.sidebar.subheader("⚡ Performance Tips")
st.sidebar.markdown("""
**For faster scanning:**
- Use "High Volume Options" preset
- Limit custom symbols to ~15
- Enable auto-refresh sparingly
- Use filters to reduce data size

**Best symbol presets:**
- **Popular Stocks**: Broad market scan
- **High Volume Options**: Most active
- **ETFs**: Market indices & sectors
- **Custom**: Your watchlist
""")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>📊 Options Flow Intelligence • Built with Streamlit • Powered by Yahoo Finance API (FREE) 🚀</p>
        <p><small>⚠️ This tool is for educational purposes only. Not financial advice.</small></p>
        <p><small>✨ Data provided by Yahoo Finance via yfinance library - completely free to use!</small></p>
    </div>
    """,
    unsafe_allow_html=True
)