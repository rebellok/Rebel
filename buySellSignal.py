import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import ta
from ta.utils import dropna
import requests
from typing import Dict, List, Tuple, Optional
import ssl
import certifi


warnings.filterwarnings('ignore')


# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context


st.set_page_config(
    page_title="Trading Signals & Screener Pro",
    page_icon="📈🎯",
    layout="wide"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .signal-buy {
        background-color: #28a745;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .signal-sell {
        background-color: #dc3545;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .signal-neutral {
        background-color: #ffc107;
        color: black;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# Header
st.markdown('''
<div class="main-header">
    <h1>📈🎯 Trading Signals & Screener Pro</h1>
    <p>Advanced Stock & Options Analysis with Real-time Yahoo Finance Data</p>
</div>
''', unsafe_allow_html=True)


class TechnicalIndicators:
    """Calculate various technical indicators"""
   
    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        return ta.momentum.RSIIndicator(data, window=window).rsi()
   
    @staticmethod
    def calculate_macd(data: pd.Series) -> Dict:
        """Calculate MACD"""
        macd = ta.trend.MACD(data)
        return {
            'macd': macd.macd(),
            'signal': macd.macd_signal(),
            'histogram': macd.macd_diff()
        }
   
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, window: int = 20) -> Dict:
        """Calculate Bollinger Bands"""
        bb = ta.volatility.BollingerBands(data, window=window)
        return {
            'upper': bb.bollinger_hband(),
            'middle': bb.bollinger_mavg(),
            'lower': bb.bollinger_lband(),
            'width': bb.bollinger_wband(),
            'percent': bb.bollinger_pband()
        }
   
    @staticmethod
    def calculate_moving_averages(data: pd.Series) -> Dict:
        """Calculate various moving averages"""
        return {
            'sma_20': ta.trend.SMAIndicator(data, window=20).sma_indicator(),
            'sma_50': ta.trend.SMAIndicator(data, window=50).sma_indicator(),
            'sma_200': ta.trend.SMAIndicator(data, window=200).sma_indicator(),
            'ema_12': ta.trend.EMAIndicator(data, window=12).ema_indicator(),
            'ema_26': ta.trend.EMAIndicator(data, window=26).ema_indicator()
        }
   
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict:
        """Calculate Stochastic Oscillator"""
        stoch = ta.momentum.StochasticOscillator(high, low, close)
        return {
            'k': stoch.stoch(),
            'd': stoch.stoch_signal()
        }
   
    @staticmethod
    def calculate_volume_indicators(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> Dict:
        """Calculate volume-based indicators"""
        return {
            'volume_sma': volume.rolling(window=20).mean(),
            'obv': ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume(),
            'mfi': ta.volume.MFIIndicator(high, low, close, volume).money_flow_index()
        }


class SignalGenerator:
    """Generate buy/sell signals based on technical analysis"""
   
    def __init__(self):
        self.ti = TechnicalIndicators()
   
    def analyze_stock(self, symbol: str, period: str = "6mo") -> Dict:
        """Comprehensive stock analysis"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            info = ticker.info
           
            if hist.empty:
                return None
           
            # Calculate indicators
            close = hist['Close']
            high = hist['High']
            low = hist['Low']
            volume = hist['Volume']
           
            # Technical indicators
            rsi = self.ti.calculate_rsi(close)
            macd = self.ti.calculate_macd(close)
            bb = self.ti.calculate_bollinger_bands(close)
            ma = self.ti.calculate_moving_averages(close)
            stoch = self.ti.calculate_stochastic(high, low, close)
            vol_ind = self.ti.calculate_volume_indicators(high, low, close, volume)
           
            # Current values
            current_price = close.iloc[-1]
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            current_macd = macd['macd'].iloc[-1] if not macd['macd'].empty else 0
            current_signal = macd['signal'].iloc[-1] if not macd['signal'].empty else 0
           
            # Generate signals
            signals = self._generate_signals(
                current_price, current_rsi, current_macd, current_signal,
                ma, bb, stoch, info
            )
           
            # Calculate additional metrics
            volatility = close.pct_change().std() * np.sqrt(252) * 100
            price_change_1d = ((current_price - close.iloc[-2]) / close.iloc[-2]) * 100 if len(close) >= 2 else 0
            price_change_5d = ((current_price - close.iloc[-6]) / close.iloc[-6]) * 100 if len(close) >= 6 else 0
           
            return {
                'symbol': symbol,
                'current_price': current_price,
                'price_change_1d': price_change_1d,
                'price_change_5d': price_change_5d,
                'volume': volume.iloc[-1],
                'avg_volume': volume.rolling(20).mean().iloc[-1],
                'rsi': current_rsi,
                'macd': current_macd,
                'macd_signal': current_signal,
                'bb_position': bb['percent'].iloc[-1] if not bb['percent'].empty else 0.5,
                'volatility': volatility,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'signals': signals,
                'technical_score': self._calculate_technical_score(signals),
                'recommendation': self._get_recommendation(signals)
            }
           
        except Exception as e:
            st.warning(f"Error analyzing {symbol}: {str(e)}")
            return None
   
    def _generate_signals(self, price, rsi, macd, signal, ma, bb, stoch, info) -> Dict:
        """Generate individual trading signals"""
        signals = {}
       
        # RSI Signals
        if rsi < 30:
            signals['rsi'] = 'BUY'
        elif rsi > 70:
            signals['rsi'] = 'SELL'
        else:
            signals['rsi'] = 'NEUTRAL'
       
        # MACD Signals
        if macd > signal and macd > 0:
            signals['macd'] = 'BUY'
        elif macd < signal and macd < 0:
            signals['macd'] = 'SELL'
        else:
            signals['macd'] = 'NEUTRAL'
       
        # Moving Average Signals
        try:
            sma_20 = ma['sma_20'].iloc[-1] if not ma['sma_20'].empty else price
            sma_50 = ma['sma_50'].iloc[-1] if not ma['sma_50'].empty else price
            if price > sma_20 > sma_50:
                signals['trend'] = 'BUY'
            elif price < sma_20 < sma_50:
                signals['trend'] = 'SELL'
            else:
                signals['trend'] = 'NEUTRAL'
        except:
            signals['trend'] = 'NEUTRAL'
       
        # Bollinger Bands Signals
        try:
            bb_pos = bb['percent'].iloc[-1] if not bb['percent'].empty else 0.5
            if bb_pos < 0.2:
                signals['bb'] = 'BUY'
            elif bb_pos > 0.8:
                signals['bb'] = 'SELL'
            else:
                signals['bb'] = 'NEUTRAL'
        except:
            signals['bb'] = 'NEUTRAL'
       
        # Stochastic Signals
        try:
            k = stoch['k'].iloc[-1] if not stoch['k'].empty else 50
            d = stoch['d'].iloc[-1] if not stoch['d'].empty else 50
            if k < 20 and d < 20:
                signals['stoch'] = 'BUY'
            elif k > 80 and d > 80:
                signals['stoch'] = 'SELL'
            else:
                signals['stoch'] = 'NEUTRAL'
        except:
            signals['stoch'] = 'NEUTRAL'
       
        # Fundamental Signals
        pe = info.get('trailingPE', 0)
        if pe and pe > 0 and pe < 15:
            signals['valuation'] = 'BUY'
        elif pe and pe > 25:
            signals['valuation'] = 'SELL'
        else:
            signals['valuation'] = 'NEUTRAL'
       
        return signals
   
    def _calculate_technical_score(self, signals: Dict) -> float:
        """Calculate overall technical score (0-100)"""
        buy_count = sum(1 for s in signals.values() if s == 'BUY')
        sell_count = sum(1 for s in signals.values() if s == 'SELL')
        total_signals = len(signals)
       
        if total_signals == 0:
            return 50.0
       
        # Score from 0 (all sell) to 100 (all buy)
        score = ((buy_count - sell_count) / total_signals + 1) * 50
        return round(score, 1)
   
    def _get_recommendation(self, signals: Dict) -> str:
        """Get overall recommendation"""
        buy_count = sum(1 for s in signals.values() if s == 'BUY')
        sell_count = sum(1 for s in signals.values() if s == 'SELL')
       
        if buy_count >= sell_count + 2:
            return 'STRONG BUY'
        elif buy_count > sell_count:
            return 'BUY'
        elif sell_count > buy_count:
            return 'SELL'
        elif sell_count >= buy_count + 2:
            return 'STRONG SELL'
        else:
            return 'HOLD'


class OptionsAnalyzer:
    """Analyze options for trading signals"""
   
    def __init__(self):
        self.signal_gen = SignalGenerator()
   
    def analyze_options_flow(self, symbol: str) -> Dict:
        """Analyze options flow for signals"""
        try:
            ticker = yf.Ticker(symbol)
           
            # Get current stock price
            info = ticker.info
            current_price = info.get('currentPrice', 0)
           
            if current_price == 0:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                else:
                    return None
           
            # Get options data
            exp_dates = ticker.options
            if not exp_dates:
                return None
           
            all_options_data = []
           
            # Analyze first few expirations
            for exp_date in exp_dates[:3]:
                try:
                    chain = ticker.option_chain(exp_date)
                   
                    # Process calls
                    calls = chain.calls.copy()
                    calls['type'] = 'call'
                    calls['expiration'] = exp_date
                    all_options_data.append(calls)
                   
                    # Process puts
                    puts = chain.puts.copy()
                    puts['type'] = 'put'
                    puts['expiration'] = exp_date
                    all_options_data.append(puts)
                   
                except Exception as e:
                    continue
           
            if not all_options_data:
                return None
           
            # Combine all options data
            options_df = pd.concat(all_options_data, ignore_index=True)
           
            # Calculate options signals
            signals = self._calculate_options_signals(options_df, current_price)
           
            return {
                'symbol': symbol,
                'current_price': current_price,
                'total_call_volume': options_df[options_df['type'] == 'call']['volume'].sum(),
                'total_put_volume': options_df[options_df['type'] == 'put']['volume'].sum(),
                'signals': signals,
                'options_data': options_df
            }
           
        except Exception as e:
            return None
   
    def _calculate_options_signals(self, options_df: pd.DataFrame, current_price: float) -> Dict:
        """Calculate options-based signals"""
        signals = {}
       
        # Clean data
        options_df = options_df.dropna(subset=['volume', 'openInterest'])
        options_df = options_df[options_df['volume'] > 0]
       
        if options_df.empty:
            return {'flow': 'NEUTRAL', 'unusual_activity': 'NEUTRAL'}
       
        # Call/Put Volume Ratio
        call_volume = options_df[options_df['type'] == 'call']['volume'].sum()
        put_volume = options_df[options_df['type'] == 'put']['volume'].sum()
       
        if put_volume > 0:
            cp_ratio = call_volume / put_volume
            if cp_ratio > 1.5:
                signals['flow'] = 'BUY'
            elif cp_ratio < 0.67:
                signals['flow'] = 'SELL'
            else:
                signals['flow'] = 'NEUTRAL'
        else:
            signals['flow'] = 'BUY' if call_volume > 0 else 'NEUTRAL'
       
        # Unusual Activity Detection
        options_df['vol_oi_ratio'] = options_df['volume'] / options_df['openInterest'].replace(0, 1)
        high_activity = options_df[options_df['vol_oi_ratio'] > 2]
       
        if len(high_activity) > 0:
            # Check if unusual activity is more calls or puts
            unusual_calls = len(high_activity[high_activity['type'] == 'call'])
            unusual_puts = len(high_activity[high_activity['type'] == 'put'])
           
            if unusual_calls > unusual_puts:
                signals['unusual_activity'] = 'BUY'
            elif unusual_puts > unusual_calls:
                signals['unusual_activity'] = 'SELL'
            else:
                signals['unusual_activity'] = 'NEUTRAL'
        else:
            signals['unusual_activity'] = 'NEUTRAL'
       
        # Options Positioning
        itm_calls = options_df[(options_df['type'] == 'call') & (options_df['strike'] < current_price)]['volume'].sum()
        otm_calls = options_df[(options_df['type'] == 'call') & (options_df['strike'] > current_price)]['volume'].sum()
       
        if otm_calls > itm_calls * 1.5:
            signals['positioning'] = 'BUY'  # Bullish speculation
        elif itm_calls > otm_calls * 1.5:
            signals['positioning'] = 'SELL'  # Bearish hedging
        else:
            signals['positioning'] = 'NEUTRAL'
       
        return signals


class StockScreener:
    """Screen stocks based on various criteria"""
   
    def __init__(self):
        self.signal_gen = SignalGenerator()
       
        # Popular stock lists
        self.sp500_sample = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'UNH', 'JNJ',
            'V', 'PG', 'JPM', 'HD', 'MA', 'CVX', 'ABBV', 'PFE', 'KO', 'BAC',
            'PEP', 'TMO', 'COST', 'AVGO', 'DIS', 'ABT', 'CRM', 'NFLX', 'ADBE', 'XOM'
        ]
       
        self.growth_stocks = [
            'NVDA', 'AMD', 'TSLA', 'CRM', 'SNOW', 'ZM', 'ROKU', 'SQ', 'SHOP', 'TWLO',
            'OKTA', 'CRWD', 'ZS', 'DDOG', 'NET', 'PLTR', 'U', 'FVRR', 'UPWK'
        ]
       
        self.dividend_stocks = [
            'KO', 'PEP', 'JNJ', 'PG', 'T', 'VZ', 'XOM', 'CVX', 'MMM', 'CAT',
            'IBM', 'WMT', 'MCD', 'O', 'VTI', 'SPY', 'SCHD'
        ]
   
    def screen_stocks(self, stock_list: List[str], filters: Dict) -> pd.DataFrame:
        """Screen stocks based on filters"""
        results = []
       
        progress_bar = st.progress(0)
        status_text = st.empty()
       
        for i, symbol in enumerate(stock_list):
            status_text.text(f"Analyzing {symbol}... ({i+1}/{len(stock_list)})")
            progress_bar.progress((i + 1) / len(stock_list))
           
            analysis = self.signal_gen.analyze_stock(symbol)
            if analysis:
                # Apply filters
                if self._passes_filters(analysis, filters):
                    results.append(analysis)
       
        progress_bar.empty()
        status_text.empty()
       
        return pd.DataFrame(results)
   
    def _passes_filters(self, analysis: Dict, filters: Dict) -> bool:
        """Check if stock passes screening filters"""
        # Price filter
        if filters.get('min_price', 0) > 0:
            if analysis['current_price'] < filters['min_price']:
                return False
       
        if filters.get('max_price', 0) > 0:
            if analysis['current_price'] > filters['max_price']:
                return False
       
        # Volume filter
        if filters.get('min_volume', 0) > 0:
            if analysis['volume'] < filters['min_volume']:
                return False
       
        # Market cap filter
        if filters.get('min_market_cap', 0) > 0:
            if analysis['market_cap'] < filters['min_market_cap']:
                return False
       
        # Technical score filter
        if filters.get('min_tech_score', 0) > 0:
            if analysis['technical_score'] < filters['min_tech_score']:
                return False
       
        # Signal filter
        if filters.get('signal_filter'):
            if analysis['recommendation'] not in filters['signal_filter']:
                return False
       
        return True


# Initialize components
@st.cache_resource
def get_components():
    return {
        'signal_gen': SignalGenerator(),
        'options_analyzer': OptionsAnalyzer(),
        'screener': StockScreener()
    }


components = get_components()


# Sidebar
st.sidebar.title("🔧 Control Panel")


# Analysis Type
analysis_type = st.sidebar.radio(
    "Analysis Type",
    ["📈 Stock Screener", "🎯 Individual Stock Analysis", "📊 Options Flow Analysis", "📋 Custom Watchlist"]
)


# Stock Screener Section
if analysis_type == "📈 Stock Screener":
    st.header("📈 Stock Screener")
   
    # Screening options
    col1, col2 = st.columns(2)
   
    with col1:
        stock_list_type = st.selectbox(
            "Stock Universe",
            ["S&P 500 Sample", "Growth Stocks", "Dividend Stocks", "Custom List"]
        )
       
        if stock_list_type == "Custom List":
            custom_symbols = st.text_area(
                "Enter symbols (one per line)",
                placeholder="AAPL\nMSFT\nGOOGL\nTSLA",
                help="Enter stock symbols, one per line"
            ).strip().split('\n')
            stock_universe = [s.strip().upper() for s in custom_symbols if s.strip()]
        else:
            stock_universe = {
                "S&P 500 Sample": components['screener'].sp500_sample,
                "Growth Stocks": components['screener'].growth_stocks,
                "Dividend Stocks": components['screener'].dividend_stocks
            }[stock_list_type]
   
    with col2:
        # Filters
        st.subheader("🎯 Screening Filters")
        min_price = st.number_input("Min Price ($)", min_value=0.0, value=5.0, step=1.0)
        max_price = st.number_input("Max Price ($)", min_value=0.0, value=0.0, step=1.0)
        min_volume = st.number_input("Min Volume", min_value=0, value=100000, step=10000)
        min_market_cap = st.number_input("Min Market Cap (B)", min_value=0.0, value=1.0, step=0.5) * 1e9
        min_tech_score = st.slider("Min Technical Score", 0, 100, 30)
       
        signal_filter = st.multiselect(
            "Signal Filter",
            ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"],
            default=["STRONG BUY", "BUY"]
        )
   
    # Run screener
    if st.button("🚀 Run Screener", type="primary"):
        filters = {
            'min_price': min_price,
            'max_price': max_price if max_price > 0 else float('inf'),
            'min_volume': min_volume,
            'min_market_cap': min_market_cap,
            'min_tech_score': min_tech_score,
            'signal_filter': signal_filter
        }
       
        with st.spinner("🔍 Screening stocks..."):
            results_df = components['screener'].screen_stocks(stock_universe[:20], filters)
       
        if not results_df.empty:
            # Sort by technical score
            results_df = results_df.sort_values('technical_score', ascending=False)
           
            # Display results
            st.success(f"✅ Found {len(results_df)} stocks matching your criteria")
           
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_score = results_df['technical_score'].mean()
                st.metric("📊 Avg Technical Score", f"{avg_score:.1f}")
            with col2:
                buy_signals = len(results_df[results_df['recommendation'].isin(['BUY', 'STRONG BUY'])])
                st.metric("📈 Buy Signals", buy_signals)
            with col3:
                avg_price = results_df['current_price'].mean()
                st.metric("💲 Avg Price", f"${avg_price:.2f}")
            with col4:
                total_market_cap = results_df['market_cap'].sum() / 1e9
                st.metric("🏢 Total Market Cap", f"${total_market_cap:.1f}B")
           
            # Results table
            display_df = results_df[['symbol', 'current_price', 'price_change_1d', 'volume',
                                   'rsi', 'technical_score', 'recommendation']].copy()
           
            display_df.columns = ['Symbol', 'Price', '1D Change %', 'Volume', 'RSI', 'Tech Score', 'Signal']
            display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
            display_df['1D Change %'] = display_df['1D Change %'].apply(lambda x: f"{x:.2f}%")
            display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,.0f}")
            display_df['RSI'] = display_df['RSI'].apply(lambda x: f"{x:.1f}")
            display_df['Tech Score'] = display_df['Tech Score'].apply(lambda x: f"{x:.1f}")
           
            # Color coding
            def highlight_signals(row):
                if row['Signal'] in ['STRONG BUY', 'BUY']:
                    return ['background-color: #d4edda'] * len(row)
                elif row['Signal'] in ['STRONG SELL', 'SELL']:
                    return ['background-color: #f8d7da'] * len(row)
                else:
                    return ['background-color: #fff3cd'] * len(row)
           
            st.dataframe(
                display_df.style.apply(highlight_signals, axis=1),
                use_container_width=True,
                height=400
            )
           
            # Charts
            col1, col2 = st.columns(2)
           
            with col1:
                # Technical score distribution
                fig1 = px.histogram(
                    results_df,
                    x='technical_score',
                    title="Technical Score Distribution",
                    nbins=10,
                    color_discrete_sequence=['#1f77b4']
                )
                st.plotly_chart(fig1, use_container_width=True)
           
            with col2:
                # Signal distribution
                signal_counts = results_df['recommendation'].value_counts()
                fig2 = px.pie(
                    values=signal_counts.values,
                    names=signal_counts.index,
                    title="Signal Distribution"
                )
                st.plotly_chart(fig2, use_container_width=True)
       
        else:
            st.warning("⚠️ No stocks found matching your criteria. Try adjusting the filters.")


# Individual Stock Analysis
elif analysis_type == "🎯 Individual Stock Analysis":
    st.header("🎯 Individual Stock Analysis")
   
    col1, col2 = st.columns([2, 1])
   
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL").upper()
   
    with col2:
        period = st.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y", "2y"])
   
    if st.button("🔍 Analyze Stock", type="primary"):
        with st.spinner(f"🔍 Analyzing {symbol}..."):
            analysis = components['signal_gen'].analyze_stock(symbol, period)
       
        if analysis:
            # Display key metrics
            col1, col2, col3, col4, col5 = st.columns(5)
           
            with col1:
                st.metric("💲 Current Price", f"${analysis['current_price']:.2f}",
                         f"{analysis['price_change_1d']:.2f}%")
            with col2:
                st.metric("📊 Technical Score", f"{analysis['technical_score']:.1f}/100")
            with col3:
                st.metric("📈 RSI", f"{analysis['rsi']:.1f}")
            with col4:
                st.metric("📊 Volume", f"{analysis['volume']:,.0f}")
            with col5:
                # Recommendation with color
                rec = analysis['recommendation']
                if rec in ['STRONG BUY', 'BUY']:
                    st.markdown(f'<div class="signal-buy">{rec}</div>', unsafe_allow_html=True)
                elif rec in ['STRONG SELL', 'SELL']:
                    st.markdown(f'<div class="signal-sell">{rec}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="signal-neutral">{rec}</div>', unsafe_allow_html=True)
           
            # Signal breakdown
            st.subheader("Signal Breakdown")
            signals_df = pd.DataFrame([
                {'Indicator': 'RSI', 'Signal': analysis['signals']['rsi']},
                {'Indicator': 'MACD', 'Signal': analysis['signals']['macd']},
                {'Indicator': 'Trend (MA)', 'Signal': analysis['signals']['trend']},
                {'Indicator': 'Bollinger Bands', 'Signal': analysis['signals']['bb']},
                {'Indicator': 'Stochastic', 'Signal': analysis['signals']['stoch']},
                {'Indicator': 'Valuation', 'Signal': analysis['signals']['valuation']}
            ])
           
            def color_signals(val):
                if val == 'BUY':
                    return 'background-color: #d4edda'
                elif val == 'SELL':
                    return 'background-color: #f8d7da'
                else:
                    return 'background-color: #fff3cd'
           
            st.dataframe(
                signals_df.style.applymap(color_signals, subset=['Signal']),
                use_container_width=True,
                height=200
            )
           
            # Chart with technical indicators
            st.subheader("Technical Analysis Chart")
           
            # Fetch historical data for charting
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
           
            if not hist.empty:
                # Calculate indicators for plotting
                close = hist['Close']
                rsi = TechnicalIndicators.calculate_rsi(close)
                macd = TechnicalIndicators.calculate_macd(close)
                bb = TechnicalIndicators.calculate_bollinger_bands(close)
                ma = TechnicalIndicators.calculate_moving_averages(close)
               
                # Create subplots
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Price & Moving Averages', 'RSI', 'MACD'),
                    vertical_spacing=0.08,
                    row_width=[0.6, 0.2, 0.2]
                )
               
                # Price chart with Bollinger Bands and MA
                fig.add_trace(
                    go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                  low=hist['Low'], close=hist['Close'], name='Price'),
                    row=1, col=1
                )
               
                if not bb['upper'].empty:
                    fig.add_trace(
                        go.Scatter(x=hist.index, y=bb['upper'], name='BB Upper',
                                  line=dict(color='gray', dash='dash')),
                        row=1, col=1
                    )
                   
                    fig.add_trace(
                        go.Scatter(x=hist.index, y=bb['lower'], name='BB Lower',
                                  line=dict(color='gray', dash='dash'), fill='tonexty'),
                        row=1, col=1
                    )
               
                if not ma['sma_20'].empty:
                    fig.add_trace(
                        go.Scatter(x=hist.index, y=ma['sma_20'], name='SMA 20',
                                  line=dict(color='blue')),
                        row=1, col=1
                    )
               
                if not ma['sma_50'].empty:
                    fig.add_trace(
                        go.Scatter(x=hist.index, y=ma['sma_50'], name='SMA 50',
                                  line=dict(color='orange')),
                        row=1, col=1
                    )
               
                # RSI
                if not rsi.empty:
                    fig.add_trace(
                        go.Scatter(x=hist.index, y=rsi, name='RSI',
                                  line=dict(color='purple')),
                        row=2, col=1
                    )
                   
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
               
                # MACD
                if not macd['macd'].empty:
                    fig.add_trace(
                        go.Scatter(x=hist.index, y=macd['macd'], name='MACD',
                                  line=dict(color='blue')),
                        row=3, col=1
                    )
               
                if not macd['signal'].empty:
                    fig.add_trace(
                        go.Scatter(x=hist.index, y=macd['signal'], name='Signal',
                                  line=dict(color='red')),
                        row=3, col=1
                    )
               
                if not macd['histogram'].empty:
                    fig.add_trace(
                        go.Bar(x=hist.index, y=macd['histogram'], name='Histogram',
                              marker_color='green'),
                        row=3, col=1
                    )
               
                fig.update_layout(
                    title=f"{symbol} Technical Analysis",
                    height=800,
                    showlegend=True,
                    xaxis_rangeslider_visible=False
                )
               
                st.plotly_chart(fig, use_container_width=True)
           
            # Additional insights
            st.subheader("Key Insights")
            insights = []
           
            if analysis['rsi'] < 30:
                insights.append("Stock is oversold (RSI < 30) - potential buying opportunity")
            elif analysis['rsi'] > 70:
                insights.append("Stock is overbought (RSI > 70) - potential selling opportunity")
           
            if analysis['price_change_5d'] > 5:
                insights.append(f"Strong upward momentum (+{analysis['price_change_5d']:.1f}% over 5 days)")
            elif analysis['price_change_5d'] < -5:
                insights.append(f"Strong downward momentum ({analysis['price_change_5d']:.1f}% over 5 days)")
           
            if analysis['volume'] > analysis['avg_volume'] * 1.5:
                insights.append("Above-average volume - increased interest")
           
            if analysis['volatility'] > 30:
                insights.append(f"High volatility ({analysis['volatility']:.1f}%) - increased risk/reward")
           
            for insight in insights:
                st.info(insight)
       
        else:
            st.error(f"Unable to analyze {symbol}. Please check the symbol and try again.")


# Options Flow Analysis
elif analysis_type == "📊 Options Flow Analysis":
    st.header("📊 Options Flow Analysis")
   
    col1, col2 = st.columns([2, 1])
   
    with col1:
        options_symbol = st.text_input("Enter Stock Symbol for Options", value="AAPL", placeholder="e.g., AAPL, TSLA, NVDA").upper()
   
    with col2:
        min_volume = st.number_input("Min Options Volume", min_value=0, value=10)
   
    if st.button("🎯 Analyze Options Flow", type="primary"):
        with st.spinner(f"🔍 Analyzing options flow for {options_symbol}..."):
            options_analysis = components['options_analyzer'].analyze_options_flow(options_symbol)
       
        if options_analysis:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
           
            with col1:
                st.metric("💲 Stock Price", f"${options_analysis['current_price']:.2f}")
           
            with col2:
                call_vol = options_analysis['total_call_volume']
                st.metric("📞 Call Volume", f"{call_vol:,.0f}")
           
            with col3:
                put_vol = options_analysis['total_put_volume']
                st.metric("📉 Put Volume", f"{put_vol:,.0f}")
           
            with col4:
                if put_vol > 0:
                    cp_ratio = call_vol / put_vol
                    st.metric("📊 Call/Put Ratio", f"{cp_ratio:.2f}")
                else:
                    st.metric("📊 Call/Put Ratio", "∞")
           
            # Options signals
            st.subheader("🎯 Options Signals")
           
            signals = options_analysis['signals']
            signals_df = pd.DataFrame([
                {'Signal Type': 'Flow Direction', 'Signal': signals.get('flow', 'NEUTRAL')},
                {'Signal Type': 'Unusual Activity', 'Signal': signals.get('unusual_activity', 'NEUTRAL')},
                {'Signal Type': 'Positioning', 'Signal': signals.get('positioning', 'NEUTRAL')}
            ])
           
            def color_options_signals(val):
                if val == 'BUY':
                    return 'background-color: #d4edda'
                elif val == 'SELL':
                    return 'background-color: #f8d7da'
                else:
                    return 'background-color: #fff3cd'
           
            st.dataframe(
                signals_df.style.applymap(color_options_signals, subset=['Signal']),
                use_container_width=True,
                height=150
            )
           
            # Options chain data
            st.subheader("📋 Active Options Contracts")
           
            options_df = options_analysis['options_data']
            options_df = options_df[options_df['volume'] >= min_volume].copy()
           
            if not options_df.empty:
                # Display top contracts by volume
                display_options = options_df.nlargest(20, 'volume')[
                    ['type', 'strike', 'expiration', 'volume', 'openInterest', 'lastPrice', 'impliedVolatility']
                ].copy()
               
                display_options.columns = ['Type', 'Strike', 'Expiration', 'Volume', 'Open Interest', 'Last Price', 'IV']
                display_options['Strike'] = display_options['Strike'].apply(lambda x: f"${x:.0f}")
                display_options['Last Price'] = display_options['Last Price'].apply(lambda x: f"${x:.2f}")
                display_options['IV'] = display_options['IV'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
                display_options['Volume'] = display_options['Volume'].apply(lambda x: f"{x:,.0f}")
                display_options['Open Interest'] = display_options['Open Interest'].apply(lambda x: f"{x:,.0f}")
               
                st.dataframe(display_options, use_container_width=True, height=400)
               
                # Options flow visualization
                col1, col2 = st.columns(2)
               
                with col1:
                    # Volume by type
                    type_volume = options_df.groupby('type')['volume'].sum()
                    fig1 = px.pie(
                        values=type_volume.values,
                        names=['Calls', 'Puts'],
                        title="Options Volume Distribution",
                        color_discrete_sequence=['#00cc96', '#ef553b']
                    )
                    st.plotly_chart(fig1, use_container_width=True)
               
                with col2:
                    # Volume by strike (around current price)
                    current_price = options_analysis['current_price']
                    nearby_strikes = options_df[
                        (options_df['strike'] >= current_price * 0.9) &
                        (options_df['strike'] <= current_price * 1.1)
                    ]
                   
                    if not nearby_strikes.empty:
                        strike_volume = nearby_strikes.groupby('strike')['volume'].sum().sort_index()
                        fig2 = px.bar(
                            x=strike_volume.index,
                            y=strike_volume.values,
                            title="Volume by Strike (Near Current Price)",
                            labels={'x': 'Strike Price', 'y': 'Volume'}
                        )
                        fig2.add_vline(x=current_price, line_dash="dash", line_color="red",
                                      annotation_text="Current Price")
                        st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning(f"⚠️ No options contracts found with volume >= {min_volume}")
       
        else:
            st.error(f"❌ Unable to analyze options for {options_symbol}. Please check the symbol and try again.")


# Custom Watchlist
elif analysis_type == "📋 Custom Watchlist":
    st.header("📋 Custom Watchlist Analysis")
   
    col1, col2 = st.columns([2, 1])
   
    with col1:
        watchlist_input = st.text_area(
            "Enter your watchlist (one symbol per line)",
            placeholder="AAPL\nMSFT\nGOOGL\nTSLA\nNVDA\nAMD\nMETA\nAMZN",
            height=200
        )
   
    with col2:
        st.subheader("⚙️ Analysis Options")
        include_options = st.checkbox("Include Options Analysis", value=False)
        sort_by = st.selectbox(
            "Sort Results By",
            ["Technical Score", "Price Change 1D", "Volume", "RSI"]
        )
       
        show_details = st.checkbox("Show Detailed Signals", value=True)
   
    if st.button("📊 Analyze Watchlist", type="primary"):
        if watchlist_input.strip():
            symbols = [s.strip().upper() for s in watchlist_input.strip().split('\n') if s.strip()]
           
            if symbols:
                results = []
                options_results = []
               
                progress_bar = st.progress(0)
                status_text = st.empty()
               
                for i, symbol in enumerate(symbols):
                    status_text.text(f"Analyzing {symbol}... ({i+1}/{len(symbols)})")
                    progress_bar.progress((i + 1) / len(symbols))
                   
                    # Stock analysis
                    stock_analysis = components['signal_gen'].analyze_stock(symbol)
                    if stock_analysis:
                        results.append(stock_analysis)
                   
                    # Options analysis if requested
                    if include_options:
                        options_analysis = components['options_analyzer'].analyze_options_flow(symbol)
                        if options_analysis:
                            options_results.append(options_analysis)
               
                progress_bar.empty()
                status_text.empty()
               
                if results:
                    results_df = pd.DataFrame(results)
                   
                    # Sort results
                    sort_mapping = {
                        "Technical Score": "technical_score",
                        "Price Change 1D": "price_change_1d",
                        "Volume": "volume",
                        "RSI": "rsi"
                    }
                   
                    results_df = results_df.sort_values(
                        sort_mapping[sort_by],
                        ascending=False if sort_by != "RSI" else True
                    )
                   
                    # Display summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                   
                    with col1:
                        avg_score = results_df['technical_score'].mean()
                        st.metric("📊 Avg Technical Score", f"{avg_score:.1f}")
                   
                    with col2:
                        buy_count = len(results_df[results_df['recommendation'].isin(['BUY', 'STRONG BUY'])])
                        st.metric("📈 Buy Signals", f"{buy_count}/{len(results_df)}")
                   
                    with col3:
                        avg_change = results_df['price_change_1d'].mean()
                        st.metric("📊 Avg 1D Change", f"{avg_change:+.2f}%")
                   
                    with col4:
                        high_vol = len(results_df[results_df['volume'] > results_df['avg_volume'] * 1.5])
                        st.metric("📊 High Volume", f"{high_vol}/{len(results_df)}")
                   
                    # Main results table
                    st.subheader("📋 Watchlist Analysis Results")
                   
                    if show_details:
                        # Detailed view with all signals
                        for _, row in results_df.iterrows():
                            with st.expander(f"📊 {row['symbol']} - {row['recommendation']} (Score: {row['technical_score']:.1f})"):
                                col1, col2, col3 = st.columns(3)
                               
                                with col1:
                                    st.metric("💲 Price", f"${row['current_price']:.2f}", f"{row['price_change_1d']:+.2f}%")
                                    st.metric("📊 Volume", f"{row['volume']:,.0f}")
                               
                                with col2:
                                    st.metric("📈 RSI", f"{row['rsi']:.1f}")
                                    st.metric("📊 Tech Score", f"{row['technical_score']:.1f}/100")
                               
                                with col3:
                                    # Individual signals
                                    signals = row['signals']
                                    st.write("**Individual Signals:**")
                                    for indicator, signal in signals.items():
                                        if signal == "BUY":
                                            st.success(f"{indicator.upper()}: {signal}")
                                        elif signal == "SELL":
                                            st.error(f"{indicator.upper()}: {signal}")
                                        else:
                                            st.warning(f"{indicator.upper()}: {signal}")
                    else:
                        # Compact table view
                        display_df = results_df[['symbol', 'current_price', 'price_change_1d', 'volume',
                                               'rsi', 'technical_score', 'recommendation']].copy()
                       
                        display_df.columns = ['Symbol', 'Price', '1D Change %', 'Volume', 'RSI', 'Tech Score', 'Signal']
                        display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
                        display_df['1D Change %'] = display_df['1D Change %'].apply(lambda x: f"{x:+.2f}%")
                        display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,.0f}")
                        display_df['RSI'] = display_df['RSI'].apply(lambda x: f"{x:.1f}")
                        display_df['Tech Score'] = display_df['Tech Score'].apply(lambda x: f"{x:.1f}")
                       
                        def highlight_watchlist_signals(row):
                            if row['Signal'] in ['STRONG BUY', 'BUY']:
                                return ['background-color: #d4edda'] * len(row)
                            elif row['Signal'] in ['STRONG SELL', 'SELL']:
                                return ['background-color: #f8d7da'] * len(row)
                            else:
                                return ['background-color: #fff3cd'] * len(row)
                       
                        st.dataframe(
                            display_df.style.apply(highlight_watchlist_signals, axis=1),
                            use_container_width=True,
                            height=400
                        )
                   
                    # Visualization
                    col1, col2 = st.columns(2)
                   
                    with col1:
                        # Technical scores
                        fig1 = px.bar(
                            results_df,
                            x='symbol',
                            y='technical_score',
                            title="Technical Scores by Symbol",
                            color='technical_score',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                   
                    with col2:
                        # Signal distribution
                        signal_counts = results_df['recommendation'].value_counts()
                        fig2 = px.pie(
                            values=signal_counts.values,
                            names=signal_counts.index,
                            title="Signal Distribution"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                   
                    # Options analysis results (if requested)
                    if include_options and options_results:
                        st.subheader("📊 Options Flow Summary")
                       
                        options_summary = []
                        for opt in options_results:
                            signals = opt['signals']
                            call_vol = opt['total_call_volume']
                            put_vol = opt['total_put_volume']
                            cp_ratio = call_vol / put_vol if put_vol > 0 else float('inf')
                           
                            options_summary.append({
                                'Symbol': opt['symbol'],
                                'Call Volume': f"{call_vol:,.0f}",
                                'Put Volume': f"{put_vol:,.0f}",
                                'C/P Ratio': f"{cp_ratio:.2f}" if cp_ratio != float('inf') else "∞",
                                'Flow Signal': signals.get('flow', 'NEUTRAL'),
                                'Unusual Activity': signals.get('unusual_activity', 'NEUTRAL')
                            })
                       
                        options_df = pd.DataFrame(options_summary)
                       
                        def highlight_options_signals(row):
                            colors = []
                            for col in row.index:
                                if col in ['Flow Signal', 'Unusual Activity']:
                                    if row[col] == 'BUY':
                                        colors.append('background-color: #d4edda')
                                    elif row[col] == 'SELL':
                                        colors.append('background-color: #f8d7da')
                                    else:
                                        colors.append('background-color: #fff3cd')
                                else:
                                    colors.append('')
                            return colors
                       
                        st.dataframe(
                            options_df.style.apply(highlight_options_signals, axis=1),
                            use_container_width=True
                        )
                   
                    # Export functionality
                    st.subheader("💾 Export Results")
                   
                    col1, col2 = st.columns(2)
                   
                    with col1:
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Analysis (CSV)",
                            data=csv_data,
                            file_name=f"watchlist_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                   
                    with col2:
                        if include_options and options_results:
                            options_csv = pd.DataFrame(options_summary).to_csv(index=False)
                            st.download_button(
                                label="📥 Download Options Analysis (CSV)",
                                data=options_csv,
                                file_name=f"options_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
               
                else:
                    st.error("❌ Unable to analyze any symbols from your watchlist. Please check the symbols and try again.")
            else:
                st.warning("⚠️ Please enter at least one valid stock symbol.")
        else:
            st.warning("⚠️ Please enter your watchlist symbols.")


# Sidebar information
st.sidebar.markdown("---")
st.sidebar.subheader("About This Tool")
st.sidebar.markdown("""
**Features:**
- Real-time stock analysis
- Technical indicators (RSI, MACD, Bollinger Bands)
- Options flow analysis
- Advanced screening
- Buy/Sell signals
- Custom watchlists


**Data Sources:**
- Yahoo Finance (Free)
- Real-time quotes
- Options chains
- Historical data


**Technical Indicators:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (SMA/EMA)
- Stochastic Oscillator
- Volume Analysis


**Signal Generation:**
- Multi-factor analysis
- Weighted scoring system
- Risk assessment
- Market sentiment


**Disclaimer:**
This tool is for educational purposes only. Not financial advice. Always do your own research before making investment decisions.
""")


# Performance Tips
st.sidebar.markdown("---")
st.sidebar.subheader("Performance Tips")
st.sidebar.markdown("""
**For best results:**
- Limit watchlists to 20-30 symbols
- Use shorter analysis periods for faster loading
- Enable options analysis only when needed
- Clear browser cache if experiencing issues


**Signal Interpretation:**
- **STRONG BUY**: Multiple bullish signals
- **BUY**: Net bullish signals  
- **HOLD**: Mixed or neutral signals
- **SELL**: Net bearish signals
- **STRONG SELL**: Multiple bearish signals
""")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Trading Signals & Screener Pro • Built with Streamlit • Powered by Yahoo Finance</p>
    <p><small>This tool is for educational and informational purposes only. Not financial advice.</small></p>
    <p><small>Always conduct your own research and consider consulting with a financial advisor before making investment decisions.</small></p>
</div>
""", unsafe_allow_html=True)







