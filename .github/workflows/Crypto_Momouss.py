import streamlit as st
import requests
import pandas as pd
from ta import momentum, trend, volatility
import numpy as np
import os
os.environ['STREAMLIT_SERVER_PORT'] = '8501'
os.environ['STREAMLIT_SERVER_ADDRESS'] = '192.168.1.25'
# ======================
# CONFIGURATION GÉNÉRALE
# ======================
st.set_page_config(
    page_title="Crypto Trading Analyst", 
    layout="wide",
    page_icon="📊"
)
st.title("📊 Crypto Trading Analyst - Stratégie Contraire")

# Styles CSS personnalisés
st.markdown("""
<style>
    .signal-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .buy-signal {
        border: 3px solid #FF4444;
        background-color: #FFEEEE;
    }
    .sell-signal {
        border: 3px solid #00C853;
        background-color: #E8F5E9;
    }
    .metric-label {
        font-size: 1.1rem !important;
        color: #616161 !important;
    }
    .metric-value {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# PARAMÈTRES
# ======================
CRYPTO_LIST = {
    "BTCUSDT": "Bitcoin",
    "ETHUSDT": "Ethereum",
    "BNBUSDT": "Binance Coin",
    "SOLUSDT": "Solana",
    "XRPUSDT": "Ripple",
    "ADAUSDT": "Cardano",
    "DOGEUSDT": "Dogecoin",
    "AVAXUSDT": "Avalanche",
    "DOTUSDT": "Polkadot",
    "TRXUSDT": "TRON",
    "LINKUSDT": "Chainlink",
    "MATICUSDT": "Polygon",
    "SHIBUSDT": "Shiba Inu",
    "LTCUSDT": "Litecoin",
    "UNIUSDT": "Uniswap",
    "ATOMUSDT": "Cosmos",
    "XLMUSDT": "Stellar",
    "ETCUSDT": "Ethereum Classic",
    "XMRUSDT": "Monero",
    "FILUSDT": "Filecoin"
}

INTERVAL = "1d"

# ======================
# FONCTIONS CORE
# ======================
@st.cache_data(ttl=3600)
def fetch_crypto_data(symbol="BTCUSDT"):
    """Récupère les données historiques depuis l'API Binance"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': symbol, 'interval': INTERVAL, 'limit': 300}
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Création du DataFrame
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(response.json(), columns=cols + ['ignore']*6)
        df = df[cols]
        
        # Conversion des types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df[['date', 'open', 'high', 'low', 'close', 'volume']].iloc[:-1]

    except Exception as e:
        st.error(f"Erreur de récupération des données : {str(e)}")
        return None

def calculate_technical_indicators(df):
    """Calcule les indicateurs techniques"""
    try:
        # Indicateurs de momentum
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # Moyennes mobiles
        df['ma50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ma200'] = ta.trend.sma_indicator(df['close'], window=200)
        
        # Calculs personnalisés
        df['pct_change_1d'] = df['close'].pct_change() * 100
        df['pct_change_3d'] = df['close'].pct_change(3) * 100
        df['volatility'] = df['close'].pct_change().rolling(14).std() * 100
        df['ma50_slope'] = df['ma50'].diff(3) / df['ma50'].shift(3) * 100
        
        return df.dropna()
    
    except Exception as e:
        st.error(f"Erreur de calcul des indicateurs : {str(e)}")
        return None

def generate_trading_signals(df):
    """Génère les signaux de trading selon la stratégie contraire"""
    if df is None or len(df) < 4:
        return []

    latest = df.iloc[-1]
    signals = []

    # Signal d'achat - Correction brutale
    buy_condition = (
        (latest['pct_change_1d'] < -5 or latest['pct_change_3d'] < -10) and
        (latest['close'] > latest['ma200']) and
        (latest['volatility'] > 15)
    )
    
    if buy_condition:
        signals.append({
            "type": "buy",
            "title": "🟥 ACHAT - Correction Brutale Détectée",
            "details": [
                f"Variation 1j : {latest['pct_change_1d']:.1f}%",
                f"Variation 3j : {latest['pct_change_3d']:.1f}%",
                f"Volatilité : {latest['volatility']:.1f}%",
                f"Position vs MA200 : +{(latest['close']/latest['ma200']-1)*100:.1f}%"
            ]
        })

    # Signal de vente - Rallye excessif
    sell_condition = (
        (latest['pct_change_1d'] > 5 or latest['pct_change_3d'] > 15) and
        (latest['ma50_slope'] > 2) and
        (latest['rsi'] > 70)
    )
    
    if sell_condition:
        signals.append({
            "type": "sell",
            "title": "🟩 VENTE - Rallye Excessif Détecté",
            "details": [
                f"Variation 1j : {latest['pct_change_1d']:.1f}%",
                f"Variation 3j : {latest['pct_change_3d']:.1f}%",
                f"Pente MA50 : {latest['ma50_slope']:.1f}%",
                f"RSI : {latest['rsi']:.1f}"
            ]
        })

    return signals

# ======================
# INTERFACE UTILISATEUR
# ======================
def main():
    # Sélection de la crypto
    st.sidebar.header("⚙️ Configuration")
    selected_crypto = st.sidebar.radio(
        "Sélectionnez une cryptomonnaie :",
        options=list(CRYPTO_LIST.keys()),
        format_func=lambda x: f"{CRYPTO_LIST[x]} ({x})",
        index=0
    )

    # Récupération des données
    df = fetch_crypto_data(selected_crypto)
    if df is None:
        st.warning("Données non disponibles pour cette cryptomonnaie")
        return

    # Calcul des indicateurs
    df = calculate_technical_indicators(df)
    if df is None:
        st.error("Erreur dans le calcul des indicateurs techniques")
        return

    # En-tête principal
    current_price = df['close'].iloc[-1]
    st.header(f"""
    {CRYPTO_LIST[selected_crypto]} ({selected_crypto}) 
    - **${current_price:,.2f}**
    """)

    # Affichage des signaux
    signals = generate_trading_signals(df)
    if signals:
        st.subheader("🚨 Signaux de Trading")
        for signal in signals:
            css_class = "buy-signal" if signal["type"] == "buy" else "sell-signal"
            st.markdown(f"""
            <div class="signal-box {css_class}">
                <h3 style='margin-top:0;'>{signal["title"]}</h3>
                <ul style='font-size:16px;'>
                    {''.join([f"<li>{detail}</li>" for detail in signal["details"]])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📌 Aucun signal de trading détecté - Position neutre recommandée")

    # Section d'analyse technique
    st.subheader("📈 Analyse Technique")
    with st.expander("Indicateurs Clés", expanded=True):
        cols = st.columns(4)
        metrics = [
            ("RSI (14j)", df['rsi'].iloc[-1], "#FF6D00"),
            ("Volatilité (14j)", df['volatility'].iloc[-1], "#2962FF"),
            ("Pente MA50", df['ma50_slope'].iloc[-1], "#00BFA5"),
            ("Distance MA200", (df['close'].iloc[-1]/df['ma200'].iloc[-1]-1)*100, "#7B1FA2")
        ]
        
        for (label, value, color), col in zip(metrics, cols):
            with col:
                st.markdown(f"""
                <div style='padding:1rem; border-radius:8px; border:2px solid {color}30;'>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-value' style='color:{color};'>{value:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

    # Graphique des prix
    st.subheader("📉 Historique des Prix (60 jours)")
    st.line_chart(
        df.set_index('date')['close'].tail(60),
        use_container_width=True,
        color="#2196F3"
    )

if __name__ == "__main__":
    main()
