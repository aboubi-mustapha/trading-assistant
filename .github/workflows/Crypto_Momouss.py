# Crypto_Momouss.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands
import logging
from typing import Optional, List, Dict

# ======================
# CONFIGURATION
# ======================
class Config:
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
        "LINKUSDT": "Chainlink"
    }
    INTERVAL = "1d"
    API_URL = "https://api.binance.com/api/v3/klines"
    CACHE_TTL = 3600  # 1 heure
    REQUEST_TIMEOUT = 15  # Augmentation du timeout

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# FONCTIONS CORE
# ======================
@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Chargement des données...")
def fetch_crypto_data(symbol: str) -> Optional[pd.DataFrame]:
    """Récupère les données historiques depuis l'API Binance avec gestion d'erreur améliorée"""
    try:
        params = {
            'symbol': symbol,
            'interval': Config.INTERVAL,
            'limit': 300
        }
        
        response = requests.get(
            Config.API_URL,
            params=params,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'},
            timeout=Config.REQUEST_TIMEOUT
        )
        response.raise_for_status()

        df = pd.DataFrame(
            response.json(),
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Conversion des types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df[['date', 'open', 'high', 'low', 'close', 'volume']].iloc[:-1]

    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur réseau: {str(e)}")
        st.error("⚠️ Impossible de se connecter à l'API Binance. Vérifiez votre connexion.")
        return None
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}")
        st.error("🚨 Erreur lors du traitement des données")
        return None

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les indicateurs techniques avec gestion des NaN"""
    try:
        # Momentum
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        
        # Trend
        df['ma50'] = SMAIndicator(df['close'], window=50).sma_indicator()
        df['ma200'] = SMAIndicator(df['close'], window=200).sma_indicator()
        
        # Volatility
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        
        # Calculs personnalisés
        df['pct_change_1d'] = df['close'].pct_change() * 100
        df['volatility'] = df['close'].pct_change().rolling(14).std() * 100
        
        return df.dropna().copy()
    
    except Exception as e:
        logger.error(f"Erreur calculs: {str(e)}")
        st.error("🚨 Erreur dans les calculs techniques")
        return pd.DataFrame()

# ======================
# UI COMPONENTS
# ======================
def display_header(selected: str):
    """Affiche l'en-tête de la page"""
    st.title(f"📊 {Config.CRYPTO_LIST[selected]} - Analyse Expert")
    st.caption("Dernière mise à jour : " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

def display_metrics(df: pd.DataFrame):
    """Affiche les métriques principales"""
    latest = df.iloc[-1]
    cols = st.columns(4)
    with cols[0]:
        st.metric("Prix Actuel", f"${latest['close']:,.2f}", delta=f"{latest['pct_change_1d']:.1f}%")
    with cols[1]:
        st.metric("RSI (14j)", f"{latest['rsi']:.1f}", 
                help="RSI > 70 = Surachat, RSI < 30 = Survendu")
    with cols[2]:
        st.metric("Volatilité", f"{latest['volatility']:.1f}%")
    with cols[3]:
        st.metric("MA200", f"${latest['ma200']:,.2f}", 
                delta=f"{(latest['close']/latest['ma200']-1)*100:.1f}%")

def display_signals(signals: List[Dict]):
    """Affiche les signaux de trading"""
    if signals:
        st.subheader("🚨 Signaux de Trading", anchor="signaux")
        for signal in signals:
            color = "#FF4B4B" if signal["type"] == "buy" else "#00C853"
            with st.container():
                st.markdown(f"""
                <div style='
                    padding:1.2rem;
                    border-radius:8px;
                    margin:1rem 0;
                    border-left: 6px solid {color};
                    background-color: #f8f9fa;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                '>
                    <h4 style='color:{color}; margin:0;'>{signal["title"]}</h4>
                    <ul style='font-size:15px; color:#333; margin:0.5rem 0 0 1rem;'>
                        {''.join([f"<li>{d}</li>" for d in signal["details"]])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📌 Aucun signal détecté - Position neutre recommandée", icon="ℹ️")

# ======================
# FONCTION PRINCIPALE
# ======================
def main():
    st.set_page_config(
        page_title="Crypto Analyst Pro",
        layout="wide",
        page_icon="📈",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        selected = st.selectbox(
            "Sélectionnez une cryptomonnaie :",
            options=list(Config.CRYPTO_LIST.keys()),
            format_func=lambda x: f"{Config.CRYPTO_LIST[x]} ({x})",
            help="Choisissez une paire de trading parmi les cryptos principales"
        )
        st.markdown("---")
        st.markdown("**ℹ️ À propos**\n\nOutil d'analyse technique pour cryptomonnaies utilisant les données Binance.")

    # Data processing
    df = fetch_crypto_data(selected)
    if df is None or df.empty:
        st.warning("Données non disponibles pour cette cryptomonnaie")
        return

    df = calculate_technical_indicators(df)
    
    # Main display
    display_header(selected)
    display_metrics(df)
    
    # Signaux
    signals = generate_signals(df)
    display_signals(signals)

    # Analyse technique détaillée
    with st.expander("📈 Analyse Technique Avancée", expanded=True):
        tab1, tab2 = st.tabs(["Graphique des prix", "Indicateurs clés"])
        
        with tab1:
            st.area_chart(
                df.set_index('date')['close'].tail(90),
                use_container_width=True,
                color="#1f77b4"
            )
            
        with tab2:
            cols = st.columns(3)
            latest = df.iloc[-1]
            with cols[0]:
                st.write("**Bandes de Bollinger**")
                st.metric("Bande Supérieure", f"${latest['bb_upper']:.2f}")
                st.metric("Bande Inférieure", f"${latest['bb_lower']:.2f}")
            with cols[1]:
                st.write("**Moyennes Mobiles**")
                st.metric("MA50", f"${latest['ma50']:.2f}")
                st.metric("MA200", f"${latest['ma200']:.2f}")
            with cols[2]:
                st.write("**Autres indicateurs**")
                st.metric("Volume 24h", f"${latest['volume']:,.0f}")
                st.metric("Variation 7j", f"{df['pct_change_1d'].tail(7).sum():.1f}%")

if __name__ == "__main__":
    main()# Crypto_Momouss.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands
import logging
from typing import Optional, List, Dict

# ======================
# CONFIGURATION
# ======================
class Config:
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
        "LINKUSDT": "Chainlink"
    }
    INTERVAL = "1d"
    API_URL = "https://api.binance.com/api/v3/klines"
    CACHE_TTL = 3600  # 1 heure
    REQUEST_TIMEOUT = 15  # Augmentation du timeout

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# FONCTIONS CORE
# ======================
@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Chargement des données...")
def fetch_crypto_data(symbol: str) -> Optional[pd.DataFrame]:
    """Récupère les données historiques depuis l'API Binance avec gestion d'erreur améliorée"""
    try:
        params = {
            'symbol': symbol,
            'interval': Config.INTERVAL,
            'limit': 300
        }
        
        response = requests.get(
            Config.API_URL,
            params=params,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'},
            timeout=Config.REQUEST_TIMEOUT
        )
        response.raise_for_status()

        df = pd.DataFrame(
            response.json(),
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Conversion des types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df[['date', 'open', 'high', 'low', 'close', 'volume']].iloc[:-1]

    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur réseau: {str(e)}")
        st.error("⚠️ Impossible de se connecter à l'API Binance. Vérifiez votre connexion.")
        return None
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}")
        st.error("🚨 Erreur lors du traitement des données")
        return None

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les indicateurs techniques avec gestion des NaN"""
    try:
        # Momentum
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        
        # Trend
        df['ma50'] = SMAIndicator(df['close'], window=50).sma_indicator()
        df['ma200'] = SMAIndicator(df['close'], window=200).sma_indicator()
        
        # Volatility
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        
        # Calculs personnalisés
        df['pct_change_1d'] = df['close'].pct_change() * 100
        df['volatility'] = df['close'].pct_change().rolling(14).std() * 100
        
        return df.dropna().copy()
    
    except Exception as e:
        logger.error(f"Erreur calculs: {str(e)}")
        st.error("🚨 Erreur dans les calculs techniques")
        return pd.DataFrame()

# ======================
# UI COMPONENTS
# ======================
def display_header(selected: str):
    """Affiche l'en-tête de la page"""
    st.title(f"📊 {Config.CRYPTO_LIST[selected]} - Analyse Expert")
    st.caption("Dernière mise à jour : " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

def display_metrics(df: pd.DataFrame):
    """Affiche les métriques principales"""
    latest = df.iloc[-1]
    cols = st.columns(4)
    with cols[0]:
        st.metric("Prix Actuel", f"${latest['close']:,.2f}", delta=f"{latest['pct_change_1d']:.1f}%")
    with cols[1]:
        st.metric("RSI (14j)", f"{latest['rsi']:.1f}", 
                help="RSI > 70 = Surachat, RSI < 30 = Survendu")
    with cols[2]:
        st.metric("Volatilité", f"{latest['volatility']:.1f}%")
    with cols[3]:
        st.metric("MA200", f"${latest['ma200']:,.2f}", 
                delta=f"{(latest['close']/latest['ma200']-1)*100:.1f}%")

def display_signals(signals: List[Dict]):
    """Affiche les signaux de trading"""
    if signals:
        st.subheader("🚨 Signaux de Trading", anchor="signaux")
        for signal in signals:
            color = "#FF4B4B" if signal["type"] == "buy" else "#00C853"
            with st.container():
                st.markdown(f"""
                <div style='
                    padding:1.2rem;
                    border-radius:8px;
                    margin:1rem 0;
                    border-left: 6px solid {color};
                    background-color: #f8f9fa;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                '>
                    <h4 style='color:{color}; margin:0;'>{signal["title"]}</h4>
                    <ul style='font-size:15px; color:#333; margin:0.5rem 0 0 1rem;'>
                        {''.join([f"<li>{d}</li>" for d in signal["details"]])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📌 Aucun signal détecté - Position neutre recommandée", icon="ℹ️")

# ======================
# FONCTION PRINCIPALE
# ======================
def main():
    st.set_page_config(
        page_title="Crypto Analyst Pro",
        layout="wide",
        page_icon="📈",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        selected = st.selectbox(
            "Sélectionnez une cryptomonnaie :",
            options=list(Config.CRYPTO_LIST.keys()),
            format_func=lambda x: f"{Config.CRYPTO_LIST[x]} ({x})",
            help="Choisissez une paire de trading parmi les cryptos principales"
        )
        st.markdown("---")
        st.markdown("**ℹ️ À propos**\n\nOutil d'analyse technique pour cryptomonnaies utilisant les données Binance.")

    # Data processing
    df = fetch_crypto_data(selected)
    if df is None or df.empty:
        st.warning("Données non disponibles pour cette cryptomonnaie")
        return

    df = calculate_technical_indicators(df)
    
    # Main display
    display_header(selected)
    display_metrics(df)
    
    # Signaux
    signals = generate_signals(df)
    display_signals(signals)

    # Analyse technique détaillée
    with st.expander("📈 Analyse Technique Avancée", expanded=True):
        tab1, tab2 = st.tabs(["Graphique des prix", "Indicateurs clés"])
        
        with tab1:
            st.area_chart(
                df.set_index('date')['close'].tail(90),
                use_container_width=True,
                color="#1f77b4"
            )
            
        with tab2:
            cols = st.columns(3)
            latest = df.iloc[-1]
            with cols[0]:
                st.write("**Bandes de Bollinger**")
                st.metric("Bande Supérieure", f"${latest['bb_upper']:.2f}")
                st.metric("Bande Inférieure", f"${latest['bb_lower']:.2f}")
            with cols[1]:
                st.write("**Moyennes Mobiles**")
                st.metric("MA50", f"${latest['ma50']:.2f}")
                st.metric("MA200", f"${latest['ma200']:.2f}")
            with cols[2]:
                st.write("**Autres indicateurs**")
                st.metric("Volume 24h", f"${latest['volume']:,.0f}")
                st.metric("Variation 7j", f"{df['pct_change_1d'].tail(7).sum():.1f}%")

if __name__ == "__main__":
    main()
