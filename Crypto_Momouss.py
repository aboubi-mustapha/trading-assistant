import streamlit as st
import requests
import pandas as pd
import ta
import numpy as np
from datetime import datetime, timedelta
import logging  # Module de logging

# ======================
# CONFIGURATION GÉNÉRALE
# ======================
st.set_page_config(
    page_title="Crypto Analyst Pro IA",
    layout="wide",
    page_icon="📊"
)
st.title("🔍 Crypto Analyst Multi-Sources & IA")

# --- Logging Configuration ---
logging.basicConfig(filename='./logs.txt',  # Fichier log dans le même répertoire
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Application Crypto Analyst Pro IA démarrée") # Log au démarrage

# Styles CSS personnalisés (intégrés directement)
st.markdown("""
<style>
    .source-badge {
        padding: 0.3rem 0.7rem;
        border-radius: 15px;
        background-color: #e3f2fd;
        display: inline-block;
        margin: 0.2rem;
    }
    .alert-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        background-color: #fef5e7;
        border: 1px solid #ffe0b2;
        color: #795548;
    }
    .advice-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        background-color: #f0f4c3;
        border: 1px solid #d4e157;
        color: #558b2f;
    }
    .buy-signal {
        background-color: #c8e6c9; /* Vert clair */
        border: 1px solid #a5d6a7;
        color: #388e3c;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .sell-signal {
        background-color: #ffcdd2; /* Rouge clair */
        border: 1px solid #ef9a9a;
        color: #d32f2f;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .neutral-signal {
        background-color: #eeeeee; /* Gris clair */
        border: 1px solid #bdbdbd;
        color: #424242;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .error-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        background-color: #ffebee;
        border: 1px solid #ef5350;
        color: #b71c1c;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# PARAMÈTRES (CONSTANTES)
# ======================
CRYPTO_LIST = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "BNB": "Binance Coin", # Même si Binance API n'est plus utilisée pour le moment, on garde BNB dans la liste des cryptos disponibles
    "SOL": "Solana",
    "XRP": "Ripple"
}

INDICATORS = {
    "RSI": ta.momentum.RSIIndicator,
    "MACD": ta.trend.MACD,
    "Bollinger Bands": ta.volatility.BollingerBands
}

MA_WINDOWS = [20, 50, 100]

# ======================
# FONCTIONS DATA FETCHING (SANS BINANCE API)
# ======================
def fetch_coingecko_data(coin_id):
    """Récupère les données depuis CoinGecko API avec gestion des erreurs"""
    logging.info(f"Début de la récupération des données CoinGecko pour {coin_id}")
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': '90',
            'interval': 'daily'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        if 'prices' not in data:
            log_message = f"Réponse CoinGecko malformée (prix manquants) pour {coin_id}"
            logging.warning(log_message)
            st.error(f"<div class='error-box'>{log_message}. Veuillez réessayer ou choisir une autre cryptomonnaie. Si le problème persiste, CoinGecko pourrait être temporairement indisponible.</div>", unsafe_allow_html=True)
            return None

        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        prices['date'] = pd.to_datetime(prices['timestamp'], unit='ms')
        logging.info(f"Données CoinGecko récupérées avec succès pour {coin_id}")
        return prices[['date', 'price']].rename(columns={'price': 'close'})

    except requests.exceptions.HTTPError as http_err:
        log_message = f"Erreur HTTP CoinGecko pour {coin_id}: {http_err}"
        logging.error(log_message)
        st.error(f"<div class='error-box'>{log_message}. Veuillez vérifier votre connexion internet et réessayer. Si le problème persiste, CoinGecko pourrait être temporairement indisponible.</div>", unsafe_allow_html=True)
        return None
    except requests.exceptions.RequestException as req_err:
        log_message = f"Erreur de requête CoinGecko pour {coin_id}: {req_err}"
        logging.error(log_message)
        st.error(f"<div class='error-box'>{log_message}. Veuillez vérifier votre connexion internet et réessayer. Si le problème persiste, CoinGecko pourrait être temporairement indisponible.</div>", unsafe_allow_html=True)
        return None
    except Exception as e:
        log_message = f"Erreur inattendue CoinGecko pour {coin_id}: {str(e)}"
        logging.critical(log_message)
        st.error(f"<div class='error-box'>{log_message}. Erreur inattendue lors de la récupération des données CoinGecko. Veuillez réessayer plus tard.</div>", unsafe_allow_html=True)
        return None


def fetch_kraken_data(symbol):
    """Récupère les données depuis Kraken API (corrigé) avec gestion des erreurs"""
    logging.info(f"Début de la récupération des données Kraken pour {symbol}")
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        params = {
            'pair': f"{symbol}USD",
            'interval': 1440,
            'since': int((datetime.now() - timedelta(days=90)).timestamp())
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        if 'result' not in data or not data['result']:
            log_message = f"Réponse Kraken API malformée ou vide pour {symbol}"
            logging.warning(log_message)
            st.error(f"<div class='error-box'>{log_message}. Veuillez réessayer ou choisir une autre cryptomonnaie. Si le problème persiste, Kraken pourrait être temporairement indisponible.</div>", unsafe_allow_html=True)
            return None

        ohlc_data = list(data['result'].values())[0]
        if not ohlc_data:
            log_message = f"Données Kraken vides pour {symbol}"
            logging.warning(log_message)
            st.error(f"<div class='error-box'>{log_message}. Veuillez réessayer ou choisir une autre cryptomonnaie. Si le problème persiste, Kraken pourrait être temporairement indisponible.</div>", unsafe_allow_html=True)
            return None

        df = pd.DataFrame(ohlc_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close',
            'volume', 'wap', 'count'
        ])
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        df['close'] = pd.to_numeric(df['close'])
        logging.info(f"Données Kraken récupérées avec succès pour {symbol}")
        return df[['date', 'close']]

    except requests.exceptions.HTTPError as http_err:
        log_message = f"Erreur HTTP Kraken pour {symbol}: {http_err}"
        logging.error(log_message)
        st.error(f"<div class='error-box'>{log_message}. Veuillez vérifier votre connexion internet et réessayer. Si le problème persiste, Kraken pourrait être temporairement indisponible.</div>", unsafe_allow_html=True)
        return None
    except requests.exceptions.RequestException as req_err:
        log_message = f"Erreur de requête Kraken pour {symbol}: {req_err}"
        logging.error(log_message)
        st.error(f"<div class='error-box'>{log_message}. Veuillez vérifier votre connexion internet et réessayer. Si le problème persiste, Kraken pourrait être temporairement indisponible.</div>", unsafe_allow_html=True)
        return None
    except Exception as e:
        log_message = f"Erreur inattendue Kraken pour {symbol}: {str(e)}"
        logging.critical(log_message)
        st.error(f"<div class='error-box'>{log_message}. Erreur inattendue lors de la récupération des données Kraken. Veuillez réessayer plus tard.</div>", unsafe_allow_html=True)
        return None


def get_crypto_data(symbol):
    """Agrège les données avec fallback et indication de la source (sans Binance)"""
    sources = {
        'CoinGecko': lambda: fetch_coingecko_data(symbol.lower()),
        'Kraken': lambda: fetch_kraken_data(symbol) # Binance API retirée
    }

    for source_name, source_func in sources.items():
        df = source_func()
        if df is not None and not df.empty:
            st.markdown(f"<div class='source-badge'>Source: {source_name}</div>", unsafe_allow_html=True)
            return df

    error_message = "Aucune source de données disponible pour cette cryptomonnaie après tentative avec CoinGecko et Kraken."
    logging.error(error_message)
    st.error(f"<div class='error-box'>{error_message} Veuillez réessayer plus tard ou choisir une autre cryptomonnaie.</div>", unsafe_allow_html=True)
    return None


# ======================
# ANALYSE TECHNIQUE AMÉLIORÉE (TOUTES DANS APP.PY)
# ======================
def calculate_indicators(df, indicators_config):
    """Calcule les indicateurs techniques sélectionnés"""
    logging.info("Début du calcul des indicateurs techniques")
    try:
        for indicator_name, indicator_params in indicators_config.items():
            if indicator_name == "RSI":
                df['rsi'] = INDICATORS["RSI"](df['close'], window=indicator_params.get('window', 14)).rsi()
            elif indicator_name == "MACD":
                macd = INDICATORS["MACD"](df['close'], window_fast=indicator_params.get('window_fast', 12), window_slow=indicator_params.get('window_slow', 26), window_sign=indicator_params.get('window_sign', 9))
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                df['macd_hist'] = macd.macd_diff()
            elif indicator_name == "Bollinger Bands":
                bb = INDICATORS["Bollinger Bands"](df['close'], window=indicator_params.get('window', 20), window_dev=indicator_params.get('window_dev', 2))
                df['bb_upper'] = bb.bollinger_hband()
                df['bb_lower'] = bb.bollinger_lband()
                df['bb_mid'] = bb.bollinger_mavg()
        for window in MA_WINDOWS:
            df[f'ma{window}'] = ta.trend.sma_indicator(df['close'], window=window)

        df['volatility'] = df['close'].pct_change().rolling(14).std() * 100
        logging.info("Calcul des indicateurs techniques terminé avec succès")
        return df.dropna()
    except Exception as e:
        log_message = f"Erreur de calcul d'indicateurs: {str(e)}"
        logging.error(log_message)
        st.error(f"<div class='error-box'>{log_message}. Erreur lors du calcul des indicateurs techniques. Veuillez réessayer.</div>", unsafe_allow_html=True)
        return None

# ======================
# CONSEILS DE TRADING (AGENT IA - SIMULATION) (TOUTES DANS APP.PY)
# ======================
def get_ia_trading_advice(df):
    """Simule des conseils de trading basés sur une IA (simplifié) avec prise en compte des positions du marché"""
    logging.info("Génération des conseils de trading IA")
    if df is None or df.empty or len(df) < 2:
        log_message = "Données insuffisantes pour générer des conseils IA."
        logging.warning(log_message)
        return f"<div class='alert-box'>{log_message}</div>"

    last_rsi = df['rsi'].iloc[-1]
    last_macd = df['macd'].iloc[-1]
    last_macd_signal = df['macd_signal'].iloc[-1]
    last_price = df['close'].iloc[-1]
    ma20 = df['ma20'].iloc[-1]
    ma50 = df['ma50'].iloc[-1]

    # Analyse de la position du marché (variation de prix récente)
    price_change = df['close'].iloc[-1] - df['close'].iloc[-2]

    advice_messages = []
    signal_type = "neutral-signal" # Signal neutre par défaut

    if price_change < 0:
        advice_messages.append("🔴 **Baisse récente du prix :** Surveiller opportunité d'achat si d'autres indicateurs confirment.")
        signal_type = "buy-signal" # Signal d'achat potentiel si baisse
    elif price_change > 0:
        advice_messages.append("🟢 **Hausse récente du prix :**  Potentiel signal de vente si surachat ou prudence si résistance approche.")
        signal_type = "sell-signal" # Signal de vente potentiel si hausse

    if last_rsi > 70:
        advice_messages.append("⚠️ **RSI élevé (surachat) :** Possible signal de vente ou de prudence.")
        if signal_type == "neutral-signal": signal_type = "sell-signal" # Priorité au signal de vente si RSI suracheté
    elif last_rsi < 30:
        advice_messages.append("✅ **RSI bas (survente) :** Possible signal d'achat.")
        if signal_type == "neutral-signal": signal_type = "buy-signal" # Priorité au signal d'achat si RSI survendu

    if last_macd > last_macd_signal and last_macd_hist > 0:
        advice_messages.append("📈 **MACD haussier :** Momentum positif.")
    elif last_macd < last_macd_signal and last_macd_hist < 0:
        advice_messages.append("📉 **MACD baissier :** Momentum négatif.")

    if last_price > ma20 and last_price > ma50:
        advice_messages.append("🐂 **Prix au-dessus des MAs 20 et 50 :** Tendance potentiellement haussière.")
    elif last_price < ma20 and last_price < ma50:
        advice_messages.append("🐻 **Prix en-dessous des MAs 20 et 50 :** Tendance potentiellement baissière.")

    if not advice_messages:
        advice = "Neutre ou signaux mixtes. Analyse plus approfondie nécessaire."
        signal_type = "neutral-signal"
    else:
        advice = "\n".join(advice_messages)
        logging.info(f"Conseils IA générés : Signal {signal_type}, Conseils : {advice}") # Log des conseils générés

    return f"<div class='advice-box {signal_type}'><h3>💡 Conseils de Trading IA (Simulé)</h3><p>{advice}</p><p><b>Dernier RSI (14j):</b> {last_rsi:.2f} | <b>Dernier MACD:</b> {last_macd:.2f} | <b>Dernier Signal MACD:</b> {last_macd_signal:.2f}</p><p><b>MA20:</b> {ma20:.2f} | <b>MA50:</b> {ma50:.2f} | <b>Prix actuel:</b> {last_price:.2f}</p></div>"


# ======================
# INTERFACE UTILISATEUR (TOUTES DANS APP.PY)
# ======================
def main():
    st.sidebar.header("⚙️ Configuration")
    selected_crypto = st.sidebar.selectbox(
        "Cryptomonnaie:",
        options=list(CRYPTO_LIST.keys()),
        format_func=lambda x: CRYPTO_LIST[x]
    )

    # Configuration des indicateurs dans la sidebar (plus flexible)
    st.sidebar.subheader("⚙️ Indicateurs Techniques")
    indicators_config = {}
    if st.sidebar.checkbox("RSI", value=True):
        indicators_config["RSI"] = {'window': st.sidebar.slider("Période RSI", 2, 30, 14)}
    if st.sidebar.checkbox("MACD"):
        indicators_config["MACD"] = {
            'window_fast': st.sidebar.slider("Période MACD Rapide", 2, 30, 12),
            'window_slow': st.sidebar.slider("Période MACD Lent", 10, 50, 26),
            'window_sign': st.sidebar.slider("Période Signal MACD", 2, 20, 9)
        }
    if st.sidebar.checkbox("Bollinger Bands"):
        indicators_config["Bollinger Bands"] = {
            'window': st.sidebar.slider("Période BB", 10, 50, 20),
            'window_dev': st.sidebar.slider("Écart-type BB", 1, 5, 2)
        }


    df = get_crypto_data(selected_crypto)
    if df is None:
        st.markdown("<div class='alert-box'>Impossible de récupérer les données. Veuillez réessayer plus tard ou choisir une autre cryptomonnaie.</div>", unsafe_allow_html=True)
        return

    df_indicators = calculate_indicators(df.copy(), indicators_config) # Calculer les indicateurs après la récupération des données
    if df_indicators is None:
        st.markdown("<div class='alert-box'>Erreur lors du calcul des indicateurs techniques.</div>", unsafe_allow_html=True)
        return

    current_price = df_indicators['close'].iloc[-1]
    st.header(f"{CRYPTO_LIST[selected_crypto]} - ${current_price:,.2f}")

    ia_advice_html = get_ia_trading_advice(df_indicators)
    st.markdown(ia_advice_html, unsafe_allow_html=True)


    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Indicateurs")
        if 'rsi' in df_indicators.columns:
            st.metric("RSI (14j)", f"{df_indicators['rsi'].iloc[-1]:.1f}")
        if 'volatility' in df_indicators.columns:
            st.metric("Volatilité (14j)", f"{df_indicators['volatility'].iloc[-1]:.1f}%")
        if 'macd' in df_indicators.columns:
             st.metric("MACD", f"{df_indicators['macd'].iloc[-1]:.2f}")

    with col2:
        st.subheader("📉 Moyennes Mobiles")
        for window in MA_WINDOWS:
            st.metric(f"MA{window}", f"{df_indicators[f'ma{window}'].iloc[-1]:,.2f}")
        if 'bb_upper' in df_indicators.columns:
            st.metric("BB Upper (20)", f"{df_indicators['bb_upper'].iloc[-1]:,.2f}")
        if 'bb_lower' in df_indicators.columns:
            st.metric("BB Lower (20)", f"{df_indicators['bb_lower'].iloc[-1]:,.2f}")


    st.subheader("📊 Historique des Prix avec Moyennes Mobiles")
    chart_data = df_indicators.set_index('date')[['close']]
    for window in MA_WINDOWS:
        if f'ma{window}' in df_indicators.columns:
            chart_data[f'MA{window}'] = df_indicators[f'ma{window}']

    st.line_chart(chart_data)

    if 'Bollinger Bands' in indicators_config and 'bb_upper' in df_indicators.columns:
        st.subheader("📊 Bollinger Bands")
        bb_chart_data = df_indicators.set_index('date')[['close', 'bb_upper', 'bb_lower', 'bb_mid']]
        st.line_chart(bb_chart_data)

    if 'MACD' in indicators_config and 'macd' in df_indicators.columns:
        st.subheader("📊 MACD")
        macd_chart_data = df_indicators.set_index('date')[['macd', 'macd_signal', 'macd_hist']]
        st.bar_chart(macd_chart_data['macd_hist'])
        st.line_chart(macd_chart_data[['macd', 'macd_signal']])


    st.markdown("<div class='alert-box'><b>Disclaimer important :</b> Les conseils de trading fournis par cet outil (IA simulée) sont uniquement à des fins informatives et éducatives. Ils ne constituent en aucun cas des conseils financiers personnalisés ou une incitation à l'investissement. Le trading de cryptomonnaies est risqué et vous pouvez perdre de l'argent.  N'investissez jamais de l'argent que vous ne pouvez pas vous permettre de perdre. Faites toujours vos propres recherches et consultez un conseiller financier professionnel avant de prendre toute décision d'investissement.  Utilisez cet outil à vos propres risques.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
                
import streamlit as st
import requests
import pandas as pd
import ta
import numpy as np
from datetime import datetime, timedelta
import logging  # Module de logging

# ======================
# CONFIGURATION GÉNÉRALE
# ======================
st.set_page_config(
    page_title="Crypto Analyst Pro IA",
    layout="wide",
    page_icon="📊"
)
st.title("🔍 Crypto Analyst Multi-Sources & IA")

# --- Logging Configuration ---
logging.basicConfig(filename='./logs.txt',  # Fichier log dans le même répertoire (ajustable pour Streamlit Cloud si besoin)
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Application Crypto Analyst Pro IA démarrée") # Log au démarrage

# Styles CSS personnalisés (intégrés directement)
st.markdown("""
<style>
    .source-badge {
        padding: 0.3rem 0.7rem;
        border-radius: 15px;
        background-color: #e3f2fd;
        display: inline-block;
        margin: 0.2rem;
    }
    .alert-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        background-color: #fef5e7;
        border: 1px solid #ffe0b2;
        color: #795548;
    }
    .advice-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        background-color: #f0f4c3;
        border: 1px solid #d4e157;
        color: #558b2f;
    }
    .buy-signal {
        background-color: #c8e6c9; /* Vert clair */
        border: 1px solid #a5d6a7;
        color: #388e3c;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .sell-signal {
        background-color: #ffcdd2; /* Rouge clair */
        border: 1px solid #ef9a9a;
        color: #d32f2f;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .neutral-signal {
        background-color: #eeeeee; /* Gris clair */
        border: 1px solid #bdbdbd;
        color: #424242;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .error-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        background-color: #ffebee;
        border: 1px solid #ef5350;
        color: #b71c1c;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# PARAMÈTRES (CONSTANTES)
# ======================
CRYPTO_LIST = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "BNB": "Binance Coin",
    "SOL": "Solana",
    "XRP": "Ripple"
}

INDICATORS = {
    "RSI": ta.momentum.RSIIndicator,
    "MACD": ta.trend.MACD,
    "Bollinger Bands": ta.volatility.BollingerBands
}

MA_WINDOWS = [20, 50, 100]

# ======================
# FONCTIONS DATA FETCHING (TOUTES DANS APP.PY)
# ======================
def fetch_coingecko_data(coin_id):
    """Récupère les données depuis CoinGecko API avec gestion des erreurs"""
    logging.info(f"Début de la récupération des données CoinGecko pour {coin_id}")
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': '90',
            'interval': 'daily'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        if 'prices' not in data:
            logging.warning(f"Réponse CoinGecko malformée (prix manquants) pour {coin_id}")
            st.error(f"<div class='error-box'>CoinGecko API response malformed (prices missing) pour {coin_id}.</div>", unsafe_allow_html=True)
            return None

        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        prices['date'] = pd.to_datetime(prices['timestamp'], unit='ms')
        logging.info(f"Données CoinGecko récupérées avec succès pour {coin_id}")
        return prices[['date', 'price']].rename(columns={'price': 'close'})

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"Erreur HTTP CoinGecko pour {coin_id}: {http_err}")
        st.error(f"<div class='error-box'>Erreur HTTP CoinGecko pour {coin_id}: {http_err}</div>", unsafe_allow_html=True)
        return None
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Erreur de requête CoinGecko pour {coin_id}: {req_err}")
        st.error(f"<div class='error-box'>Erreur de requête CoinGecko pour {coin_id}: {req_err}</div>", unsafe_allow_html=True)
        return None
    except Exception as e:
        logging.critical(f"Erreur inattendue CoinGecko pour {coin_id}: {str(e)}")
        st.error(f"<div class='error-box'>Erreur inattendue CoinGecko pour {coin_id}: {str(e)}</div>", unsafe_allow_html=True)
        return None


def fetch_binance_data(symbol):
    """Récupère les données depuis Binance API avec gestion des erreurs"""
    logging.info(f"Début de la récupération des données Binance pour {symbol}")
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': f"{symbol}USDT",
            'interval': '1d',
            'limit': 90
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        df = pd.DataFrame(response.json())
        if df.empty:
            logging.warning(f"Données Binance vides pour {symbol}")
            st.error(f"<div class='error-box'>Données Binance vides pour {symbol}.</div>", unsafe_allow_html=True)
            return None

        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(response.json(), columns=cols + ['ignore']*6)[cols]
        df['close'] = pd.to_numeric(df['close'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info(f"Données Binance récupérées avec succès pour {symbol}")
        return df[['date', 'close']]

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"Erreur HTTP Binance pour {symbol}: {http_err}")
        st.error(f"<div class='error-box'>Erreur HTTP Binance pour {symbol}: {http_err}</div>", unsafe_allow_html=True)
        return None
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Erreur de requête Binance pour {symbol}: {req_err}")
        st.error(f"<div class='error-box'>Erreur de requête Binance pour {symbol}: {req_err}</div>", unsafe_allow_html=True)
        return None
    except Exception as e:
        logging.critical(f"Erreur inattendue Binance pour {symbol}: {str(e)}")
        st.error(f"<div class='error-box'>Erreur inattendue Binance pour {symbol}: {str(e)}</div>", unsafe_allow_html=True)
        return None


def fetch_kraken_data(symbol):
    """Récupère les données depuis Kraken API (corrigé) avec gestion des erreurs"""
    logging.info(f"Début de la récupération des données Kraken pour {symbol}")
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        params = {
            'pair': f"{symbol}USD",
            'interval': 1440,
            'since': int((datetime.now() - timedelta(days=90)).timestamp())
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        if 'result' not in data or not data['result']:
            logging.warning(f"Réponse Kraken API malformée ou vide pour {symbol}")
            st.error(f"<div class='error-box'>Réponse Kraken API malformée ou vide pour {symbol}.</div>", unsafe_allow_html=True)
            return None

        ohlc_data = list(data['result'].values())[0]
        if not ohlc_data:
            logging.warning(f"Données Kraken vides pour {symbol}")
            st.error(f"<div class='error-box'>Données Kraken vides pour {symbol}.</div>", unsafe_allow_html=True)
            return None

        df = pd.DataFrame(ohlc_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close',
            'volume', 'wap', 'count'
        ])
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        df['close'] = pd.to_numeric(df['close'])
        logging.info(f"Données Kraken récupérées avec succès pour {symbol}")
        return df[['date', 'close']]

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"Erreur HTTP Kraken pour {symbol}: {http_err}")
        st.error(f"<div class='error-box'>Erreur HTTP Kraken pour {symbol}: {http_err}</div>", unsafe_allow_html=True)
        return None
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Erreur de requête Kraken pour {symbol}: {req_err}")
        st.error(f"<div class='error-box'>Erreur de requête Kraken pour {symbol}: {req_err}</div>", unsafe_allow_html=True)
        return None
    except Exception as e:
        logging.critical(f"Erreur inattendue Kraken pour {symbol}: {str(e)}")
        st.error(f"<div class='error-box'>Erreur inattendue Kraken pour {symbol}: {str(e)}</div>", unsafe_allow_html=True)
        return None


def get_crypto_data(symbol):
    """Agrège les données avec fallback et indication de la source"""
    sources = {
        'CoinGecko': lambda: fetch_coingecko_data(symbol.lower()),
        'Binance': lambda: fetch_binance_data(symbol),
        'Kraken': lambda: fetch_kraken_data(symbol)
    }

    for source_name, source_func in sources.items():
        df = source_func()
        if df is not None and not df.empty:
            st.markdown(f"<div class='source-badge'>Source: {source_name}</div>", unsafe_allow_html=True)
            return df

    logging.error("Aucune source de données disponible pour cette cryptomonnaie.")
    st.error("<div class='error-box'>Aucune source de données disponible pour cette cryptomonnaie.</div>", unsafe_allow_html=True)
    return None


# ======================
# ANALYSE TECHNIQUE AMÉLIORÉE (TOUTES DANS APP.PY)
# ======================
def calculate_indicators(df, indicators_config):
    """Calcule les indicateurs techniques sélectionnés"""
    logging.info("Début du calcul des indicateurs techniques")
    try:
        for indicator_name, indicator_params in indicators_config.items():
            if indicator_name == "RSI":
                df['rsi'] = INDICATORS["RSI"](df['close'], window=indicator_params.get('window', 14)).rsi()
            elif indicator_name == "MACD":
                macd = INDICATORS["MACD"](df['close'], window_fast=indicator_params.get('window_fast', 12), window_slow=indicator_params.get('window_slow', 26), window_sign=indicator_params.get('window_sign', 9))
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                df['macd_hist'] = macd.macd_diff()
            elif indicator_name == "Bollinger Bands":
                bb = INDICATORS["Bollinger Bands"](df['close'], window=indicator_params.get('window', 20), window_dev=indicator_params.get('window_dev', 2))
                df['bb_upper'] = bb.bollinger_hband()
                df['bb_lower'] = bb.bollinger_lband()
                df['bb_mid'] = bb.bollinger_mavg()
        for window in MA_WINDOWS:
            df[f'ma{window}'] = ta.trend.sma_indicator(df['close'], window=window)

        df['volatility'] = df['close'].pct_change().rolling(14).std() * 100
        logging.info("Calcul des indicateurs techniques terminé avec succès")
        return df.dropna()
    except Exception as e:
        logging.error(f"Erreur de calcul d'indicateurs: {str(e)}")
        st.error(f"<div class='error-box'>Erreur de calcul d'indicateurs: {str(e)}</div>", unsafe_allow_html=True)
        return None

# ======================
# CONSEILS DE TRADING (AGENT IA - SIMULATION) (TOUTES DANS APP.PY)
# ======================
def get_ia_trading_advice(df):
    """Simule des conseils de trading basés sur une IA (simplifié) avec prise en compte des positions du marché"""
    logging.info("Génération des conseils de trading IA")
    if df is None or df.empty or len(df) < 2:
        logging.warning("Données insuffisantes pour générer des conseils IA.")
        return "<div class='alert-box'>Données insuffisantes pour générer des conseils de trading IA.</div>"

    last_rsi = df['rsi'].iloc[-1]
    last_macd = df['macd'].iloc[-1]
    last_macd_signal = df['macd_signal'].iloc[-1]
    last_price = df['close'].iloc[-1]
    ma20 = df['ma20'].iloc[-1]
    ma50 = df['ma50'].iloc[-1]

    # Analyse de la position du marché (variation de prix récente)
    price_change = df['close'].iloc[-1] - df['close'].iloc[-2]

    advice_messages = []
    signal_type = "neutral-signal" # Signal neutre par défaut

    if price_change < 0:
        advice_messages.append("🔴 **Baisse récente du prix :** Surveiller opportunité d'achat si d'autres indicateurs confirment.")
        signal_type = "buy-signal" # Signal d'achat potentiel si baisse
    elif price_change > 0:
        advice_messages.append("🟢 **Hausse récente du prix :**  Potentiel signal de vente si surachat ou prudence si résistance approche.")
        signal_type = "sell-signal" # Signal de vente potentiel si hausse

    if last_rsi > 70:
        advice_messages.append("⚠️ **RSI élevé (surachat) :** Possible signal de vente ou de prudence.")
        if signal_type == "neutral-signal": signal_type = "sell-signal" # Priorité au signal de vente si RSI suracheté
    elif last_rsi < 30:
        advice_messages.append("✅ **RSI bas (survente) :** Possible signal d'achat.")
        if signal_type == "neutral-signal": signal_type = "buy-signal" # Priorité au signal d'achat si RSI survendu

    if last_macd > last_macd_signal and last_macd_hist > 0:
        advice_messages.append("📈 **MACD haussier :** Momentum positif.")
    elif last_macd < last_macd_signal and last_macd_hist < 0:
        advice_messages.append("📉 **MACD baissier :** Momentum négatif.")

    if last_price > ma20 and last_price > ma50:
        advice_messages.append("🐂 **Prix au-dessus des MAs 20 et 50 :** Tendance potentiellement haussière.")
    elif last_price < ma20 and last_price < ma50:
        advice_messages.append("🐻 **Prix en-dessous des MAs 20 et 50 :** Tendance potentiellement baissière.")

    if not advice_messages:
        advice = "Neutre ou signaux mixtes. Analyse plus approfondie nécessaire."
        signal_type = "neutral-signal"
    else:
        advice = "\n".join(advice_messages)
        logging.info(f"Conseils IA générés : Signal {signal_type}, Conseils : {advice}") # Log des conseils générés

    return f"<div class='advice-box {signal_type}'><h3>💡 Conseils de Trading IA (Simulé)</h3><p>{advice}</p><p><b>Dernier RSI (14j):</b> {last_rsi:.2f} | <b>Dernier MACD:</b> {last_macd:.2f} | <b>Dernier Signal MACD:</b> {last_macd_signal:.2f}</p><p><b>MA20:</b> {ma20:.2f} | <b>MA50:</b> {ma50:.2f} | <b>Prix actuel:</b> {last_price:.2f}</p></div>"


# ======================
# INTERFACE UTILISATEUR (TOUTES DANS APP.PY)
# ======================
def main():
    st.sidebar.header("⚙️ Configuration")
    selected_crypto = st.sidebar.selectbox(
        "Cryptomonnaie:",
        options=list(CRYPTO_LIST.keys()),
        format_func=lambda x: CRYPTO_LIST[x]
    )

    # Configuration des indicateurs dans la sidebar (plus flexible)
    st.sidebar.subheader("⚙️ Indicateurs Techniques")
    indicators_config = {}
    if st.sidebar.checkbox("RSI", value=True):
        indicators_config["RSI"] = {'window': st.sidebar.slider("Période RSI", 2, 30, 14)}
    if st.sidebar.checkbox("MACD"):
        indicators_config["MACD"] = {
            'window_fast': st.sidebar.slider("Période MACD Rapide", 2, 30, 12),
            'window_slow': st.sidebar.slider("Période MACD Lent", 10, 50, 26),
            'window_sign': st.sidebar.slider("Période Signal MACD", 2, 20, 9)
        }
    if st.sidebar.checkbox("Bollinger Bands"):
        indicators_config["Bollinger Bands"] = {
            'window': st.sidebar.slider("Période BB", 10, 50, 20),
            'window_dev': st.sidebar.slider("Écart-type BB", 1, 5, 2)
        }


    df = get_crypto_data(selected_crypto)
    if df is None:
        st.markdown("<div class='alert-box'>Impossible de récupérer les données. Veuillez réessayer plus tard ou choisir une autre cryptomonnaie.</div>", unsafe_allow_html=True)
        return

    df_indicators = calculate_indicators(df.copy(), indicators_config) # Calculer les indicateurs après la récupération des données
    if df_indicators is None:
        st.markdown("<div class='alert-box'>Erreur lors du calcul des indicateurs techniques.</div>", unsafe_allow_html=True)
        return

    current_price = df_indicators['close'].iloc[-1]
    st.header(f"{CRYPTO_LIST[selected_crypto]} - ${current_price:,.2f}")

    ia_advice_html = get_ia_trading_advice(df_indicators)
    st.markdown(ia_advice_html, unsafe_allow_html=True)


    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Indicateurs")
        if 'rsi' in df_indicators.columns:
            st.metric("RSI (14j)", f"{df_indicators['rsi'].iloc[-1]:.1f}")
        if 'volatility' in df_indicators.columns:
            st.metric("Volatilité (14j)", f"{df_indicators['volatility'].iloc[-1]:.1f}%")
        if 'macd' in df_indicators.columns:
             st.metric("MACD", f"{df_indicators['macd'].iloc[-1]:.2f}")

    with col2:
        st.subheader("📉 Moyennes Mobiles")
        for window in MA_WINDOWS:
            st.metric(f"MA{window}", f"{df_indicators[f'ma{window}'].iloc[-1]:,.2f}")
        if 'bb_upper' in df_indicators.columns:
            st.metric("BB Upper (20)", f"{df_indicators['bb_upper'].iloc[-1]:,.2f}")
        if 'bb_lower' in df_indicators.columns:
            st.metric("BB Lower (20)", f"{df_indicators['bb_lower'].iloc[-1]:,.2f}")


    st.subheader("📊 Historique des Prix avec Moyennes Mobiles")
    chart_data = df_indicators.set_index('date')[['close']]
    for window in MA_WINDOWS:
        if f'ma{window}' in df_indicators.columns:
            chart_data[f'MA{window}'] = df_indicators[f'ma{window}']

    st.line_chart(chart_data)

    if 'Bollinger Bands' in indicators_config and 'bb_upper' in df_indicators.columns:
        st.subheader("📊 Bollinger Bands")
        bb_chart_data = df_indicators.set_index('date')[['close', 'bb_upper', 'bb_lower', 'bb_mid']]
        st.line_chart(bb_chart_data)

    if 'MACD' in indicators_config and 'macd' in df_indicators.columns:
        st.subheader("📊 MACD")
        macd_chart_data = df_indicators.set_index('date')[['macd', 'macd_signal', 'macd_hist']]
        st.bar_chart(macd_chart_data['macd_hist'])
        st.line_chart(macd_chart_data[['macd', 'macd_signal']])


    st.markdown("<div class='alert-box'><b>Disclaimer important :</b> Les conseils de trading fournis par cet outil (IA simulée) sont uniquement à des fins informatives et éducatives. Ils ne constituent en aucun cas des conseils financiers personnalisés ou une incitation à l'investissement. Le trading de cryptomonnaies est risqué et vous pouvez perdre de l'argent.  N'investissez jamais de l'argent que vous ne pouvez pas vous permettre de perdre. Faites toujours vos propres recherches et consultez un conseiller financier professionnel avant de prendre toute décision d'investissement.  Utilisez cet outil à vos propres risques.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
