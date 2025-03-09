#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Application de Trading Cryptographique
Version : 2.2.0
Auteur : Aboubi Mustapha
"""

import os
import logging
from typing import Optional, List, Dict
from dotenv import load_dotenv
import streamlit as st
import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands

# Configuration initiale
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        "LINKUSDT": "Chainlink",
        "MATICUSDT": "Polygon",
        "SHIBUSDT": "Shiba Inu"
    }
    
    COINGECKO_MAP = {
        "BTCUSDT": "bitcoin",
        "ETHUSDT": "ethereum",
        "BNBUSDT": "binancecoin",
        "SOLUSDT": "solana",
        "XRPUSDT": "ripple",
        "ADAUSDT": "cardano",
        "DOGEUSDT": "dogecoin",
        "AVAXUSDT": "avalanche-2",
        "DOTUSDT": "polkadot",
        "LINKUSDT": "chainlink",
        "MATICUSDT": "matic-network",
        "SHIBUSDT": "shiba-inu"
    }

    API_URLS = {
        'binance': 'https://api.binance.com/api/v3/klines',
        'coingecko': 'https://api.coingecko.com/api/v3/coins/{}/market_chart'
    }

    PROXIES = {
        'http': os.getenv('PROXY_URL', ''),
        'https': os.getenv('PROXY_URL', '')
    }

# ======================
# FONCTIONS CORE
# ======================
@st.cache_data(ttl=3600)
def fetch_crypto_data(symbol: str) -> Optional[pd.DataFrame]:
    """Récupère les données depuis Binance ou CoinGecko avec fallback"""
    try:
        # Essai avec Binance
        response = requests.get(
            Config.API_URLS['binance'],
            params={'symbol': symbol, 'interval': '1d', 'limit': 300},
            proxies=Config.PROXIES,
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=10
        )
        response.raise_for_status()

        df = pd.DataFrame(
            response.json(),
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.iloc[:-1]

    except Exception as binance_error:
        logger.warning(f"Erreur Binance: {binance_error} - Tentative CoinGecko")
        
        try:
            # Fallback sur CoinGecko
            coin_id = Config.COINGECKO_MAP.get(symbol)
            if not coin_id:
                raise ValueError(f"Pas de mapping pour {symbol}")

            response = requests.get(
                Config.API_URLS['coingecko'].format(coin_id),
                params={'vs_currency': 'usd', 'days': 300},
                timeout=10
            )
            response.raise_for_status()

            prices = response.json()['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df[['date', 'close']]

        except Exception as cg_error:
            logger.error(f"Erreur CoinGecko: {cg_error}")
            st.error("Échec de récupération des données")
            return None

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les indicateurs techniques"""
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
        
        return df.dropna()
    
    except Exception as e:
        logger.error(f"Erreur calculs: {str(e)}")
        st.error("Erreur dans les calculs techniques")
        return df

def generate_trading_signals(df: pd.DataFrame) -> List[Dict]:
    """Génère les signaux de trading"""
    signals = []
    if df.empty:
        return signals

    latest = df.iloc[-1]

    # Signal Achat
    buy_cond = (
        (latest['pct_change_1d'] < -5) &
        (latest['close'] > latest['ma200']) &
        (latest['volatility'] > 15)
    )
    if buy_cond:
        signals.append({
            "type": "buy",
            "title": "🟥 ACHAT - Correction Brutale",
            "details": [
                f"RSI: {latest['rsi']:.1f}",
                f"Volatilité: {latest['volatility']:.1f}%",
                f"Distance MA200: {(latest['close']/latest['ma200']-1)*100:.1f}%"
            ]
        })

    # Signal Vente
    sell_cond = (
        (latest['rsi'] > 70) &
        (latest['close'] > latest['bb_upper']) &
        (df['ma50'].diff(3).mean() > 2)
    )
    if sell_cond:
        signals.append({
            "type": "sell",
            "title": "🟩 VENTE - Surachat",
            "details": [
                f"RSI: {latest['rsi']:.1f}",
                f"Bande Sup BB: {latest['bb_upper']:.2f}",
                f"Pente MA50: {df['ma50'].diff(3).mean():.1f}%"
            ]
        })

    return signals

# ======================
# INTERFACE UTILISATEUR
# ======================
def main():
    st.set_page_config(
        page_title="Crypto Analyst Pro",
        layout="wide",
        page_icon="📈"
    )
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    selected = st.sidebar.selectbox(
        "Cryptomonnaie :",
        options=list(Config.CRYPTO_LIST.keys()),
        format_func=lambda x: f"{Config.CRYPTO_LIST[x]} ({x})"
    )

    # Data processing
    df = fetch_crypto_data(selected)
    if df is None:
        return

    df = calculate_technical_indicators(df)
    
    # Main display
    st.title(f"{Config.CRYPTO_LIST[selected]} Analysis")
    st.metric("Prix Actuel", f"${df['close'].iloc[-1]:,.2f}")

    # Signals display
    signals = generate_trading_signals(df)
    if signals:
        st.subheader("🚨 Signaux de Trading")
        for signal in signals:
            color = "#FF4444" if signal["type"] == "buy" else "#00C853"
            with st.container():
                st.markdown(f"""
                <div style='
                    padding:1.5rem;
                    border-radius:10px;
                    margin:1rem 0;
                    border:2px solid {color};
                    background-color: {color}22;
                '>
                    <h3 style='color:{color}; margin-top:0;'>{signal["title"]}</h3>
                    <ul style='font-size:16px; color:#333;'>
                        {''.join([f"<li>{d}</li>" for d in signal["details"]])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📌 Aucun signal détecté - Position neutre recommandée")

    # Technical analysis
    with st.expander("📊 Analyse Technique Détaillée", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.line_chart(
                df.set_index('date')['close'].tail(60),
                use_container_width=True,
                color="#2196F3"
            )
        with col2:
            st.metric("RSI (14j)", f"{df['rsi'].iloc[-1]:.1f}")
            st.metric("Volatilité", f"{df['volatility'].iloc[-1]:.1f}%")
            st.metric("Distance MA200", f"{(df['close'].iloc[-1]/df['ma200'].iloc[-1]-1)*100:.1f}%")

if __name__ == "__main__":
    main()
