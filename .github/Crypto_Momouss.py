#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Application de Trading Cryptographique
Version : 1.2.0
Auteur : Aboubi Mustapha
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import logging
from typing import Optional, Dict, List
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

# Configuration initiale
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ======================
# CONFIGURATION
# ======================
class Config:
    """Configuration de l'application"""
    BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
    CRYPTO_LIST = {
        "BTCUSDT": "Bitcoin",
        "ETHUSDT": "Ethereum",
        # ... (liste complète)
    }
    INTERVAL = "1d"
    CACHE_TTL = 3600  # 1 heure

# ======================
# FONCTIONS CORE
# ======================
@st.cache_data(ttl=Config.CACHE_TTL)
def fetch_crypto_data(symbol: str) -> Optional[pd.DataFrame]:
    """Récupère les données historiques depuis l'API Binance"""
    try:
        params = {
            'symbol': symbol,
            'interval': Config.INTERVAL,
            'limit': 300
        }
        
        response = requests.get(
            Config.BINANCE_API_URL,
            params=params,
            timeout=10
        )
        response.raise_for_status()

        df = pd.DataFrame(
            response.json(),
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df[['date', 'open', 'high', 'low', 'close', 'volume']].iloc[:-1]

    except Exception as e:
        logger.error(f"Erreur API Binance : {str(e)}")
        st.error("Erreur de connexion à l'API Binance")
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
        logger.error(f"Erreur calcul indicateurs : {str(e)}")
        st.error("Erreur dans le calcul des indicateurs")
        return df

def generate_trading_signals(df: pd.DataFrame) -> List[Dict]:
    """Génère les signaux de trading"""
    signals = []
    if df.empty:
        return signals

    latest = df.iloc[-1]

    # Signal Achat
    if (latest['pct_change_1d'] < -5 or df['pct_change_1d'].tail(3).sum() < -10) 
        and (latest['close'] > latest['ma200']) 
        and (latest['volatility'] > 15):
        
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
    if (latest['rsi'] > 70) 
        and (latest['close'] > latest['bb_upper']) 
        and (df['ma50'].diff(3).mean() > 2):
        
        signals.append({
            "type": "sell",
            "title": "🟩 VENTE - Surachat",
            "details": [
                f"RSI: {latest['rsi']:.1f}",
                f"Bande Supérieure BB: {latest['bb_upper']:.2f}",
                f"Pente MA50: {df['ma50'].diff(3).mean():.1f}%"
            ]
        })

    return signals

# ======================
# INTERFACE UTILISATEUR
# ======================
def main():
    """Configuration principale de l'interface Streamlit"""
    st.set_page_config(
        page_title="Crypto Trading Analyst",
        layout="wide",
        page_icon="📊"
    )
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    selected_symbol = st.sidebar.selectbox(
        "Cryptomonnaie :",
        options=list(Config.CRYPTO_LIST.keys()),
        format_func=lambda x: f"{Config.CRYPTO_LIST[x]} ({x})"
    )

    # Récupération données
    df = fetch_crypto_data(selected_symbol)
    if df is None:
        return

    df = calculate_technical_indicators(df)

    # Affichage principal
    st.title(f"{Config.CRYPTO_LIST[selected_symbol]} ({selected_symbol})")
    st.metric("Prix Actuel", f"${df['close'].iloc[-1]:,.2f}")

    # Signaux
    signals = generate_trading_signals(df)
    if signals:
        st.subheader("🚨 Signaux de Trading")
        for signal in signals:
            color = "#FF4444" if signal["type"] == "buy" else "#00C853"
            st.markdown(f"""
            <div style='padding:1rem; border-radius:8px; border:2px solid {color}; margin:1rem 0;'>
                <h3 style='color:{color}; margin-top:0;'>{signal["title"]}</h3>
                <ul style='font-size:16px;'>
                    {''.join([f"<li>{detail}</li>" for detail in signal["details"]])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📌 Aucun signal détecté - Position neutre recommandée")

    # Graphiques
    with st.expander("📈 Analyse Technique", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(df.set_index('date')['close'], use_container_width=True)
        with col2:
            st.area_chart(df.set_index('date')[['ma50', 'ma200']])

if __name__ == "__main__":
    main()
