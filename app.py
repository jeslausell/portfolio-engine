import os
import time
import xml.etree.ElementTree as ET
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf
from google import genai

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Portfolio Intelligence Engine",
    page_icon="📈",
    layout="wide",
)

# --- SECRETS RETRIEVAL ---
IBKR_TOKEN = st.secrets.get("IBKR_TOKEN", os.getenv("IBKR_TOKEN", ""))
IBKR_QUERY_ID = st.secrets.get("IBKR_QUERY_ID", os.getenv("IBKR_QUERY_ID", ""))
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))


# --- DATA FETCHING: IBKR FLEX QUERY ---
@st.cache_data(ttl=3600)
def fetch_ibkr_portfolio(token: str, query_id: str):
    """Pulls open positions and cash from IBKR Flex Web Service."""
    if not token or not query_id:
        return None, 0.0

    send_url = f"https://ndcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.SendRequest?t={token}&q={query_id}&v=3"
    try:
        r = requests.get(send_url, timeout=15)
        root = ET.fromstring(r.content)

        if root.find("Status").text != "Success":
            return None, 0.0

        ref_code = root.find("ReferenceCode").text
        statement_url = f"https://ndcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.GetStatement?q={ref_code}&t={token}&v=3"

        # Poll report generation (up to 4 attempts)
        for _ in range(4):
            time.sleep(2)
            stmt_resp = requests.get(statement_url, timeout=15)
            if "Statement" in stmt_resp.text:
                break

        doc = ET.fromstring(stmt_resp.content)
        positions = []
        cash_balance = 0.0

        for pos in doc.iter("OpenPosition"):
            symbol = pos.attrib.get("symbol")
            shares = float(pos.attrib.get("position", 0))
            cost_basis = float(pos.attrib.get("costBasisMoney", 0))
            market_val = float(pos.attrib.get("positionValue", 0))

            if symbol and shares > 0:
                positions.append(
                    {
                        "Ticker": symbol,
                        "Shares": shares,
                        "CostBasis": cost_basis,
                        "MarketValue": market_val,
                    }
                )

        for cash in doc.iter("CashReport"):
            if cash.attrib.get("currency") == "BASE_SUMMARY":
                cash_balance = float(cash.attrib.get("endingCash", 0.0))

        df = pd.DataFrame(positions)
        return df, cash_balance
    except Exception:
        return None, 0.0


# --- DATA ENRICHMENT: YFINANCE ---
@st.cache_data(ttl=3600)
def enrich_tickers(tickers: list):
    """Fetches market fundamentals, PEG ratios, and business info."""
    details = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            details[ticker] = {
                "Price": info.get("currentPrice", info.get("regularMarketPrice", 0.0)),
                "Sector": info.get("sector", "Core ETF / Other"),
                "ForwardPE": info.get("forwardPE", None),
                "PEG": info.get("pegRatio", None),
                "Beta": info.get("beta", 1.0),
                "52wHigh": info.get("fiftyTwoWeekHigh", 0.0),
                "Summary": info.get("longBusinessSummary", "")[:300],
            }
        except Exception:
            details[ticker] = {
                "Price": 0.0,
                "Sector": "Unknown",
                "ForwardPE": None,
                "PEG": None,
                "Beta": 1.0,
                "52wHigh": 0.0,
                "Summary": "",
            }
    return details


# --- QUALITATIVE AI ANALYSIS: GEMINI ---
def analyze_thematic_exposure(holdings_summary: list, api_key: str):
    """Uses Gemini 2.5 Flash to tag thematic risk (e.g. AI exposure) and provide qualitative grades."""
    if not api_key:
        return (
            "API key missing. Add GEMINI_API_KEY in Streamlit Secrets.",
            {"General": 100},
        )

    client = genai.Client(api_key=api_key)
    prompt = f"""
    Analyze these portfolio holdings: {holdings_summary}.
    1. Identify hidden thematic concentration (e.g. total AI exposure, tech sensitivity).
    2. Give an executive qualitative grade (A, B, C) for current balance.
    3. Note any immediate thesis risks or trim candidates based on current market dynamics.
    4. Provide a single new stock or ETF discovery candidate that increases balance.

    Be clear, direct, and concise. No conversational fluff.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"AI Analysis temporarily unavailable: {e}"


# --- USER INTERFACE ---
st.title("🛡️ Portfolio Intelligence & Allocation Engine")

df_positions, cash_avail = fetch_ibkr_portfolio(IBKR_TOKEN, IBKR_QUERY_ID)

# Fallback baseline data if credentials are empty or initializing
if df_positions is None or df_positions.empty:
    st.info("Displaying demonstration data. Connect your IBKR credentials in secrets to stream your live portfolio.")
    df_positions = pd.DataFrame(
        [
            {"Ticker": "CSPX", "Shares": 25, "CostBasis": 12500, "MarketValue": 14200},
            {"Ticker": "VWRA", "Shares": 70, "CostBasis": 7700, "MarketValue": 8800},
            {"Ticker": "GOOGL", "Shares": 30, "CostBasis": 4800, "MarketValue": 5400},
            {"Ticker": "MSFT", "Shares": 12, "CostBasis": 4500, "MarketValue": 5100},
            {"Ticker": "BRK.B", "Shares": 10, "CostBasis": 3800, "MarketValue": 4500},
        ]
    )
    cash_avail = 5000.0

enriched_info = enrich_tickers(df_positions["Ticker"].tolist())
df_positions["Sector"] = df_positions["Ticker"].apply(lambda x: enriched_info.get(x, {}).get("Sector", "Other"))
df_positions["CurrentPrice"] = df_positions["Ticker"].apply(lambda x: enriched_info.get(x, {}).get("Price", 0.0))
df_positions["PEG"] = df_positions["Ticker"].apply(lambda x: enriched_info.get(x, {}).get("PEG", "N/A"))

total_equity = df_positions["MarketValue"].sum()
total_portfolio = total_equity + cash_avail

# Metric Overview Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Net Worth", f"${total_portfolio:,.2f}")
c2.metric("Invested Equity", f"${total_equity:,.2f}")
c3.metric("Cash Balance", f"${cash_avail:,.2f}")
c4.metric("Active Holdings", f"{len(df_positions)} Tickers")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Holdings & Concentration", "🎯 Capital Deployment ($)", "🧠 AI Thesis & Exit Signals"])

with tab1:
    col_chart, col_table = st.columns([1, 1])
    with col_chart:
        fig = px.pie(df_positions, values="MarketValue", names="Sector", title="Thematic Sector Allocation", hole=0.45)
        st.plotly_chart(fig, use_container_width=True)
    with col_table:
        st.subheader("Current Holdings")
        st.dataframe(
            df_positions[["Ticker", "Shares", "MarketValue", "Sector", "PEG"]].style.format(
                {"MarketValue": "${:,.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

with tab2:
    st.subheader("Smart Capital Allocator")
    st.write("Calculate exact buy orders to deploy cash without creating unbalanced positions.")

    deploy_amount = st.number_input("Capital to Deploy (USD):", min_value=100.0, max_value=500000.0, value=float(cash_avail if cash_avail > 0 else 5000.0), step=500.0)

    # Allocation heuristic: Core ETF emphasis + underweight quality picks
    target_weights = {"CSPX": 0.40, "VWRA": 0.30, "GOOGL": 0.10, "MSFT": 0.10, "BRK.B": 0.10}
    rebalance_data = []

    for ticker, target_w in target_weights.items():
        price = enriched_info.get(ticker, {}).get("Price", 100.0)
        current_val = df_positions.loc[df_positions["Ticker"] == ticker, "MarketValue"].sum() if ticker in df_positions["Ticker"].values else 0.0
        allocated_cash = deploy_amount * target_w
        shares_to_buy = int(allocated_cash // price) if price > 0 else 0
        actual_spend = shares_to_buy * price

        rebalance_data.append(
            {
                "Ticker": ticker,
                "Current Price": f"${price:,.2f}",
                "Target Weight": f"{target_w * 100:.0f}%",
                "Cash Allocated": f"${allocated_cash:,.2f}",
                "Shares to Buy": shares_to_buy,
                "Total Outlay": f"${actual_spend:,.2f}",
            }
        )

    st.table(pd.DataFrame(rebalance_data))

with tab3:
    st.subheader("Live Qualitative Screening & Exit Alerts")
    if st.button("Run Gemini Deep-Dive Audit"):
        with st.spinner("Analyzing holdings, checking concentration, and scanning news..."):
            summary_list = [f"{row['Ticker']} (${row['MarketValue']})" for _, row in df_positions.iterrows()]
            result_text = analyze_thematic_exposure(summary_list, GEMINI_API_KEY)
            st.markdown(result_text)
    else:
        st.info("Click the button above to run an on-demand macro scan and exit check powered by Gemini.")
