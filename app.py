"""
Analiza pietei SUA + BVB - aplicatie Streamlit cu 4 meniuri:
Screener | Analiza tehnica | Portofoliu | Backtesting

Rulare:
    pip install streamlit yfinance pandas numpy plotly
    streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="Analiza Piata SUA + BVB", layout="wide")

# ---------------------------------------------------------------- date
LISTE = {
    "SUA": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "KO"],
    "BVB": ["TLV.RO", "SNP.RO", "H2O.RO", "SNG.RO", "SNN.RO", "BRD.RO",
            "TGN.RO", "EL.RO", "M.RO", "DIGI.RO", "ONE.RO", "TEL.RO"],
}


@st.cache_data(ttl=900, show_spinner=False)
def incarca(ticker: str, perioada: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=perioada, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])


# ------------------------------------------------------------ indicatori
def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def macd(s: pd.Series, f=12, l=26, sg=9):
    linie = s.ewm(span=f, adjust=False).mean() - s.ewm(span=l, adjust=False).mean()
    semnal = linie.ewm(span=sg, adjust=False).mean()
    return linie, semnal, linie - semnal


def moneda(ticker: str) -> str:
    return "RON" if ticker.upper().endswith(".RO") else "USD"


# ---------------------------------------------------------------- meniu
st.sidebar.title("📈 Meniu")
pagina = st.sidebar.radio(
    "Alege secțiunea",
    ["🔎 Screener", "📊 Analiză tehnică", "💼 Portofoliu", "🧪 Backtesting"],
)
st.sidebar.caption("Date: Yahoo Finance (întârziate, neoficiale). "
                   "Informații analitice, nu sfaturi financiare.")


# ================================================================ 1. SCREENER
def pagina_screener():
    st.header("🔎 Screener de acțiuni")
    c1, c2 = st.columns([1, 2])
    piata = c1.selectbox("Piața", ["SUA", "BVB", "Ambele"])
    baza = LISTE["SUA"] + LISTE["BVB"] if piata == "Ambele" else LISTE[piata]
    tickere = c2.text_area("Tickere (separate prin virgulă)", ", ".join(baza), height=90)
    lista = [t.strip().upper() for t in tickere.split(",") if t.strip()]

    f1, f2, f3 = st.columns(3)
    rsi_min, rsi_max = f1.slider("Interval RSI", 0, 100, (0, 100))
    peste200 = f2.checkbox("Preț peste SMA200")
    var_min = f3.number_input("Variație minimă 1 lună (%)", value=-100.0)

    if not st.button("Rulează screener", type="primary"):
        return

    rows, bar = [], st.progress(0.0)
    for i, t in enumerate(lista):
        df = incarca(t)
        bar.progress((i + 1) / len(lista))
        if len(df) < 60:
            continue
        c = df["Close"]
        sma200 = c.rolling(200).mean().iloc[-1]
        rows.append({
            "Ticker": t, "Monedă": moneda(t),
            "Preț": round(c.iloc[-1], 2),
            "Var 1 lună %": round((c.iloc[-1] / c.iloc[-21] - 1) * 100, 2),
            "Var 6 luni %": round((c.iloc[-1] / c.iloc[-min(126, len(c) - 1)] - 1) * 100, 2),
            "RSI": round(rsi(c).iloc[-1], 1),
            "SMA50": round(c.rolling(50).mean().iloc[-1], 2),
            "SMA200": round(sma200, 2) if not np.isnan(sma200) else None,
            "Volum mediu 20z": int(df["Volume"].tail(20).mean()),
        })
    bar.empty()
    if not rows:
        st.warning("Nu am găsit date. Verifică tickerele.")
        return

    r = pd.DataFrame(rows)
    r = r[(r["RSI"].between(rsi_min, rsi_max)) & (r["Var 1 lună %"] >= var_min)]
    if peste200:
        r = r[r["Preț"] > r["SMA200"]]
    st.write(f"**{len(r)}** rezultate")
    st.dataframe(r.sort_values("RSI"), use_container_width=True, hide_index=True)
    st.download_button("Descarcă CSV", r.to_csv(index=False), "screener.csv")


# ============================================================ 2. ANALIZA TEHNICA
def pagina_tehnica():
    st.header("📊 Analiză tehnică și semnale")
    c1, c2 = st.columns(2)
    ticker = c1.text_input("Ticker", "AAPL").upper().strip()
    perioada = c2.selectbox("Perioadă", ["6mo", "1y", "2y", "5y"], index=1)
    df = incarca(ticker, "5y")
    if df.empty:
        st.error(f"Nu există date pentru {ticker}.")
        return
    luni = {"6mo": 126, "1y": 252, "2y": 504, "5y": 1260}[perioada]

    c = df["Close"]
    df["SMA50"], df["SMA200"] = c.rolling(50).mean(), c.rolling(200).mean()
    df["RSI"] = rsi(c)
    df["MACD"], df["SIG"], df["HIST"] = macd(c)
    d = df.tail(luni)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.2, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"],
                                 low=d["Low"], close=d["Close"], name="Preț"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA50"], name="SMA50"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA200"], name="SMA200"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI"), 2, 1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], name="MACD"), 3, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["SIG"], name="Semnal"), 3, 1)
    fig.add_trace(go.Bar(x=d.index, y=d["HIST"], name="Histogramă"), 3, 1)
    fig.update_layout(height=700, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # semnale
    u, p = df.iloc[-1], df.iloc[-2]
    st.subheader("Semnale curente")
    semnale = []
    if u["RSI"] < 30:
        semnale.append(("🟢", f"RSI {u['RSI']:.1f}: zonă de supravânzare"))
    elif u["RSI"] > 70:
        semnale.append(("🔴", f"RSI {u['RSI']:.1f}: zonă de supracumpărare"))
    else:
        semnale.append(("⚪", f"RSI {u['RSI']:.1f}: neutru"))
    if pd.notna(u["SMA200"]):
        if p["SMA50"] <= p["SMA200"] and u["SMA50"] > u["SMA200"]:
            semnale.append(("🟢", "Golden cross (SMA50 a trecut peste SMA200)"))
        elif p["SMA50"] >= p["SMA200"] and u["SMA50"] < u["SMA200"]:
            semnale.append(("🔴", "Death cross (SMA50 a trecut sub SMA200)"))
        elif u["SMA50"] > u["SMA200"]:
            semnale.append(("🟢", "Trend ascendent: SMA50 > SMA200"))
        else:
            semnale.append(("🔴", "Trend descendent: SMA50 < SMA200"))
    if p["MACD"] <= p["SIG"] and u["MACD"] > u["SIG"]:
        semnale.append(("🟢", "MACD a traversat în sus linia de semnal"))
    elif p["MACD"] >= p["SIG"] and u["MACD"] < u["SIG"]:
        semnale.append(("🔴", "MACD a traversat în jos linia de semnal"))
    else:
        semnale.append(("🟢" if u["MACD"] > u["SIG"] else "🔴",
                        "MACD peste semnal" if u["MACD"] > u["SIG"] else "MACD sub semnal"))
    for ico, txt in semnale:
        st.write(f"{ico} {txt}")

    st.subheader("Alertă")
    with st.expander("Trimite semnalele pe Telegram (opțional)"):
        st.caption("Completează token-ul botului și chat_id pentru a activa.")
        tok = st.text_input("Bot token", type="password")
        chat = st.text_input("Chat ID")
        if st.button("Trimite acum") and tok and chat:
            import requests
            msg = f"{ticker}\n" + "\n".join(f"{i} {t}" for i, t in semnale)
            r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                              data={"chat_id": chat, "text": msg}, timeout=10)
            st.success("Trimis") if r.ok else st.error(r.text)


# ================================================================ 3. PORTOFOLIU
def pagina_portofoliu():
    st.header("💼 Monitorizare portofoliu")
    if "pf" not in st.session_state:
        st.session_state.pf = pd.DataFrame({
            "Ticker": ["AAPL", "TLV.RO"],
            "Cantitate": [10.0, 500.0],
            "Preț achiziție": [150.0, 22.0],
        })

    up = st.file_uploader("Încarcă portofoliu CSV (Ticker, Cantitate, Preț achiziție)",
                          type="csv")
    if up is not None:
        st.session_state.pf = pd.read_csv(up)

    pf = st.data_editor(st.session_state.pf, num_rows="dynamic",
                        use_container_width=True, key="editor")
    st.session_state.pf = pf

    if not st.button("Actualizează prețuri", type="primary"):
        return
    pf = pf.dropna(subset=["Ticker"]).copy()
    pf["Ticker"] = pf["Ticker"].str.upper().str.strip()
    preturi, var_zi = [], []
    for t in pf["Ticker"]:
        df = incarca(t, "1mo")
        preturi.append(df["Close"].iloc[-1] if not df.empty else np.nan)
        var_zi.append((df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
                      if len(df) > 1 else np.nan)
    pf["Monedă"] = pf["Ticker"].map(moneda)
    pf["Preț curent"] = preturi
    pf["Var azi %"] = var_zi
    pf["Valoare"] = pf["Cantitate"] * pf["Preț curent"]
    pf["Cost"] = pf["Cantitate"] * pf["Preț achiziție"]
    pf["P/L"] = pf["Valoare"] - pf["Cost"]
    pf["P/L %"] = pf["P/L"] / pf["Cost"] * 100
    st.dataframe(pf.round(2), use_container_width=True, hide_index=True)

    st.subheader("Totaluri pe monedă")
    tot = pf.groupby("Monedă")[["Cost", "Valoare", "P/L"]].sum()
    tot["P/L %"] = tot["P/L"] / tot["Cost"] * 100
    st.dataframe(tot.round(2), use_container_width=True)
    st.caption("Monedele nu sunt convertite între ele.")

    fig = go.Figure(go.Pie(labels=pf["Ticker"], values=pf["Valoare"].fillna(0), hole=0.5))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Descarcă portofoliu", pf.to_csv(index=False), "portofoliu.csv")


# ================================================================ 4. BACKTESTING
def pozitie_rsi(r: pd.Series, jos: float, sus: float) -> pd.Series:
    stare, out = 0, []
    for v in r:
        if v < jos:
            stare = 1
        elif v > sus:
            stare = 0
        out.append(stare)
    return pd.Series(out, index=r.index)


def pagina_backtest():
    st.header("🧪 Backtesting strategii")
    c1, c2, c3 = st.columns(3)
    ticker = c1.text_input("Ticker", "AAPL").upper().strip()
    ani = c2.selectbox("Istoric", [2, 5, 10], index=1)
    comision = c3.number_input("Comision per tranzacție (%)", 0.0, 2.0, 0.1, 0.05)
    strategie = st.selectbox("Strategie", ["Încrucișare SMA", "RSI revenire la medie"])

    if strategie == "Încrucișare SMA":
        a, b = st.columns(2)
        rapid = a.number_input("SMA rapid", 5, 100, 50)
        lent = b.number_input("SMA lent", 20, 400, 200)
    else:
        a, b = st.columns(2)
        jos = a.number_input("Cumpără când RSI <", 5, 50, 30)
        sus = b.number_input("Vinde când RSI >", 50, 95, 70)

    if not st.button("Rulează backtest", type="primary"):
        return
    df = incarca(ticker, f"{ani}y")
    if len(df) < 250:
        st.error("Date insuficiente.")
        return
    c = df["Close"]
    if strategie == "Încrucișare SMA":
        poz = (c.rolling(int(rapid)).mean() > c.rolling(int(lent)).mean()).astype(int)
    else:
        poz = pozitie_rsi(rsi(c), jos, sus)

    poz = poz.shift(1).fillna(0)  # fara look-ahead: actionam a doua zi
    ret = c.pct_change().fillna(0)
    tranz = poz.diff().abs().fillna(0)
    strat = poz * ret - tranz * comision / 100
    eq_s, eq_b = (1 + strat).cumprod(), (1 + ret).cumprod()

    def metrici(eq, r):
        ani_ = len(r) / 252
        dd = (eq / eq.cummax() - 1).min()
        sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
        return {"Randament total %": (eq.iloc[-1] - 1) * 100,
                "Randament anual %": (eq.iloc[-1] ** (1 / ani_) - 1) * 100,
                "Drawdown maxim %": dd * 100, "Sharpe": sharpe}

    m = pd.DataFrame({"Strategie": metrici(eq_s, strat),
                      "Buy & Hold": metrici(eq_b, ret)}).round(2)
    st.dataframe(m, use_container_width=True)
    st.write(f"Număr tranzacții: **{int(tranz.sum())}**, "
             f"timp în piață: **{poz.mean() * 100:.0f}%**")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq_s.index, y=eq_s, name="Strategie"))
    fig.add_trace(go.Scatter(x=eq_b.index, y=eq_b, name="Buy & Hold"))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Valoare (start = 1)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Rezultatele trecute nu garantează performanța viitoare.")


# ---------------------------------------------------------------- router
{
    "🔎 Screener": pagina_screener,
    "📊 Analiză tehnică": pagina_tehnica,
    "💼 Portofoliu": pagina_portofoliu,
    "🧪 Backtesting": pagina_backtest,
}[pagina]()
