import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

# ML imports check
try:
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# Session state initialization
if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Nifty500 Weekly Buy/Sell Predictor", layout="wide")
st.title("📊 Nifty500 Buy/Sell Predictor (Weekly) — Rules + Elliott Wave + GB ML")

# ---------------- TICKERS ----------------
NIFTY500_TICKERS = [
    "360ONE.NS","3MINDIA.NS","ABB.NS","TIPSMUSIC.NS","ACC.NS","ACMESOLAR.NS","AIAENG.NS","APLAPOLLO.NS","AUBANK.NS","AWL.NS","AADHARHFC.NS",
    "AARTIIND.NS","AAVAS.NS","ABBOTINDIA.NS","ACE.NS","ADANIENSOL.NS","ADANIENT.NS","ADANIGREEN.NS","ADANIPORTS.NS","ADANIPOWER.NS","ATGL.NS",
    "ABCAPITAL.NS","ABFRL.NS","ABREL.NS","ABSLAMC.NS","AEGISLOG.NS","AFCONS.NS","AFFLE.NS","AJANTPHARM.NS","AKUMS.NS","APLLTD.NS",
    "ALIVUS.NS","ALKEM.NS","ALKYLAMINE.NS","ALOKINDS.NS","ARE&M.NS","AMBER.NS","AMBUJACEM.NS","ANANDRATHI.NS","ANANTRAJ.NS","ANGELONE.NS",
    "APARINDS.NS","APOLLOHOSP.NS","APOLLOTYRE.NS","APTUS.NS","ASAHIINDIA.NS","ASHOKLEY.NS","ASIANPAINT.NS","ASTERDM.NS","ASTRAZEN.NS","ASTRAL.NS",
    "ATUL.NS","AUROPHARMA.NS","AIIL.NS","DMART.NS","AXISBANK.NS","BASF.NS","BEML.NS","BLS.NS","BSE.NS","BAJAJ-AUTO.NS",
    "BAJFINANCE.NS","BAJAJFINSV.NS","BAJAJHLDNG.NS","BAJAJHFL.NS","BALKRISIND.NS","BALRAMCHIN.NS","BANDHANBNK.NS","BANKBARODA.NS","BANKINDIA.NS","MAHABANK.NS",
    "BATAINDIA.NS","BAYERCROP.NS","BERGEPAINT.NS","BDL.NS","BEL.NS","BHARATFORG.NS","BHEL.NS","BPCL.NS","BHARTIARTL.NS","BHARTIHEXA.NS",
    "BIKAJI.NS","BIOCON.NS","BSOFT.NS","BLUEDART.NS","BLUESTARCO.NS","BBTC.NS","BOSCHLTD.NS","FIRSTCRY.NS","BRIGADE.NS","BRITANNIA.NS",
    "MAPMYINDIA.NS","CCL.NS","CESC.NS","CGPOWER.NS","CRISIL.NS","CAMPUS.NS","CANFINHOME.NS","CANBK.NS","CAPLIPOINT.NS","CGCL.NS",
    "CARBORUNIV.NS","CASTROLIND.NS","CEATLTD.NS","CENTRALBK.NS","CDSL.NS","CENTURYPLY.NS","CERA.NS","CHALET.NS","CHAMBLFERT.NS","CHENNPETRO.NS",
    "CHOLAHLDNG.NS","CHOLAFIN.NS","CIPLA.NS","CUB.NS","CLEAN.NS","COALINDIA.NS","COCHINSHIP.NS","COFORGE.NS","COHANCE.NS","COLPAL.NS",
    "CAMS.NS","CONCORDBIO.NS","CONCOR.NS","COROMANDEL.NS","CRAFTSMAN.NS","CREDITACC.NS","CROMPTON.NS","CUMMINSIND.NS","CYIENT.NS","DCMSHRIRAM.NS",
    "DLF.NS","DOMS.NS","DABUR.NS","DALBHARAT.NS","DATAPATTNS.NS","DEEPAKFERT.NS","DEEPAKNTR.NS","DELHIVERY.NS","DEVYANI.NS","DIVISLAB.NS",
    "DIXON.NS","LALPATHLAB.NS","DRREDDY.NS","DUMMYDBRLT.NS","EIDPARRY.NS","EIHOTEL.NS","EICHERMOT.NS","ELECON.NS","ELGIEQUIP.NS","EMAMILTD.NS",
    "EMCURE.NS","ENDURANCE.NS","ENGINERSIN.NS","ERIS.NS","ESCORTS.NS","ETERNAL.NS","EXIDEIND.NS","NYKAA.NS","FEDERALBNK.NS","FACT.NS",
    "FINCABLES.NS","FINPIPE.NS","FSL.NS","FIVESTAR.NS","FORTIS.NS","GAIL.NS","GVT&D.NS","GMRAIRPORT.NS","GRSE.NS","GICRE.NS",
    "GILLETTE.NS","GLAND.NS","GLAXO.NS","GLENMARK.NS","MEDANTA.NS","GODIGIT.NS","GPIL.NS","GODFRYPHLP.NS","GODREJAGRO.NS","GODREJCP.NS",
    "GODREJIND.NS","GODREJPROP.NS","GRANULES.NS","GRAPHITE.NS","GRASIM.NS","GRAVITA.NS","GESHIP.NS","FLUOROCHEM.NS","GUJGASLTD.NS","GMDCLTD.NS",
    "GNFC.NS","GPPL.NS","GSPL.NS","HEG.NS","HBLENGINE.NS","HCLTECH.NS","HDFCAMC.NS","HDFCBANK.NS","HDFCLIFE.NS","HFCL.NS",
    "HAPPSTMNDS.NS","HAVELLS.NS","HEROMOTOCO.NS","HSCL.NS","HINDALCO.NS","HAL.NS","HINDCOPPER.NS","HINDPETRO.NS","HINDUNILVR.NS","HINDZINC.NS",
    "POWERINDIA.NS","HOMEFIRST.NS","HONASA.NS","HONAUT.NS","HUDCO.NS","HYUNDAI.NS","ICICIBANK.NS","ICICIGI.NS","ICICIPRULI.NS","IDBI.NS",
    "IDFCFIRSTB.NS","IFCI.NS","IIFL.NS","INOXINDIA.NS","IRB.NS","IRCON.NS","ITC.NS","ITI.NS","INDGN.NS","INDIACEM.NS",
    "INDIAMART.NS","INDIANB.NS","IEX.NS","INDHOTEL.NS","IOC.NS","IOB.NS","IRCTC.NS","IRFC.NS","IREDA.NS","IGL.NS",
    "INDUSTOWER.NS","INDUSINDBK.NS","NAUKRI.NS","INFY.NS","INOXWIND.NS","INTELLECT.NS","INDIGO.NS","IGIL.NS","IKS.NS","IPCALAB.NS",
    "JBCHEPHARM.NS","JKCEMENT.NS","JBMA.NS","JKTYRE.NS","JMFINANCIL.NS","JSWENERGY.NS","JSWHL.NS","JSWINFRA.NS","JSWSTEEL.NS","JPPOWER.NS",
    "J&KBANK.NS","JINDALSAW.NS","JSL.NS","JINDALSTEL.NS","JIOFIN.NS","JUBLFOOD.NS","JUBLINGREA.NS","JUBLPHARMA.NS","JWL.NS","JUSTDIAL.NS",
    "JYOTHYLAB.NS","JYOTICNC.NS","KPRMILL.NS","KEI.NS","KNRCON.NS","KPITTECH.NS","KAJARIACER.NS","KPIL.NS","KALYANKJIL.NS","KANSAINER.NS",
    "KARURVYSYA.NS","KAYNES.NS","KEC.NS","KFINTECH.NS","KIRLOSBROS.NS","KIRLOSENG.NS","KOTAKBANK.NS","KIMS.NS","LTF.NS","LTTS.NS",
    "LICHSGFIN.NS","LTFOODS.NS","LTIM.NS","LT.NS","LATENTVIEW.NS","LAURUSLABS.NS","LEMONTREE.NS","LICI.NS","LINDEINDIA.NS","LLOYDSME.NS",
    "LODHA.NS","LUPIN.NS","MMTC.NS","MRF.NS","MGL.NS","MAHSEAMLES.NS","M&MFIN.NS","M&M.NS","MANAPPURAM.NS","MRPL.NS",
    "MANKIND.NS","MARICO.NS","MARUTI.NS","MASTEK.NS","MFSL.NS","MAXHEALTH.NS","MAZDOCK.NS","METROPOLIS.NS","MINDACORP.NS","MSUMI.NS",
    "MOTILALOFS.NS","MPHASIS.NS","MCX.NS","MUTHOOTFIN.NS","NATCOPHARM.NS","NBCC.NS","NCC.NS","NHPC.NS","NLCINDIA.NS","NMDC.NS",
    "NSLNISP.NS","NTPCGREEN.NS","NTPC.NS","NH.NS","NATIONALUM.NS","NAVA.NS","NAVINFLUOR.NS","NESTLEIND.NS","NETWEB.NS","NETWORK18.NS",
    "NEULANDLAB.NS","NEWGEN.NS","NAM-INDIA.NS","NIVABUPA.NS","NUVAMA.NS","OBEROIRLTY.NS","ONGC.NS","OIL.NS","OLAELEC.NS","OLECTRA.NS",
    "PAYTM.NS","POLICYBZR.NS","PCBL.NS","PGEL.NS","PIIND.NS","PNBHOUSING.NS","PNCINFRA.NS","PTCIL.NS","PVRINOX.NS",
    "PAGEIND.NS","PATANJALI.NS","PERSISTENT.NS","PETRONET.NS","PFIZER.NS","PHOENIXLTD.NS","PIDILITIND.NS","PEL.NS","PPLPHARMA.NS","POLYMED.NS",
    "POLYCAB.NS","POONAWALLA.NS","PFC.NS","POWERGRID.NS","PRAJIND.NS","PREMIERENE.NS","PRESTIGE.NS","PNB.NS","RRKABEL.NS","RBLBANK.NS",
    "RECLTD.NS","RHIM.NS","RITES.NS","RADICO.NS","RVNL.NS","RAILTEL.NS","RAINBOW.NS","RKFORGE.NS","RCF.NS","RTNINDIA.NS",
    "RAYMONDLSL.NS","RAYMOND.NS","REDINGTON.NS","RELIANCE.NS","RPOWER.NS","ROUTE.NS","SBFC.NS","SBICARD.NS","SBILIFE.NS","SJVN.NS",
    "SKFINDIA.NS","SRF.NS","SAGILITY.NS","SAILIFE.NS","SAMMAANCAP.NS","MOTHERSON.NS","SAPPHIRE.NS","SARDAEN.NS","SAREGAMA.NS","SCHAEFFLER.NS",
    "SCHNEIDER.NS","SCI.NS","SHREECEM.NS","RENUKA.NS","SHRIRAMFIN.NS","SHYAMMETL.NS","SIEMENS.NS","SIGNATURE.NS","SOBHA.NS","SOLARINDS.NS",
    "SONACOMS.NS","SONATSOFTW.NS","STARHEALTH.NS","SBIN.NS","SAIL.NS","SWSOLAR.NS","SUMICHEM.NS","SUNPHARMA.NS","SUNTV.NS","SUNDARMFIN.NS",
    "SUNDRMFAST.NS","SUPREMEIND.NS","SUZLON.NS","SWANENERGY.NS","SWIGGY.NS","SYNGENE.NS","SYRMA.NS","TBOTEK.NS","TVSMOTOR.NS","TANLA.NS",
    "TATACHEM.NS","TATACOMM.NS","TCS.NS","TATACONSUM.NS","TATAELXSI.NS","TATAINVEST.NS","TATAMOTORS.NS","TATAPOWER.NS","TATASTEEL.NS","TATATECH.NS",
    "TTML.NS","TECHM.NS","TECHNOE.NS","TEJASNET.NS","NIACL.NS","RAMCOCEM.NS","THERMAX.NS","TIMKEN.NS","TITAGARH.NS","TITAN.NS",
    "TORNTPHARM.NS","TORNTPOWER.NS","TARIL.NS","TRENT.NS","TRIDENT.NS","TRIVENI.NS","TRITURBINE.NS","TIINDIA.NS","UCOBANK.NS","UNOMINDA.NS",
    "UPL.NS","UTIAMC.NS","ULTRACEMCO.NS","UNIONBANK.NS","UBL.NS","UNITDSPR.NS","USHAMART.NS","VGUARD.NS","DBREALTY.NS","VTL.NS",
    "VBL.NS","MANYAVAR.NS","VEDL.NS","VIJAYA.NS","VMM.NS","IDEA.NS","VOLTAS.NS","WAAREEENER.NS","WELCORP.NS","WELSPUNLIV.NS",
    "WESTLIFE.NS","WHIRLPOOL.NS","WIPRO.NS","WOCKPHARMA.NS","YESBANK.NS","ZFCVINDIA.NS","ZEEL.NS","ZENTEC.NS","ZENSARTECH.NS","ZYDUSLIFE.NS",
    "ECLERX.NS",
]

# ---------------- UTIL ----------------
def add_tradingview_links(df):
    df = df.copy()
    if "Ticker" in df.columns:
        df["TradingView"] = df["Ticker"].apply(
            lambda t: f'<a href="https://www.tradingview.com/chart/?symbol=NSE:{t.replace(".NS","")}" target="_blank">📈 Chart</a>'
        )
    return df

class _TQDM:
    def __init__(self, total, desc=""):
        self.pb = st.progress(0, text=desc)
        self.total = max(total, 1)
        self.i = 0
    def update(self):
        self.i += 1
        self.pb.progress(min(self.i / self.total, 1.0), text=f"{self.i}/{self.total}")
    def close(self):
        self.pb.empty()

def stqdm(iterable, total=None, desc=""):
    if total is None:
        try: total = len(iterable)
        except: total = 100
    bar = _TQDM(total=total, desc=desc)
    for x in iterable:
        yield x
        bar.update()
    bar.close()

# ---------------- DATA DOWNLOAD ----------------
@st.cache_data(show_spinner=False)
def download_data_multi(tickers, period="5y", interval="1wk"):
    if isinstance(tickers, str): tickers = [tickers]
    frames = []
    batch_size = 50
    for i in stqdm(range(0, len(tickers), batch_size), desc="Downloading weekly data", total=len(tickers)//batch_size + 1):
        batch = tickers[i : i + batch_size]
        try:
            df = yf.download(batch, period=period, interval=interval, group_by="ticker", progress=False, threads=True)
            if df is not None and not df.empty: frames.append(df)
        except: pass
    if not frames: return None
    out = pd.concat(frames, axis=1)
    if isinstance(out.columns, pd.MultiIndex):
        idx = pd.MultiIndex.from_tuples(list(dict.fromkeys(out.columns.tolist())))
        out = out.loc[:, idx]
    return out

@st.cache_data(show_spinner=False)
def load_history_for_ticker(ticker, period="5y", interval="1wk"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=True)
        return df
    except: return pd.DataFrame()

# ---------------- ELLIOTT WAVE (ZigZag + Heuristics) ----------------
def zigzag_pivots(close: pd.Series, pct=0.05, min_bars=5):
    if close.isna().all() or len(close) < max(50, min_bars * 4):
        return pd.DataFrame(columns=["idx", "price", "type"])
    c = close.values.astype(float)
    idxs = close.index
    piv = []
    last_piv_i, last_extreme_i, trend = 0, 0, None
    last_extreme_p = c[0]
    for i in range(1, len(c)):
        if trend in (None, "up"):
            if c[i] > last_extreme_p: last_extreme_p, last_extreme_i = c[i], i
        if trend in (None, "down"):
            if c[i] < last_extreme_p: last_extreme_p, last_extreme_i = c[i], i
        if trend in (None, "up"):
            dd = (c[i] - last_extreme_p) / last_extreme_p if last_extreme_p != 0 else 0
            if dd <= -pct and (i - last_piv_i) >= min_bars:
                piv.append((idxs[last_extreme_i], float(last_extreme_p), "H"))
                last_piv_i, trend = last_extreme_i, "down"
                last_extreme_p, last_extreme_i = c[i], i
        if trend in (None, "down"):
            uu = (c[i] - last_extreme_p) / last_extreme_p if last_extreme_p != 0 else 0
            if uu >= pct and (i - last_piv_i) >= min_bars:
                piv.append((idxs[last_extreme_i], float(last_extreme_p), "L"))
                last_piv_i, trend = last_extreme_i, "up"
                last_extreme_p, last_extreme_i = c[i], i
    if len(piv) >= 2:
        cleaned = [piv[0]]
        for i in range(1, len(piv)):
            if piv[i][2] == cleaned[-1][2]:
                if (piv[i][2] == "H" and piv[i][1] > cleaned[-1][1]) or (piv[i][2] == "L" and piv[i][1] < cleaned[-1][1]):
                    cleaned[-1] = piv[i]
            else: cleaned.append(piv[i])
        piv = cleaned
    if not piv: return pd.DataFrame(columns=["idx", "price", "type"])
    idx, price, typ = zip(*piv)
    return pd.DataFrame({"idx": list(idx), "price": list(price), "type": list(typ)})

def fib_okay(a, b, ratio, tol=0.18):
    if b == 0 or np.isnan(a) or np.isnan(b): return False
    return abs((a / b) - ratio) <= tol * ratio

def elliott_phase_from_pivots(pivots: pd.DataFrame):
    out = {"phase": "Unknown", "wave_no": 0, "bullish": False, "bearish": False}
    if pivots.empty: return out
    if len(pivots) >= 5:
        p5 = pivots.iloc[-5:].reset_index(drop=True)
        if all(p5.loc[i, "type"] != p5.loc[i-1, "type"] for i in range(1, 5)):
            prices, types = p5["price"].values, p5["type"].values
            if types.tolist() == ["L", "H", "L", "H", "L"]:
                if prices[3] > prices[1] and prices[4] > prices[2]:
                    out.update({"phase": "ImpulseUp", "wave_no": 5, "bullish": True})
                    return out
            if types.tolist() == ["H", "L", "H", "L", "H"]:
                if prices[3] < prices[1] and prices[4] < prices[2]:
                    out.update({"phase": "ImpulseDown", "wave_no": 5, "bearish": True})
                    return out
    if len(pivots) >= 3:
        p3 = pivots.iloc[-3:].reset_index(drop=True)
        if all(p3.loc[i, "type"] != p3.loc[i-1, "type"] for i in range(1, 3)):
            t = p3["type"].tolist()
            if t == ["L", "H", "L"]: out.update({"phase": "CorrectionUp", "wave_no": 3, "bullish": True})
            elif t == ["H", "L", "H"]: out.update({"phase": "CorrectionDown", "wave_no": 3, "bearish": True})
    return out

def add_elliott_features_core(df_close: pd.Series, pct=0.05, min_bars=5):
    piv = zigzag_pivots(df_close, pct=pct, min_bars=min_bars)
    phase = elliott_phase_from_pivots(piv)
    return phase, piv

# ---------------- FEATURE ENGINEERING ----------------
def compute_features(df, sma_windows=(20, 50, 200), support_window=30, zz_pct=0.05, zz_min_bars=5):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    if "Close" not in df.columns or df["Close"].dropna().empty: return pd.DataFrame()
    df = df.copy()
    try: df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    except: df["RSI"] = np.nan
    for win in sma_windows: df[f"SMA{win}"] = df["Close"].rolling(window=win, min_periods=1).mean()
    df["Support"] = df["Close"].rolling(window=support_window, min_periods=1).min()
    df["RSI_Direction"] = df["RSI"].diff(5)
    df["Price_Direction"] = df["Close"].diff(5)
    df["Bullish_Div"] = (df["RSI_Direction"] > 0) & (df["Price_Direction"] < 0)
    df["Bearish_Div"] = (df["RSI_Direction"] < 0) & (df["Price_Direction"] > 0)
    for w in (1, 3, 5, 10): df[f"Ret_{w}"] = df["Close"].pct_change(w)
    for win in sma_windows: df[f"Dist_SMA{win}"] = (df["Close"] - df[f"SMA{win}"]) / df[f"SMA{win}"]
    for col in ["RSI"] + [f"SMA{w}" for w in sma_windows]: df[f"{col}_slope"] = df[col].diff()
    try:
        phase, piv = add_elliott_features_core(df["Close"], pct=zz_pct, min_bars=zz_min_bars)
        p_map = {"ImpulseUp":1, "ImpulseDown":-1, "CorrectionUp":2, "CorrectionDown":-2, "Unknown":0}
        df["Elliott_Phase_Code"] = p_map.get(phase["phase"], 0)
        df["Elliott_Wave_No"] = int(phase.get("wave_no", 0))
        df["Elliott_Bullish_Int"] = int(phase.get("bullish", False))
        df["Elliott_Bearish_Int"] = int(phase.get("bearish", False))
    except:
        df["Elliott_Phase_Code"] = 0
        df["Elliott_Wave_No"] = 0
        df["Elliott_Bullish_Int"] = 0
        df["Elliott_Bearish_Int"] = 0
    return df

def get_latest_features_for_ticker(ticker_df, ticker, sma_windows, support_window, zz_pct, zz_min_bars):
    df = compute_features(ticker_df, sma_windows, support_window, zz_pct, zz_min_bars).dropna()
    if df.empty: return None
    latest = df.iloc[-1]
    return {
        "Ticker": ticker, "Close": float(latest["Close"]), "RSI": float(latest["RSI"]), "Support": float(latest["Support"]),
        **{f"SMA{w}": float(latest.get(f"SMA{w}", np.nan)) for w in sma_windows},
        "Bullish_Div": bool(latest["Bullish_Div"]), "Bearish_Div": bool(latest["Bearish_Div"]),
        "Elliott_Phase_Code": int(latest.get("Elliott_Phase_Code", 0)),
        "Elliott_Wave_No": int(latest.get("Elliott_Wave_No", 0)),
        "Elliott_Bullish_Int": int(latest.get("Elliott_Bullish_Int", 0)),
        "Elliott_Bearish_Int": int(latest.get("Elliott_Bearish_Int", 0)),
    }

def get_features_for_all(tickers, sma_windows, support_window, zz_pct, zz_min_bars):
    multi_df = download_data_multi(tickers)
    if multi_df is None or multi_df.empty: return pd.DataFrame()
    features_list = []
    available = multi_df.columns.get_level_values(0).unique() if isinstance(multi_df.columns, pd.MultiIndex) else [tickers[0]]
    for ticker in tickers:
        tdf = multi_df[ticker].dropna() if isinstance(multi_df.columns, pd.MultiIndex) else multi_df.dropna()
        if tdf.empty: continue
        feats = get_latest_features_for_ticker(tdf, ticker, sma_windows, support_window, zz_pct, zz_min_bars)
        if feats: features_list.append(feats)
    return pd.DataFrame(features_list)

# ---------------- STRATEGY & LABELING ----------------
def predict_buy_sell_rule(df, rsi_buy=30, rsi_sell=70):
    if df.empty: return df
    results = df.copy()
    reversal_buy = (results["RSI"] < rsi_buy) & (results.get("Bullish_Div", True)) & (results["Close"] > results["SMA20"])
    trend_buy = (results["Close"] > results["SMA20"]) & (results["SMA20"] > results["SMA50"]) & (results["RSI"] > 40)
    base_sell = ((results["RSI"] > rsi_sell) & (results.get("Bearish_Div", True))) | (results["Close"] < results.get("Support", results["Close"]))
    ew_bull = (results.get("Elliott_Bullish_Int", 0) == 1) | (results.get("Elliott_Phase_Code", 0) == 1)
    ew_bear = (results.get("Elliott_Bearish_Int", 0) == 1) | (results.get("Elliott_Phase_Code", 0) == -1)
    results["Reversal_Buy"] = reversal_buy | ew_bull
    results["Trend_Buy"] = trend_buy | ew_bull
    results["Sell_Point"] = results["Reversal_Buy"] | results["Trend_Buy"]
    results["Buy_Point"] = base_sell | ew_bear
    return results

def label_from_rule_based(df, rsi_buy=30, rsi_sell=70):
    rules = predict_buy_sell_rule(df, rsi_buy=rsi_buy, rsi_sell=rsi_sell)
    label = pd.Series(0, index=rules.index, dtype=int)
    label[rules["Buy_Point"]] = 1
    label[rules["Sell_Point"]] = -1
    return label

def label_from_future_returns(df, horizon=8, buy_thr=0.05, sell_thr=-0.05):
    fut_ret = df["Close"].shift(-horizon) / df["Close"] - 1.0
    label = pd.Series(0, index=df.index, dtype=int)
    label[fut_ret >= buy_thr] = 1
    label[fut_ret <= sell_thr] = -1
    return label

# ---------------- ML DATASET & TRAINING ----------------
def build_ml_dataset_for_tickers(tickers, sma_windows, support_window, label_mode="rule", horizon=8, buy_thr=0.05, sell_thr=-0.05, rsi_buy=30, rsi_sell=70, min_rows=150, zz_pct=0.05, zz_min_bars=5):
    X_list, y_list, feature_cols = [], [], None
    for t in stqdm(tickers, desc="Preparing ML data"):
        hist = load_history_for_ticker(t, period="5y", interval="1wk")
        if hist is None or len(hist) < min_rows: continue
        feat = compute_features(hist, sma_windows, support_window, zz_pct, zz_min_bars)
        y = label_from_rule_based(feat, rsi_buy=rsi_buy, rsi_sell=rsi_sell) if label_mode == "rule" else label_from_future_returns(feat, horizon, buy_thr, sell_thr)
        data = feat.join(y.rename("Label")).dropna()
        if data.empty: continue
        use = data.select_dtypes(include=[np.number]).drop(columns=["Label", "Support", "Bullish_Div", "Bearish_Div"], errors="ignore")
        if feature_cols is None: feature_cols = list(use.columns)
        X_list.append(use[feature_cols]); y_list.append(data["Label"])
    if not X_list: return pd.DataFrame(), pd.Series(dtype=int), [], pd.Series()
    return pd.concat(X_list), pd.concat(y_list), feature_cols, pd.Series()

##########################################################################
# MODIFIED: PRIMARY ML FUNCTION NOW USES GRADIENT BOOSTING
##########################################################################
def train_gb_classifier(X, y, random_state=42):
    if X.empty or y.empty: return None, None, None
    stratify_opt = y if len(np.unique(y)) > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=stratify_opt, random_state=random_state)
    except:
        split = int(len(X) * 0.8)
        X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    # Swapped RandomForest for GradientBoosting
    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    report = classification_report(y_test, clf.predict(X_test), zero_division=0)
    return clf, acc, report

def compare_ml_models(X, y):
    models = {
        "GradientBoosting": GradientBoostingClassifier(),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=200)
    }
    if XGB_OK: models["XGBoost"] = XGBClassifier(eval_metric="mlogloss", random_state=42)
    results = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            results.append({"Model": name, "Accuracy": accuracy_score(y_test, pred), "F1 Score": f1_score(y_test, pred, average="weighted", zero_division=0)})
        except: pass
    return pd.DataFrame(results).sort_values("Accuracy", ascending=False)

def latest_feature_row_for_ticker(ticker, sma_windows, support_window, feature_cols, zz_pct, zz_min_bars):
    hist = load_history_for_ticker(ticker, period="5y", interval="1wk")
    if hist is None or hist.empty: return None
    feat = compute_features(hist, sma_windows, support_window, zz_pct, zz_min_bars).dropna()
    if feat.empty: return None
    row = feat.select_dtypes(include=[np.number]).iloc[-1:].copy()
    for m in [c for c in feature_cols if c not in row.columns]: row[m] = 0.0
    return row[feature_cols]

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Settings")
    select_all = st.checkbox("Select all stocks", value=True)
    selected_tickers = st.multiselect("Select stocks", NIFTY500_TICKERS, default=NIFTY500_TICKERS if select_all else NIFTY500_TICKERS[:25])
    sma_w1 = st.number_input("SMA 1", 5, 250, 20)
    sma_w2 = st.number_input("SMA 2", 5, 250, 50)
    sma_w3 = st.number_input("SMA 3", 5, 250, 200)
    support_window = st.number_input("Support", 5, 200, 30)
    st.markdown("---")
    zz_pct = st.slider("ZigZag %", 2, 12, 5) / 100.0
    zz_min_bars = st.slider("Min Bars", 3, 15, 5)
    st.markdown("---")
    label_mode = st.radio("ML Mode", ["Rule-based (teach the rules)", "Future Returns"])
    if label_mode == "Rule-based (teach the rules)":
        rsi_buy = st.slider("RSI Buy", 5, 50, 30); rsi_sell = st.slider("RSI Sell", 50, 95, 70)
        ml_horizon, ml_buy_thr, ml_sell_thr = 8, 0.05, -0.05
    else:
        rsi_buy = st.slider("RSI Buy", 5, 50, 30); rsi_sell = st.slider("RSI Sell", 50, 95, 70)
        ml_horizon = st.number_input("Horizon", 1, 52, 8)
        ml_buy_thr = st.number_input("Buy Thr", 0.01, 0.5, 0.05)
        ml_sell_thr = st.number_input("Sell Thr", -0.5, -0.01, -0.05)
    if st.button("Run Weekly Analysis"): st.session_state.analysis_run = True

# ---------------- MAIN ----------------
if st.session_state.analysis_run:
    sma_tuple = (sma_w1, sma_w2, sma_w3)
    feats = get_features_for_all(selected_tickers, sma_tuple, support_window, zz_pct, zz_min_bars)
    if feats.empty: st.error("No valid data.")
    else:
        preds_rule = predict_buy_sell_rule(feats, rsi_buy, rsi_sell)
        tab1, tab2, tab3, tab4 = st.tabs(["✅ Rule Buy", "❌ Rule Sell", "📈 Chart", "🤖 ML Signals"])
        
        with tab1:
            df_buy = preds_rule[preds_rule["Buy_Point"]].copy()
            if not df_buy.empty: st.write(add_tradingview_links(df_buy).to_html(escape=False, index=False), unsafe_allow_html=True)
        with tab2:
            df_sell = preds_rule[preds_rule["Sell_Point"]].copy()
            if not df_sell.empty: st.write(add_tradingview_links(df_sell).to_html(escape=False, index=False), unsafe_allow_html=True)
        with tab3:
            t_chart = st.selectbox("Chart Ticker", selected_tickers)
            c_df = yf.download(t_chart, period="3y", interval="1wk", progress=False)
            if not c_df.empty:
                c_df = compute_features(c_df, sma_tuple, support_window, zz_pct, zz_min_bars).dropna()
                st.line_chart(c_df[["Close", f"SMA{sma_w1}", f"SMA{sma_w2}", f"SMA{sma_w3}"]])
        with tab4:
            X, y, f_cols, _ = build_ml_dataset_for_tickers(selected_tickers, sma_tuple, support_window, label_mode, ml_horizon, ml_buy_thr, ml_sell_thr, rsi_buy, rsi_sell, 150, zz_pct, zz_min_bars)
            if X.empty: st.warning("Not enough data.")
            else:
                clf, acc, report = train_gb_classifier(X, y)
                st.caption(f"Gradient Boosting Accuracy: **{acc:.3f}**")
                st.subheader("📊 Model Comparison")
                st.dataframe(compare_ml_models(X, y), use_container_width=True)
                
                rows = []
                for t in stqdm(selected_tickers, desc="Scoring"):
                    r = latest_feature_row_for_ticker(t, sma_tuple, support_window, f_cols, zz_pct, zz_min_bars)
                    if r is not None:
                        pred = clf.predict(r)[0]; prob = clf.predict_proba(r)[0]
                        rows.append({"Ticker": t, "ML_Pred": {1:"BUY", 0:"HOLD", -1:"SELL"}[int(pred)], "Confidence": max(prob)})
                
                ml_df = pd.DataFrame(rows).merge(feats[["Ticker", "Close"]], on="Ticker")
                st.dataframe(ml_df, use_container_width=True)
                st.download_button("📥 Download CSV", ml_df.to_csv(index=False).encode(), "signals.csv", "text/csv")
        st.markdown("---")
        st.markdown("⚠ Educational use only — not financial advice.")
