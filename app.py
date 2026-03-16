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

# Check for sklearn
try:
    from sklearn.ensemble import GradientBoostingClassifier
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# Initialize session state
if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Nifty500 Weekly Predictor - Gradient Boosting", layout="wide")
st.title("📊 Nifty500 Predictor (Weekly) — Gradient Boosting + Elliott Wave")

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

# ---------------- ELLIOTT WAVE & FEATURES ----------------
def zigzag_pivots(close: pd.Series, pct=0.05, min_bars=5):
    if close.isna().all() or len(close) < 50: return pd.DataFrame()
    c = close.values.astype(float)
    idxs = close.index
    piv = []
    last_piv_i, last_piv_p, trend = 0, c[0], None
    last_extreme_i, last_extreme_p = 0, c[0]

    for i in range(1, len(c)):
        if trend in (None, "up"):
            if c[i] > last_extreme_p: last_extreme_p, last_extreme_i = c[i], i
        if trend in (None, "down"):
            if c[i] < last_extreme_p: last_extreme_p, last_extreme_i = c[i], i
        
        if trend in (None, "up") and last_extreme_p != 0:
            if (c[i] - last_extreme_p) / last_extreme_p <= -pct and (i - last_piv_i) >= min_bars:
                piv.append((idxs[last_extreme_i], float(last_extreme_p), "H"))
                last_piv_i, trend = last_extreme_i, "down"
                last_extreme_p, last_extreme_i = c[i], i
        elif trend in (None, "down") and last_extreme_p != 0:
            if (c[i] - last_extreme_p) / last_extreme_p >= pct and (i - last_piv_i) >= min_bars:
                piv.append((idxs[last_extreme_i], float(last_extreme_p), "L"))
                last_piv_i, trend = last_extreme_i, "up"
                last_extreme_p, last_extreme_i = c[i], i
    return pd.DataFrame(piv, columns=["idx", "price", "type"]) if piv else pd.DataFrame()

def elliott_phase_from_pivots(pivots: pd.DataFrame):
    out = {"phase": "Unknown", "wave_no": 0, "bullish": False, "bearish": False}
    if len(pivots) < 3: return out
    p3 = pivots.iloc[-3:]
    types = p3["type"].tolist()
    if types == ["L", "H", "L"]: out.update({"phase": "CorrectionUp", "wave_no": 3, "bullish": True})
    elif types == ["H", "L", "H"]: out.update({"phase": "CorrectionDown", "wave_no": 3, "bearish": True})
    return out

def compute_features(df, sma_windows=(20, 50, 200), support_window=30, zz_pct=0.05, zz_min_bars=5):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    if "Close" not in df.columns or df["Close"].dropna().empty: return pd.DataFrame()
    
    df = df.copy()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    for win in sma_windows: df[f"SMA{win}"] = df["Close"].rolling(window=win).mean()
    df["Support"] = df["Close"].rolling(window=support_window).min()
    
    # Elliott
    piv = zigzag_pivots(df["Close"], pct=zz_pct, min_bars=zz_min_bars)
    phase = elliott_phase_from_pivots(piv)
    phase_map = {"ImpulseUp": 1, "ImpulseDown": -1, "CorrectionUp": 2, "CorrectionDown": -2, "Unknown": 0}
    df["Elliott_Phase_Code"] = phase_map.get(phase["phase"], 0)
    df["Elliott_Bullish_Int"] = int(phase["bullish"])
    df["Elliott_Bearish_Int"] = int(phase["bearish"])
    
    # Returns/Distances
    for w in (1, 5): df[f"Ret_{w}"] = df["Close"].pct_change(w)
    for win in sma_windows: df[f"Dist_SMA{win}"] = (df["Close"] - df[f"SMA{win}"]) / df[f"SMA{win}"]
    return df

# ---------------- ML ENGINE (GRADIENT BOOSTING) ----------------
def train_gb_classifier(X, y, random_state=42):
    if X.empty or y.empty: return None, None, None
    stratify_opt = y if len(np.unique(y)) > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=stratify_opt, random_state=random_state)
    except:
        split = int(len(X) * 0.8)
        X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    # Updated to Gradient Boosting
    clf = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=4, 
        subsample=0.8,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    report = classification_report(y_test, clf.predict(X_test), zero_division=0)
    return clf, acc, report

def compare_ml_models(X, y):
    models = {
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=500)
    }
    if XGB_OK: models["XGBoost"] = XGBClassifier(eval_metric="mlogloss")
    
    results = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            results.append({"Model": name, "Accuracy": accuracy_score(y_test, pred), "F1 Score": f1_score(y_test, pred, average="weighted", zero_division=0)})
        except: pass
    return pd.DataFrame(results).sort_values("Accuracy", ascending=False)

# ---------------- APP LOGIC HELPERS ----------------
def get_features_for_all(tickers, sma_windows, support_window, zz_pct, zz_min_bars):
    multi_df = download_data_multi(tickers)
    if multi_df is None: return pd.DataFrame()
    feats_list = []
    available = multi_df.columns.get_level_values(0).unique()
    for t in tickers:
        if t in available:
            df = compute_features(multi_df[t].dropna(), sma_windows, support_window, zz_pct, zz_min_bars).dropna()
            if not df.empty:
                row = df.iloc[-1].to_dict()
                row["Ticker"] = t
                feats_list.append(row)
    return pd.DataFrame(feats_list)

def predict_buy_sell_rule(df, rsi_buy=30, rsi_sell=70):
    res = df.copy()
    res["Buy_Point"] = (res["RSI"] < rsi_buy) | (res["Elliott_Bullish_Int"] == 1)
    res["Sell_Point"] = (res["RSI"] > rsi_sell) | (res["Elliott_Bearish_Int"] == 1)
    return res

def build_ml_dataset_for_tickers(tickers, sma_windows, support_window, label_mode="rule", rsi_buy=30, rsi_sell=70, zz_pct=0.05, zz_min_bars=5):
    X_list, y_list = [], []
    for t in stqdm(tickers, desc="Data Prep"):
        hist = load_history_for_ticker(t)
        if len(hist) < 100: continue
        feat = compute_features(hist, sma_windows, support_window, zz_pct, zz_min_bars).dropna()
        if feat.empty: continue
        
        # Simple labeling
        y = pd.Series(0, index=feat.index)
        y[(feat["RSI"] < rsi_buy)] = 1
        y[(feat["RSI"] > rsi_sell)] = -1
        
        data = feat.join(y.rename("Label")).dropna()
        X_list.append(data.select_dtypes(include=[np.number]).drop(columns=["Label"]))
        y_list.append(data["Label"])
    
    if not X_list: return pd.DataFrame(), pd.Series(), []
    X = pd.concat(X_list)
    return X, pd.concat(y_list), X.columns.tolist()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Settings")
    selected_tickers = st.multiselect("Stocks", NIFTY500_TICKERS, default=NIFTY500_TICKERS[:10])
    sma_w1 = st.number_input("SMA 1", 5, 200, 20)
    sma_w2 = st.number_input("SMA 2", 5, 200, 50)
    sma_w3 = st.number_input("SMA 3", 5, 200, 200)
    zz_pct = st.slider("ZigZag %", 1, 15, 5) / 100.0
    zz_min_bars = st.slider("Min Bars", 1, 10, 5)
    rsi_buy = st.slider("RSI Buy", 10, 50, 30)
    rsi_sell = st.slider("RSI Sell", 50, 90, 70)
    if st.button("Run Weekly Analysis"): st.session_state.analysis_run = True

# ---------------- MAIN ----------------
if st.session_state.analysis_run:
    sma_tuple = (sma_w1, sma_w2, sma_w3)
    feats = get_features_for_all(selected_tickers, sma_tuple, 30, zz_pct, zz_min_bars)
    
    if not feats.empty:
        preds_rule = predict_buy_sell_rule(feats, rsi_buy, rsi_sell)
        tab1, tab2, tab3 = st.tabs(["✅ Rule Buy", "🤖 ML Signals", "📊 Comparison"])
        
        with tab1:
            st.dataframe(preds_rule[preds_rule["Buy_Point"]][["Ticker", "Close", "RSI"]])
        
        with tab2:
            X, y, cols = build_ml_dataset_for_tickers(selected_tickers, sma_tuple, 30, rsi_buy=rsi_buy, rsi_sell=rsi_sell)
            clf, acc, rep = train_gb_classifier(X, y)
            st.success(f"Gradient Boosting Model Accuracy: {acc:.2%}")
            
            # Predict latest
            ml_results = []
            for t in selected_tickers:
                hist = load_history_for_ticker(t)
                f = compute_features(hist, sma_tuple, 30, zz_pct, zz_min_bars).dropna()
                if not f.empty:
                    row = f.iloc[-1:][cols]
                    pred = clf.predict(row)[0]
                    prob = clf.predict_proba(row)[0]
                    ml_results.append({"Ticker": t, "ML_Pred": {1:"BUY", 0:"HOLD", -1:"SELL"}[pred], "Confidence": max(prob)})
            st.dataframe(pd.DataFrame(ml_results))

        with tab3:
            st.subheader("Model Comparison")
            st.table(compare_ml_models(X, y))
