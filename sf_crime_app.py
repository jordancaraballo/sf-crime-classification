"""
SF Crime Classification - Streamlit App
Based on: https://github.com/OscarLacomba/sf-crime-classification
Dataset: SFPD Crime Incident Reporting System (2003-2015) via Kaggle
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
import warnings
import io

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SF Crime Classification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.4rem; font-weight: 800; color: #c0392b; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1rem; color: #555; margin-bottom: 1.5rem; }
    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #2c3e50;
        border-bottom: 2px solid #c0392b; padding-bottom: 4px; margin-bottom: 1rem;
    }
    .info-box {
        background: #eaf4fb; border: 1px solid #aed6f1; border-radius: 6px;
        padding: 0.8rem 1rem; margin-bottom: 1rem; font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CRIME_CATEGORIES = [
    "LARCENY/THEFT", "OTHER OFFENSES", "NON-CRIMINAL", "ASSAULT",
    "DRUG/NARCOTIC", "VEHICLE THEFT", "VANDALISM", "WARRANTS",
    "BURGLARY", "SUSPICIOUS OCC", "MISSING PERSON", "ROBBERY",
    "FRAUD", "FORGERY/COUNTERFEITING", "SECONDARY CODES",
    "WEAPON LAWS", "PROSTITUTION", "TRESPASS", "STOLEN PROPERTY",
    "SEX OFFENSES FORCIBLE", "DISORDERLY CONDUCT", "DRUNKENNESS",
    "RECOVERED VEHICLE", "KIDNAPPING", "DRIVING UNDER THE INFLUENCE",
    "RUNAWAY", "LIQUOR LAWS", "ARSON", "LOITERING",
    "EMBEZZLEMENT", "SUICIDE", "FAMILY OFFENSES", "BAD CHECKS",
    "BRIBERY", "EXTORTION", "SEX OFFENSES NON FORCIBLE",
    "GAMBLING", "PORNOGRAPHY/OBSCENE MAT", "TREA"
]

PD_DISTRICTS = [
    "SOUTHERN", "MISSION", "NORTHERN", "BAYVIEW", "CENTRAL",
    "TENDERLOIN", "INGLESIDE", "TARAVAL", "PARK", "RICHMOND"
]

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

FEATURES = ["Hour", "Month", "Year", "DayOfWeek_enc", "District_enc", "X", "Y"]

# ─────────────────────────────────────────────
# Synthetic data
# ─────────────────────────────────────────────
@st.cache_data
def generate_synthetic_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2003-01-06", "2015-05-13", periods=n_samples)
    dates = dates[rng.choice(len(dates), n_samples, replace=True)]
    hours = rng.integers(0, 24, n_samples)
    day_of_week = [DAYS_OF_WEEK[d] for d in rng.integers(0, 7, n_samples)]
    cat_weights = np.array([
        157182, 62453, 60583, 55956, 46246, 41988, 29789, 28507,
        24975, 21631, 21364, 18994, 16677, 10501, 9963, 8441,
        7990, 7326, 5765, 4931, 4621, 4546, 3898, 2341,
        2268, 1946, 1901, 1748, 1653, 1380, 1142, 883,
        781, 680, 634, 452, 360, 274, 84
    ], dtype=float)
    categories = rng.choice(CRIME_CATEGORIES, n_samples, p=cat_weights / cat_weights.sum())
    district_weights = np.array([157182, 119908, 105296, 91782, 87480,
                                  78372, 76644, 72007, 50464, 46696], dtype=float)
    districts = rng.choice(PD_DISTRICTS, n_samples, p=district_weights / district_weights.sum())
    lat = rng.uniform(37.70, 37.83, n_samples)
    lon = rng.uniform(-122.52, -122.35, n_samples)
    resolutions = rng.choice(
        ["NONE", "ARREST, BOOKED", "ARREST, CITED", "LOCATED", "PSYCHOPATHIC CASE", "NOT PROSECUTED"],
        n_samples, p=[0.55, 0.22, 0.10, 0.06, 0.04, 0.03]
    )
    df = pd.DataFrame({
        "Dates": dates, "Category": categories, "DayOfWeek": day_of_week,
        "PdDistrict": districts, "Resolution": resolutions, "X": lon, "Y": lat,
        "Hour": hours, "Month": pd.DatetimeIndex(dates).month, "Year": pd.DatetimeIndex(dates).year,
    })
    return df

# ─────────────────────────────────────────────
# Feature engineering
# Uses fixed-list encoding so prediction encodes identically to training
# ─────────────────────────────────────────────
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Dates" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Dates"]):
        df["Dates"] = pd.to_datetime(df["Dates"], errors="coerce")
    if "Dates" in df.columns:
        df["Hour"] = df["Dates"].dt.hour
        df["Month"] = df["Dates"].dt.month
        df["Year"] = df["Dates"].dt.year
        df["DayOfWeek"] = df["Dates"].dt.day_name()
    # Use index into the fixed constant lists — guaranteed to match at predict time
    df["DayOfWeek_enc"] = df["DayOfWeek"].apply(
        lambda d: DAYS_OF_WEEK.index(d) if d in DAYS_OF_WEEK else 0
    )
    df["District_enc"] = df["PdDistrict"].apply(
        lambda d: PD_DISTRICTS.index(d) if d in PD_DISTRICTS else 0
    )
    return df

# ─────────────────────────────────────────────
# Model training — returns scaler alongside other objects
# ─────────────────────────────────────────────
def train_models(df, model_names, top_n_classes, n_estimators_rf, test_size):
    df = feature_engineer(df)
    top_cats = df["Category"].value_counts().head(top_n_classes).index.tolist()
    df = df[df["Category"].isin(top_cats)].copy()

    le_cat = LabelEncoder()
    df["label"] = le_cat.fit_transform(df["Category"])

    X = df[FEATURES].fillna(0).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Fit scaler on training data only; transform both splits
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model_map = {}
    if "Random Forest" in model_names:
        model_map["Random Forest"] = RandomForestClassifier(
            n_estimators=n_estimators_rf, max_depth=15,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )
    if "Gradient Boosting" in model_names:
        model_map["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=min(n_estimators_rf, 100), max_depth=5,
            learning_rate=0.1, random_state=42
        )
    if "Logistic Regression" in model_names:
        model_map["Logistic Regression"] = LogisticRegression(
            max_iter=500, random_state=42, C=1.0, solver="lbfgs"
        )

    results = {}
    for name, model in model_map.items():
        model.fit(X_train_s, y_train)
        y_pred  = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s) if hasattr(model, "predict_proba") else None
        results[name] = {
            "model":    model,
            "acc":      accuracy_score(y_test, y_pred),
            "log_loss": log_loss(y_test, y_proba) if y_proba is not None else None,
            "report":   classification_report(y_test, y_pred, target_names=le_cat.classes_, output_dict=True),
            "cm":       confusion_matrix(y_test, y_pred),
            "y_test":   y_test,
            "y_pred":   y_pred,
        }

    fi = None
    if "Random Forest" in results:
        fi = pd.Series(
            results["Random Forest"]["model"].feature_importances_, index=FEATURES
        ).sort_values(ascending=False)

    # Return scaler so it can be stored in session state and reused at predict time
    return results, le_cat, fi, scaler

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Flag_of_San_Francisco.svg/200px-Flag_of_San_Francisco.svg.png", width=60)
    st.markdown("## ⚙️ Configuration")

    st.markdown("### 📂 Data Source")
    uploaded = st.file_uploader("Upload train.csv (optional)", type=["csv"],
        help="Upload the Kaggle SF Crime train.csv. If not provided, synthetic demo data is used.")
    n_synthetic = st.slider("Synthetic samples (if no upload)", 1000, 20000, 5000, step=500)

    st.markdown("---")
    st.markdown("### 🤖 Models to Train")
    model_choices = st.multiselect(
        "Select classifiers",
        ["Random Forest", "Gradient Boosting", "Logistic Regression"],
        default=["Random Forest", "Logistic Regression"]
    )

    st.markdown("### 🔢 Training Settings")
    top_n     = st.slider("Top N crime categories", 5, 39, 15)
    test_frac = st.slider("Test split fraction", 0.1, 0.4, 0.2, step=0.05)
    n_trees   = st.slider("Trees (RF / GB)", 20, 200, 80, step=10)

    st.markdown("---")
    train_btn = st.button("🚀 Train Models", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### 🔮 Crime Predictor")
    pred_hour     = st.slider("Hour of day", 0, 23, 12)
    pred_day      = st.selectbox("Day of week", DAYS_OF_WEEK)
    pred_month    = st.slider("Month", 1, 12, 6)
    pred_district = st.selectbox("PD District", PD_DISTRICTS)
    pred_lat      = st.number_input("Latitude",  37.70, 37.83, 37.77, step=0.005, format="%.4f")
    pred_lon      = st.number_input("Longitude", -122.52, -122.35, -122.42, step=0.005, format="%.4f")
    predict_btn   = st.button("🔍 Predict Crime Category", use_container_width=True)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
@st.cache_data
def load_uploaded(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

if uploaded:
    raw_df      = load_uploaded(uploaded.read())
    data_source = "Uploaded CSV"
else:
    raw_df      = generate_synthetic_data(n_synthetic)
    data_source = f"Synthetic demo data ({n_synthetic:,} samples)"

df_eng = feature_engineer(raw_df.copy())

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<p class="main-header">🔍 SF Crime Classification</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="sub-header">Supervised ML on the SFPD Crime Dataset (2003-2015) &nbsp;|&nbsp; '
    f'Data: <b>{data_source}</b> &nbsp;|&nbsp; {len(raw_df):,} records</p>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# Session state — initialise all keys once
# ─────────────────────────────────────────────
for key in ["results", "le_cat", "fi", "scaler"]:
    if key not in st.session_state:
        st.session_state[key] = None

if train_btn:
    if not model_choices:
        st.warning("Please select at least one model.")
    else:
        with st.spinner("Training models… this may take a moment ⏳"):
            res, le, fi, scaler = train_models(raw_df, model_choices, top_n, n_trees, test_frac)
        st.session_state.results = res
        st.session_state.le_cat  = le
        st.session_state.fi      = fi
        st.session_state.scaler  = scaler   # ← stored here
        st.success("✅ Models trained successfully!")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA", "🧠 Model Results", "📈 Feature Importance", "🗺️ Crime Map", "🔮 Predictor"
])

# ══════════════════════════════════════════════
# TAB 1 – EDA
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Exploratory Data Analysis</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",    f"{len(raw_df):,}")
    c2.metric("Crime Categories", raw_df["Category"].nunique()  if "Category"   in raw_df.columns else "—")
    c3.metric("PD Districts",     raw_df["PdDistrict"].nunique() if "PdDistrict" in raw_df.columns else "—")
    c4.metric("Years Covered",    f"{df_eng['Year'].min()}–{df_eng['Year'].max()}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Top Crime Categories**")
        top_crimes = raw_df["Category"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = plt.cm.Reds_r(np.linspace(0.2, 0.8, len(top_crimes)))
        top_crimes.plot(kind="barh", ax=ax, color=colors)
        ax.set_xlabel("Count"); ax.invert_yaxis(); ax.tick_params(labelsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col_b:
        st.markdown("**Crimes by Day of Week**")
        dow_counts = raw_df["DayOfWeek"].value_counts().reindex(DAYS_OF_WEEK, fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(dow_counts.index, dow_counts.values, color="#c0392b", alpha=0.8)
        ax.set_ylabel("Crimes"); ax.tick_params(axis="x", rotation=30, labelsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("**Crimes by Hour of Day**")
        hour_counts = df_eng["Hour"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(hour_counts.index, hour_counts.values, color="#c0392b", linewidth=2.5, marker="o", markersize=4)
        ax.fill_between(hour_counts.index, hour_counts.values, alpha=0.15, color="#c0392b")
        ax.set_xlabel("Hour (24h)"); ax.set_ylabel("Crimes"); ax.set_xticks(range(0, 24, 2))
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col_d:
        st.markdown("**Crimes by Police District**")
        dist_counts = raw_df["PdDistrict"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        dist_counts.plot(kind="bar", ax=ax, color="#2980b9", alpha=0.85)
        ax.set_ylabel("Crimes"); ax.tick_params(axis="x", rotation=35, labelsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("**Crimes per Year**")
    year_counts = df_eng["Year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(year_counts.index, year_counts.values, color="#16a085", alpha=0.85)
    ax.set_xlabel("Year"); ax.set_ylabel("Crimes")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("**Crime Heatmap: Hour × Day of Week**")
    pivot_data = df_eng.groupby(["DayOfWeek", "Hour"]).size().unstack(fill_value=0).reindex(DAYS_OF_WEEK)
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(pivot_data, ax=ax, cmap="YlOrRd", linewidths=0.3, cbar_kws={"label": "Crimes"})
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Day of Week")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with st.expander("📋 Raw Data Preview"):
        st.dataframe(raw_df.head(200), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 – Model Results
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Model Evaluation</p>', unsafe_allow_html=True)

    if st.session_state.results is None:
        st.markdown('<div class="info-box">⬅️ Configure your models in the sidebar and click <b>Train Models</b> to see results here.</div>', unsafe_allow_html=True)
    else:
        results = st.session_state.results
        le_cat  = st.session_state.le_cat

        st.markdown("**Accuracy Comparison**")
        acc_data = {name: r["acc"] for name, r in results.items()}
        fig, ax = plt.subplots(figsize=(max(4, len(acc_data) * 2.5), 4))
        bars = ax.bar(acc_data.keys(), acc_data.values(),
                      color=["#c0392b", "#2980b9", "#27ae60"][:len(acc_data)], alpha=0.85)
        ax.axhline(0.80, color="orange", linestyle="--", linewidth=1.5, label="80% target")
        ax.set_ylim(0, 1); ax.set_ylabel("Accuracy"); ax.legend()
        for bar, val in zip(bars, acc_data.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        for name, r in results.items():
            with st.expander(f"📊 {name} — Accuracy: {r['acc']:.4f}", expanded=(name == list(results.keys())[0])):
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Accuracy",  f"{r['acc']:.4f}")
                mc2.metric("Log Loss",  f"{r['log_loss']:.4f}" if r["log_loss"] else "—")
                macro = r["report"].get("macro avg", {})
                mc3.metric("Macro F1",  f"{macro.get('f1-score', 0):.4f}")

                st.markdown("**Confusion Matrix (top classes)**")
                class_names = le_cat.classes_
                cm          = r["cm"]
                n_show      = min(10, len(class_names))
                top_idx     = np.argsort(cm.sum(axis=1))[-n_show:][::-1]
                cm_sub      = cm[np.ix_(top_idx, top_idx)]
                sub_names   = [class_names[i][:18] for i in top_idx]
                fig, ax = plt.subplots(figsize=(10, 7))
                sns.heatmap(cm_sub, annot=True, fmt="d", cmap="Blues",
                            xticklabels=sub_names, yticklabels=sub_names, ax=ax,
                            linewidths=0.3, annot_kws={"size": 8})
                ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                ax.tick_params(axis="x", rotation=45, labelsize=7)
                ax.tick_params(axis="y", rotation=0,  labelsize=7)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                st.markdown("**Per-class Metrics**")
                report_df = pd.DataFrame(r["report"]).T
                report_df = report_df.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
                report_df = report_df[["precision", "recall", "f1-score", "support"]].round(3)
                report_df["support"] = report_df["support"].astype(int)
                st.dataframe(report_df.sort_values("support", ascending=False), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 – Feature Importance
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Feature Importance</p>', unsafe_allow_html=True)

    FEATURE_LABELS = {
        "Hour": "Hour of Day", "Month": "Month", "Year": "Year",
        "DayOfWeek_enc": "Day of Week", "District_enc": "PD District",
        "X": "Longitude", "Y": "Latitude",
    }

    if st.session_state.fi is None:
        st.markdown('<div class="info-box">Train a <b>Random Forest</b> model to see feature importances.</div>', unsafe_allow_html=True)
    else:
        fi = st.session_state.fi.rename(FEATURE_LABELS)
        col_fi1, col_fi2 = st.columns([2, 1])
        with col_fi1:
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(fi)))[::-1]
            fi.plot(kind="barh", ax=ax, color=colors)
            ax.set_xlabel("Importance Score"); ax.set_title("Random Forest Feature Importances")
            ax.invert_yaxis()
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        with col_fi2:
            st.dataframe(
                fi.reset_index().rename(columns={"index": "Feature", 0: "Importance"}).round(4),
                use_container_width=True
            )

    st.markdown("---")
    st.markdown("**Crime Distribution across Features**")
    feat_choice = st.selectbox("Select feature to explore", ["Hour", "Month", "DayOfWeek", "PdDistrict"])
    top10_cats  = raw_df["Category"].value_counts().head(8).index.tolist()
    sub         = df_eng[df_eng["Category"].isin(top10_cats)]
    fig, ax     = plt.subplots(figsize=(14, 5))
    palette     = sns.color_palette("tab10", n_colors=len(top10_cats))
    for i, cat in enumerate(top10_cats):
        grp = sub[sub["Category"] == cat]
        if feat_choice == "DayOfWeek":
            counts = grp["DayOfWeek"].value_counts().reindex(DAYS_OF_WEEK, fill_value=0)
        elif feat_choice == "PdDistrict":
            counts = grp["PdDistrict"].value_counts().reindex(PD_DISTRICTS, fill_value=0)
        else:
            counts = grp[feat_choice].value_counts().sort_index()
        ax.plot(counts.index, counts.values, label=cat[:20], color=palette[i], linewidth=1.5)
    ax.set_xlabel(feat_choice); ax.set_ylabel("Count")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ══════════════════════════════════════════════
# TAB 4 – Crime Map
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">Geographic Distribution</p>', unsafe_allow_html=True)

    map_cat = st.selectbox("Filter by category (map)", ["All"] + CRIME_CATEGORIES[:20])
    map_df  = df_eng.dropna(subset=["X", "Y"])
    map_df  = map_df[(map_df["X"] > -123) & (map_df["X"] < -122) &
                     (map_df["Y"] > 37)   & (map_df["Y"] < 38)]
    if map_cat != "All":
        map_df = map_df[map_df["Category"] == map_cat]
    sample_map = map_df.sample(min(3000, len(map_df)), random_state=1)
    st.map(sample_map.rename(columns={"Y": "latitude", "X": "longitude"})[["latitude", "longitude"]], zoom=11)

    st.markdown("**Crime Density by District (bubble chart)**")
    dist_summary = raw_df.groupby("PdDistrict").size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(
        range(len(dist_summary)), dist_summary["count"],
        s=dist_summary["count"] / dist_summary["count"].max() * 3000,
        c=dist_summary["count"], cmap="Reds", alpha=0.8, edgecolors="grey"
    )
    ax.set_xticks(range(len(dist_summary)))
    ax.set_xticklabels(dist_summary["PdDistrict"], rotation=30, ha="right")
    ax.set_ylabel("Total Crimes")
    plt.colorbar(scatter, ax=ax, label="Crime Count")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ══════════════════════════════════════════════
# TAB 5 – Predictor
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">🔮 Crime Category Predictor</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Configure inputs in the sidebar, train a model, then click <b>Predict Crime Category</b>.</div>', unsafe_allow_html=True)

    if predict_btn:
        if st.session_state.results is None or st.session_state.scaler is None:
            st.warning("Please train a model first.")
        else:
            results = st.session_state.results
            le_cat  = st.session_state.le_cat
            scaler  = st.session_state.scaler   # ← retrieved here

            day_enc  = DAYS_OF_WEEK.index(pred_day)
            dist_enc = PD_DISTRICTS.index(pred_district)

            # Build raw vector in the same column order as FEATURES
            raw_vec  = np.array([[
                pred_hour,    # Hour
                pred_month,   # Month
                2009,         # Year — midpoint of 2003-2015 training range
                day_enc,      # DayOfWeek_enc
                dist_enc,     # District_enc
                pred_lon,     # X
                pred_lat,     # Y
            ]])

            # Apply the SAME scaler that was fitted during training
            feat_vec = scaler.transform(raw_vec)

            st.markdown("### 🎯 Predictions")
            for name, r in results.items():
                model         = r["model"]
                pred_label    = model.predict(feat_vec)[0]
                pred_category = le_cat.inverse_transform([pred_label])[0]
                proba         = model.predict_proba(feat_vec)[0]
                top5_idx      = np.argsort(proba)[-5:][::-1]
                top5_cats     = le_cat.inverse_transform(top5_idx)
                top5_probs    = proba[top5_idx]

                with st.container():
                    st.markdown(f"**{name}** → `{pred_category}` (confidence: `{max(proba)*100:.1f}%`)")
                    fig, ax = plt.subplots(figsize=(8, 2.5))
                    bar_colors = (["#aab7c4"] * 4 + ["#c0392b"])[::-1]  # highlight top bar
                    ax.barh(top5_cats[::-1], top5_probs[::-1] * 100, color=bar_colors)
                    ax.set_xlabel("Probability (%)")
                    ax.set_xlim(0, 100)
                    for i, (cat, p) in enumerate(zip(top5_cats[::-1], top5_probs[::-1])):
                        ax.text(p * 100 + 0.5, i, f"{p*100:.1f}%", va="center", fontsize=9)
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>Built from <a href='https://github.com/OscarLacomba/sf-crime-classification' target='_blank'>"
    "OscarLacomba/sf-crime-classification</a> · "
    "Dataset: SFPD Crime Incident Reporting System via Kaggle SF Crime 2003–2015</small>",
    unsafe_allow_html=True
)