"""
Sales Forecasting & Demand Intelligence Dashboard
Task 7 — Streamlit deployment

Run locally with:  streamlit run app.py
Deploy on Streamlit Community Cloud by pushing this repo (with train.csv,
requirements.txt, and the .streamlit/config.toml theme file) to GitHub and
connecting it at share.streamlit.io.

UI NOTE: All forecasting, anomaly detection, clustering, filters, and model
logic are 100% unchanged from the original app. Only presentation (theme,
layout, KPI cards, chart styling, table styling, sidebar branding) was added.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens / palette
# ---------------------------------------------------------------------------
PALETTE = ["#3B82F6", "#22D3EE", "#818CF8", "#34D399", "#F472B6", "#FBBF24", "#F97316"]
BG_CARD = "#111827"
BG_CARD_ALT = "#0B1220"
TEXT_MUTED = "#94A3B8"
GRID_COLOR = "rgba(255,255,255,0.08)"
FONT_FAMILY = "'Inter', 'Segoe UI', sans-serif"

# ---------------------------------------------------------------------------
# Global styling (theme, KPI cards, containers, sidebar, widgets, tables)
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: {FONT_FAMILY};
}}

/* App background — deep navy/charcoal gradient */
.stApp {{
    background: radial-gradient(circle at 15% 0%, #10192E 0%, #0B1220 45%, #070C16 100%);
}}

/* Hide default hamburger footer clutter, keep menu */
footer {{visibility: hidden;}}

/* ---- Header / title block ---- */
.app-header {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 6px 0 18px 0;
    margin-bottom: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}}
.app-header .icon-badge {{
    width: 48px; height: 48px;
    border-radius: 14px;
    background: linear-gradient(135deg, #3B82F6 0%, #22D3EE 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    box-shadow: 0 8px 20px rgba(59,130,246,0.35);
    flex-shrink: 0;
}}
.app-header h1 {{
    font-size: 26px;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #FFFFFF 0%, #A5B4FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.app-header p {{
    margin: 2px 0 0 0;
    color: {TEXT_MUTED};
    font-size: 13.5px;
}}

/* ---- Section titles above chart cards ---- */
.section-title {{
    font-size: 16px;
    font-weight: 700;
    color: #F1F5F9;
    margin: 0 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.section-label {{
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT_MUTED};
    font-weight: 700;
    margin: 22px 0 10px 0;
}}

/* ---- KPI cards ---- */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 26px;
}}
@media (max-width: 900px) {{
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}
.kpi-card {{
    background: linear-gradient(160deg, {BG_CARD} 0%, {BG_CARD_ALT} 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 10px 24px rgba(0,0,0,0.35);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    position: relative;
    overflow: hidden;
}}
.kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 16px 32px rgba(59,130,246,0.18);
    border-color: rgba(59,130,246,0.45);
}}
.kpi-card::after {{
    content: "";
    position: absolute;
    top: -30px; right: -30px;
    width: 90px; height: 90px;
    background: radial-gradient(circle, rgba(59,130,246,0.20) 0%, rgba(59,130,246,0) 70%);
    border-radius: 50%;
}}
.kpi-top {{
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 10px;
}}
.kpi-icon {{
    width: 38px; height: 38px;
    border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}}
.kpi-icon.blue   {{ background: rgba(59,130,246,0.16); }}
.kpi-icon.cyan   {{ background: rgba(34,211,238,0.16); }}
.kpi-icon.violet {{ background: rgba(129,140,248,0.16); }}
.kpi-icon.green  {{ background: rgba(52,211,153,0.16); }}
.kpi-label {{
    font-size: 12.5px;
    font-weight: 600;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.kpi-value {{
    font-size: 26px;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.1;
}}
.kpi-sub {{
    margin-top: 6px;
    font-size: 12px;
    color: {TEXT_MUTED};
}}

/* ---- Chart / content cards (st.container(border=True)) ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: linear-gradient(160deg, {BG_CARD} 0%, {BG_CARD_ALT} 100%) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    box-shadow: 0 10px 24px rgba(0,0,0,0.30) !important;
    padding: 6px 6px 2px 6px !important;
    transition: box-shadow 0.18s ease, border-color 0.18s ease;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    box-shadow: 0 14px 30px rgba(59,130,246,0.14) !important;
    border-color: rgba(59,130,246,0.30) !important;
}}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D1424 0%, #070C16 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}}
.sidebar-brand {{
    display: flex; align-items: center; gap: 10px;
    padding: 4px 0 16px 0;
    margin-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}}
.sidebar-brand .badge {{
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #3B82F6, #22D3EE);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}}
.sidebar-brand .brand-name {{
    font-weight: 800; font-size: 15px; color: #F8FAFC; line-height: 1.1;
}}
.sidebar-brand .brand-sub {{
    font-size: 11px; color: {TEXT_MUTED}; letter-spacing: 0.04em;
}}
.sidebar-footer {{
    margin-top: 30px;
    padding-top: 14px;
    border-top: 1px solid rgba(255,255,255,0.07);
    font-size: 11px;
    color: {TEXT_MUTED};
}}
section[data-testid="stSidebar"] .stRadio label {{
    font-size: 14.5px !important;
    padding: 4px 0;
}}
section[data-testid="stSidebar"] .stRadio > div {{
    gap: 4px;
}}

/* ---- Metrics ---- */
div[data-testid="stMetric"] {{
    background: linear-gradient(160deg, {BG_CARD} 0%, {BG_CARD_ALT} 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 8px 18px rgba(0,0,0,0.25);
}}
div[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; }}

/* ---- Widgets: selects, sliders, multiselect chips ---- */
.stSlider [data-baseweb="slider"] > div > div {{ background: #3B82F6 !important; }}
div[data-baseweb="select"] > div {{
    background-color: {BG_CARD} !important;
    border-color: rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
}}
span[data-baseweb="tag"] {{
    background-color: rgba(59,130,246,0.25) !important;
    border-radius: 8px !important;
}}

/* ---- Dataframe / table container ---- */
div[data-testid="stDataFrame"] {{
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07);
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached so it only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Order Date"]).drop_duplicates()
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    return df


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    return "Fall"


# ---------------------------------------------------------------------------
# UI helper functions (presentation only — no business logic here)
# ---------------------------------------------------------------------------
def style_fig(fig, height=420):
    """Apply a consistent, modern dark Plotly theme to any figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color="#E5E7EB", size=13),
        title=dict(font=dict(size=16, color="#F8FAFC")),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(size=12),
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        hoverlabel=dict(bgcolor="#1E293B", font_size=12.5, font_family=FONT_FAMILY, bordercolor="#3B82F6"),
        height=height,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor="rgba(255,255,255,0.12)")
    return fig


def chart_card(title_icon, title_text):
    """Open a styled card and render a section title inside it."""
    container = st.container(border=True)
    with container:
        st.markdown(f'<div class="section-title">{title_icon} {title_text}</div>', unsafe_allow_html=True)
    return container


def styled_table(dframe):
    """Return a pandas Styler with alternating rows and a highlighted header."""
    return (
        dframe.style
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#1E293B"),
                ("color", "#F8FAFC"),
                ("font-weight", "700"),
                ("border-bottom", "2px solid #3B82F6"),
                ("text-align", "left"),
                ("padding", "8px 10px"),
            ]},
            {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#111827")]},
            {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#0B1220")]},
            {"selector": "td", "props": [("padding", "7px 10px"), ("border-color", "#1F2937")]},
        ])
        .set_properties(**{"color": "#E5E7EB"})
    )


def kpi_card(icon, color_class, label, value, sub=""):
    return f"""
    <div class="kpi-card">
        <div class="kpi-top">
            <div class="kpi-icon {color_class}">{icon}</div>
        </div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = load_data()

# ---------------------------------------------------------------------------
# Sidebar — branding + navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="badge">📦</div>
        <div>
            <div class="brand-name">Demand Intelligence</div>
            <div class="brand-sub">ENTERPRISE ANALYTICS SUITE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_options = ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"]
    nav_icons = {
        "Sales Overview": "📊",
        "Forecast Explorer": "🔮",
        "Anomaly Report": "🚨",
        "Product Demand Segments": "🧩",
    }
    st.markdown('<div class="section-label">Navigate</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        nav_options,
        format_func=lambda x: f"{nav_icons[x]}   {x}",
        label_visibility="collapsed",
    )

    st.markdown(f"""
    <div class="sidebar-footer">
        Data range: {df['Order Date'].min().date()} → {df['Order Date'].max().date()}<br/>
        {len(df):,} records loaded
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="icon-badge">📦</div>
    <div>
        <h1>Sales Forecasting & Demand Intelligence System</h1>
        <p>Real-time performance, forecasting, anomaly detection & product segmentation</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAGE 1 — Sales Overview
# ---------------------------------------------------------------------------
if page == "Sales Overview":

    # ---- KPI row ----
    total_sales = df["Sales"].sum()
    total_orders = df["Order ID"].nunique() if "Order ID" in df.columns else len(df)
    avg_order_value = total_sales / total_orders if total_orders else 0
    total_profit = df["Profit"].sum() if "Profit" in df.columns else None

    profit_display = f"${total_profit:,.0f}" if total_profit is not None else "N/A"
    profit_sub = "Net profit across all orders" if total_profit is not None else "Profit column not found"

    st.markdown(f"""
    <div class="kpi-grid">
        {kpi_card("💰", "blue", "Total Sales", f"${total_sales:,.0f}", "All-time revenue")}
        {kpi_card("🧾", "cyan", "Total Orders", f"{total_orders:,}", "Unique orders placed")}
        {kpi_card("📊", "violet", "Avg Order Value", f"${avg_order_value:,.0f}", "Revenue per order")}
        {kpi_card("📈", "green", "Total Profit", profit_display, profit_sub)}
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown('<div class="section-title">📅 Total Sales by Year</div>', unsafe_allow_html=True)
            yearly = df.groupby("Year")["Sales"].sum().reset_index()
            fig = px.bar(yearly, x="Year", y="Sales", color_discrete_sequence=[PALETTE[0]])
            fig.update_traces(marker_line_width=0, hovertemplate="Year %{x}<br>Sales: $%{y:,.0f}<extra></extra>")
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(style_fig(fig), use_container_width=True, config={"displaylogo": False})

    with col2:
        with st.container(border=True):
            st.markdown('<div class="section-title">📈 Monthly Sales Trend</div>', unsafe_allow_html=True)
            monthly = df.set_index("Order Date").resample("MS")["Sales"].sum().reset_index()
            fig = px.line(monthly, x="Order Date", y="Sales", color_discrete_sequence=[PALETTE[1]])
            fig.update_traces(line_width=3, hovertemplate="%{x|%b %Y}<br>Sales: $%{y:,.0f}<extra></extra>")
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(style_fig(fig), use_container_width=True, config={"displaylogo": False})

    st.markdown('<div class="section-label">Breakdown</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="section-title">🌍 Sales by Region & Category</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            region_filter = st.multiselect(
                "🌍 Filter Region(s)", df["Region"].unique(), default=list(df["Region"].unique())
            )
        with c2:
            category_filter = st.multiselect(
                "🏷️ Filter Category(ies)", df["Category"].unique(), default=list(df["Category"].unique())
            )

        filtered = df[df["Region"].isin(region_filter) & df["Category"].isin(category_filter)]
        grouped = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
        fig = px.bar(
            grouped, x="Region", y="Sales", color="Category", barmode="group",
            color_discrete_sequence=PALETTE,
        )
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(style_fig(fig, height=440), use_container_width=True, config={"displaylogo": False})


# ---------------------------------------------------------------------------
# PAGE 2 — Forecast Explorer
# ---------------------------------------------------------------------------
elif page == "Forecast Explorer":

    with st.container(border=True):
        st.markdown('<div class="section-title">⚙️ Forecast Settings</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            dim_type = st.selectbox("📁 Select dimension type", ["Category", "Region"])
        with c2:
            dim_value = st.selectbox(f"🔎 Select {dim_type}", sorted(df[dim_type].unique()))
        with c3:
            horizon = st.slider("🗓️ Forecast horizon (months ahead)", 1, 3, 3)

    @st.cache_data
    def forecast_dimension(dim_type, dim_value, horizon):
        sub = df[df[dim_type] == dim_value]
        series = sub.set_index("Order Date").resample("MS")["Sales"].sum()

        frame = series.to_frame("Sales")
        frame["Lag1"] = frame["Sales"].shift(1)
        frame["Lag2"] = frame["Sales"].shift(2)
        frame["Lag3"] = frame["Sales"].shift(3)
        frame["RollingMean3"] = frame["Sales"].shift(1).rolling(3).mean()
        frame["Month"] = frame.index.month
        frame["Quarter"] = frame.index.quarter
        frame = frame.dropna()

        if len(frame) < 12:
            return None, None, None

        feats = ["Lag1", "Lag2", "Lag3", "RollingMean3", "Month", "Quarter"]

        # Hold out last 3 months to compute MAE/RMSE for display
        test_len = min(3, len(frame) // 4)
        train_frame, test_frame = frame.iloc[:-test_len], frame.iloc[-test_len:]

        model = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
        model.fit(train_frame[feats], train_frame["Sales"])
        test_pred = model.predict(test_frame[feats])
        mae = mean_absolute_error(test_frame["Sales"], test_pred)
        rmse = np.sqrt(mean_squared_error(test_frame["Sales"], test_pred))

        # Refit on full history, then forecast forward recursively
        model.fit(frame[feats], frame["Sales"])
        hist = series.copy()
        preds = []
        for _ in range(horizon):
            next_date = hist.index[-1] + pd.offsets.MonthBegin(1)
            lag1, lag2, lag3 = hist.iloc[-1], hist.iloc[-2], hist.iloc[-3]
            roll = hist.iloc[-3:].mean()
            row = pd.DataFrame([{
                "Lag1": lag1, "Lag2": lag2, "Lag3": lag3, "RollingMean3": roll,
                "Month": next_date.month, "Quarter": next_date.quarter,
            }])[feats]
            pred = model.predict(row)[0]
            preds.append((next_date, pred))
            hist.loc[next_date] = pred

        forecast_series = pd.Series({d: v for d, v in preds})
        return series, forecast_series, (mae, rmse)

    series, forecast_series, metrics = forecast_dimension(dim_type, dim_value, horizon)

    if series is None:
        st.warning("⚠️ Not enough history for this segment to build a reliable forecast.")
    else:
        with st.container(border=True):
            st.markdown(
                f'<div class="section-title">🔮 Forecast — {dim_value} ({dim_type})</div>',
                unsafe_allow_html=True,
            )
            plot_df = pd.concat([
                series.to_frame("Sales").assign(Type="Actual"),
                forecast_series.to_frame("Sales").assign(Type="Forecast"),
            ]).reset_index().rename(columns={"index": "Date"})

            fig = px.line(
                plot_df, x="Date", y="Sales", color="Type",
                color_discrete_map={"Actual": PALETTE[0], "Forecast": PALETTE[4]},
                line_dash="Type",
            )
            fig.update_traces(line_width=3)
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(style_fig(fig, height=440), use_container_width=True, config={"displaylogo": False})

        mae, rmse = metrics
        c1, c2 = st.columns(2)
        c1.metric("📏 MAE (on held-out test months)", f"{mae:,.0f}")
        c2.metric("📐 RMSE (on held-out test months)", f"{rmse:,.0f}")


# ---------------------------------------------------------------------------
# PAGE 3 — Anomaly Report
# ---------------------------------------------------------------------------
elif page == "Anomaly Report":

    weekly = df.set_index("Order Date").resample("W")["Sales"].sum().to_frame("WeeklySales")

    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly["iso_anomaly"] = iso.fit_predict(weekly[["WeeklySales"]])
    anomalies = weekly[weekly["iso_anomaly"] == -1]

    c1, c2 = st.columns(2)
    c1.metric("🚨 Anomalies Detected", f"{len(anomalies):,}")
    c2.metric("🗓️ Weeks Analyzed", f"{len(weekly):,}")

    with st.container(border=True):
        st.markdown('<div class="section-title">📉 Weekly Sales with Anomalies</div>', unsafe_allow_html=True)
        plot_df = weekly.reset_index()
        fig = px.line(plot_df, x="Order Date", y="WeeklySales", color_discrete_sequence=[PALETTE[0]])
        fig.update_traces(line_width=2.5, hovertemplate="%{x|%d %b %Y}<br>Sales: $%{y:,.0f}<extra></extra>")
        fig.add_scatter(
            x=anomalies.index, y=anomalies["WeeklySales"],
            mode="markers", marker=dict(color="#F87171", size=11, line=dict(width=1.5, color="#FFFFFF")),
            name="Anomaly",
        )
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(style_fig(fig, height=440), use_container_width=True, config={"displaylogo": False})

    with st.container(border=True):
        st.markdown('<div class="section-title">🚨 Detected Anomaly Dates</div>', unsafe_allow_html=True)
        anomaly_table = anomalies[["WeeklySales"]].reset_index().rename(
            columns={"Order Date": "Week", "WeeklySales": "Sales"}
        )
        st.dataframe(
            styled_table(anomaly_table).format({"Sales": "${:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# PAGE 4 — Product Demand Segments
# ---------------------------------------------------------------------------
elif page == "Product Demand Segments":

    subcat = df.groupby("Sub-Category").agg(
        TotalSales=("Sales", "sum"),
        AvgOrderValue=("Sales", "mean"),
    )
    sc_year = df.groupby(["Sub-Category", "Year"])["Sales"].sum().reset_index()

    def yoy_growth(g):
        g = g.sort_values("Year")
        if len(g) < 2 or g["Sales"].iloc[0] == 0:
            return np.nan
        return (g["Sales"].iloc[-1] - g["Sales"].iloc[0]) / g["Sales"].iloc[0] * 100

    growth_rate = sc_year.groupby("Sub-Category").apply(yoy_growth).rename("GrowthRate")
    sc_month = df.groupby(["Sub-Category", pd.Grouper(key="Order Date", freq="MS")])["Sales"].sum().reset_index()
    volatility = sc_month.groupby("Sub-Category")["Sales"].std().rename("Volatility")

    features = subcat.join(growth_rate).join(volatility).dropna()
    X = features[["TotalSales", "GrowthRate", "Volatility", "AvgOrderValue"]]
    X_scaled = StandardScaler().fit_transform(X)

    with st.container(border=True):
        st.markdown('<div class="section-title">🧩 Segmentation Controls</div>', unsafe_allow_html=True)
        k = st.slider("🔢 Number of clusters (k)", 2, 6, 4)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    features["Cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    features["PCA1"], features["PCA2"] = coords[:, 0], coords[:, 1]

    with st.container(border=True):
        st.markdown(
            '<div class="section-title">🧭 Product Sub-Category Clusters (PCA-reduced)</div>',
            unsafe_allow_html=True,
        )
        fig = px.scatter(
            features.reset_index(), x="PCA1", y="PCA2", color=features["Cluster"].astype(str),
            text="Sub-Category", color_discrete_sequence=PALETTE,
        )
        fig.update_traces(
            textposition="top center",
            marker=dict(size=13, line=dict(width=1.5, color="rgba(255,255,255,0.6)")),
        )
        fig.update_layout(legend_title_text="Cluster")
        st.plotly_chart(style_fig(fig, height=480), use_container_width=True, config={"displaylogo": False})

    with st.container(border=True):
        st.markdown('<div class="section-title">📋 Sub-Categories by Cluster</div>', unsafe_allow_html=True)
        table = (
            features[["Cluster", "TotalSales", "GrowthRate", "Volatility", "AvgOrderValue"]]
            .sort_values("Cluster")
        )
        st.dataframe(
            styled_table(table).format({
                "TotalSales": "${:,.0f}",
                "GrowthRate": "{:,.1f}%",
                "Volatility": "${:,.0f}",
                "AvgOrderValue": "${:,.0f}",
            }),
            use_container_width=True,
        )
