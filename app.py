"""
Equilym Technologies Pvt Ltd - Assessment 4: Automated Insight Generation
Streamlit Web Application & Interactive Analytics Dashboard
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from analytics_engine import HealthcareAnalyticsEngine, NUMERICAL_INDICATORS

st.set_page_config(
    page_title="Auto-Analytics Engine | Equilym Technologies",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive dashboard styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-med {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-low {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_default_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "healthcare_district_data.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        # Fallback inline dataset if running standalone
        data = """month,district,anc_coverage,institutional_delivery,immunization,high_risk_cases
2026-07,Ahmedabad,85,91,93,10
2026-08,Ahmedabad,69,90,92,13
2026-07,Surat,81,87,91,13
2026-08,Surat,83,89,92,12
2026-07,Vadodara,90,93,96,7
2026-08,Vadodara,91,94,97,6
2026-07,Rajkot,79,85,89,15
2026-08,Rajkot,80,86,90,14
2026-07,Mehsana,84,89,92,11
2026-08,Mehsana,42,88,91,28
2026-07,Bhavnagar,77,82,87,17
2026-08,Bhavnagar,79,84,89,15"""
        from io import StringIO
        return pd.read_csv(StringIO(data))


raw_df = load_default_data()

# -----------------------------------------------------------------------------
# Sidebar Configuration & Live Filters
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/medical-heart.png", width=64)
st.sidebar.title("Engine Controls")

st.sidebar.subheader("1. Live Filters")
all_districts = sorted(raw_df["district"].unique().tolist())
all_months = sorted(raw_df["month"].unique().tolist())

selected_districts = st.sidebar.multiselect(
    "Districts",
    options=all_districts,
    default=all_districts,
    help="Filter analytics to specific districts",
)

selected_months = st.sidebar.multiselect(
    "Months",
    options=all_months,
    default=all_months,
    help="Filter analysis time window",
)

selected_indicators = st.sidebar.multiselect(
    "Indicators",
    options=NUMERICAL_INDICATORS,
    default=NUMERICAL_INDICATORS,
    help="Filter specific health metrics",
)

st.sidebar.markdown("---")
st.sidebar.subheader("2. Configurable Thresholds")

trend_threshold = st.sidebar.slider(
    "Trend Significance Threshold (±%)",
    min_value=1.0,
    max_value=40.0,
    value=10.0,
    step=0.5,
    help="Minimum % change from previous month to trigger a trend insight",
)

outlier_method = st.sidebar.selectbox(
    "Outlier Detection Method",
    options=["IQR", "Z_SCORE"],
    index=0,
    help="IQR is robust for small samples; Z-score measures standard deviations from the mean",
)

if outlier_method == "IQR":
    iqr_multiplier = st.sidebar.slider(
        "IQR Multiplier",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="Standard Tukey outlier threshold is 1.5",
    )
    z_threshold = 2.5
else:
    z_threshold = st.sidebar.slider(
        "Z-Score Threshold (σ)",
        min_value=1.5,
        max_value=3.5,
        value=2.5,
        step=0.1,
        help="Number of standard deviations from the mean",
    )
    iqr_multiplier = 1.5

corr_threshold = st.sidebar.slider(
    "Correlation Threshold (|r|)",
    min_value=0.50,
    max_value=0.95,
    value=0.70,
    step=0.05,
    help="Minimum absolute Pearson correlation to flag indicator pairs",
)

# -----------------------------------------------------------------------------
# Initialize Engine with Filtered Data
# -----------------------------------------------------------------------------
filtered_df = raw_df[
    (raw_df["district"].isin(selected_districts)) & (raw_df["month"].isin(selected_months))
].copy()

if filtered_df.empty:
    st.error("No data available for the selected filters. Please expand your selections.")
    st.stop()

engine = HealthcareAnalyticsEngine(df=filtered_df)

# Header Section
st.markdown('<div class="main-header">🏥 Automated Healthcare Insight Generation Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Equilym Technologies Pvt Ltd — Assignment 4 Auto-Analytics System</div>',
    unsafe_allow_html=True,
)

# Compute Insights
insights_df = engine.generate_all_insights(
    trend_threshold_pct=trend_threshold,
    outlier_method=outlier_method,
    iqr_multiplier=iqr_multiplier,
    z_threshold=z_threshold,
    correlation_threshold=corr_threshold,
)

# Filter insights by indicator selection if applicable
if selected_indicators:
    # Keep insights where indicator matches or is a combo containing selected
    def matches_indicator(ind_str):
        for sel in selected_indicators:
            if sel in ind_str:
                return True
        return False
    insights_df = insights_df[insights_df["indicator"].apply(matches_indicator)].reset_index(drop=True)

# -----------------------------------------------------------------------------
# KPI Overview Cards
# -----------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
total_insights = len(insights_df)
high_sev = len(insights_df[insights_df["severity"] == "High"])
med_sev = len(insights_df[insights_df["severity"] == "Medium"])
low_sev = len(insights_df[insights_df["severity"] == "Low"])

c1.metric("Total Insights", total_insights)
c2.metric("High Severity 🔴", high_sev)
c3.metric("Medium Severity 🟡", med_sev)
c4.metric("Low Severity 🔵", low_sev)
c5.metric("Districts Analyzed", len(selected_districts))

# -----------------------------------------------------------------------------
# Tabs Layout
# -----------------------------------------------------------------------------
tab_insights, tab_viz, tab_corr, tab_validation = st.tabs([
    "💡 Automated Insights",
    "📈 Interactive Visualizations",
    "🔗 Correlation & Fragility",
    "📋 Validation & Raw Data",
])

with tab_insights:
    st.subheader(f"Generated Dynamic Insights ({len(insights_df)} Found)")
    st.caption("Insights are strictly data-driven with no hardcoded district narratives. Threshold breaches and statistical anomalies are flagged automatically.")

    if not insights_df.empty:
        # Filters for insights table
        f_col1, f_col2 = st.columns([1, 1])
        with f_col1:
            type_filter = st.multiselect(
                "Filter Insight Type",
                options=insights_df["type"].unique().tolist(),
                default=insights_df["type"].unique().tolist(),
            )
        with f_col2:
            sev_filter = st.multiselect(
                "Filter Severity",
                options=["High", "Medium", "Low"],
                default=["High", "Medium", "Low"],
            )

        view_insights = insights_df[
            (insights_df["type"].isin(type_filter)) & (insights_df["severity"].isin(sev_filter))
        ]

        # Display formatted insights
        for _, row in view_insights.iterrows():
            sev_class = "badge-high" if row["severity"] == "High" else ("badge-med" if row["severity"] == "Medium" else "badge-low")
            st.markdown(
                f"""
                <div class="card">
                    <span class="{sev_class}">{row['severity']}</span> &nbsp;
                    <strong>[{row['insight_id']}] {row['type'].upper()}</strong> &nbsp;|&nbsp;
                    <em>Entity: {row['entity']} ({row['period']})</em> &nbsp;|&nbsp;
                    Metric: <code>{row['indicator']}</code>
                    <p style="margin-top: 0.5rem; margin-bottom: 0.2rem; font-size: 1.05rem;">{row['explanation']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Insights Table (Section 4.1 Schema)")
        st.dataframe(view_insights, use_container_width=True)

        # Download Button
        csv_data = view_insights.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Insights CSV",
            data=csv_data,
            file_name="insights.csv",
            mime="text/csv",
        )
    else:
        st.info("No insights breached current thresholds. Try adjusting the sensitivity sliders in the sidebar.")

with tab_viz:
    st.subheader("Visual Analytics (Part F)")
    v1, v2 = st.columns([1, 1])

    with v1:
        st.markdown("**1. Severity Count Distribution**")
        if not insights_df.empty:
            sev_counts = insights_df["severity"].value_counts().reset_index()
            sev_counts.columns = ["Severity", "Count"]
            color_map = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#3B82F6"}
            fig_bar = px.bar(
                sev_counts,
                x="Severity",
                y="Count",
                color="Severity",
                color_discrete_map=color_map,
                text="Count",
                title="Insight Volume by Severity Level",
            )
            fig_bar.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.write("No insights to plot.")

    with v2:
        st.markdown("**2. Indicator Distribution & Outliers (Box Plot)**")
        metric_choice = st.selectbox("Select Indicator for Distribution", options=selected_indicators)
        fig_box = px.box(
            filtered_df,
            x="month",
            y=metric_choice,
            points="all",
            hover_data=["district"],
            title=f"Boxplot of {metric_choice} by Month",
            color="month",
        )
        fig_box.update_layout(height=350)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("**3. Per-District Indicator Trend Lines**")
    line_metric = st.selectbox("Select Indicator to Trace Over Time", options=selected_indicators, key="line_select")
    fig_line = px.line(
        filtered_df,
        x="month",
        y=line_metric,
        color="district",
        markers=True,
        title=f"Per-District Trajectory: {line_metric}",
    )
    fig_line.update_layout(height=400)
    st.plotly_chart(fig_line, use_container_width=True)

with tab_corr:
    st.subheader("Pearson Correlation Matrix & Anomaly Detection (Part D)")
    
    corr_matrix = engine.compute_correlation_matrix()
    
    col_corr1, col_corr2 = st.columns([1.2, 1])
    
    with col_corr1:
        st.markdown("**Correlation Heatmap**")
        fig_heat = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pearson Correlation Heatmap (df.corr())",
        )
        fig_heat.update_layout(height=420)
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with col_corr2:
        st.markdown(f"**Flagged High Correlations (|r| ≥ {corr_threshold:.2f})**")
        corr_flagged, fragility_note = engine.detect_high_correlations(threshold_r=corr_threshold)
        if not corr_flagged.empty:
            st.dataframe(corr_flagged[["indicator_pair", "pearson_r", "relationship"]], use_container_width=True)
        else:
            st.info("No indicator pairs met the |r| threshold.")

        # Crucial Note 2 Callout
        st.warning(
            f"""
            ⚠️ **Rubric Note 2 — Fragility of Correlation on Small Sample:**
            {fragility_note}
            """
        )

    # Download Correlation Matrix
    corr_csv = corr_matrix.to_csv().encode("utf-8")
    st.download_button(
        label="📥 Download Correlation Matrix CSV",
        data=corr_csv,
        file_name="correlation_matrix.csv",
        mime="text/csv",
    )

with tab_validation:
    st.subheader("Data Loading, Validation & Missing-Value Diagnostics (Part A)")
    
    val_rep = engine.get_validation_report()
    
    r1, r2, r3 = st.columns(3)
    r1.metric("Rows Loaded", val_rep["total_rows"])
    r2.metric("Columns", val_rep["total_columns"])
    r3.metric("Missing Values", sum(val_rep["missing_values_per_column"].values()))
    
    st.markdown("**Column Missing-Value Count:**")
    st.json(val_rep["missing_values_per_column"])
    
    st.markdown("**Active Filtered Dataset:**")
    st.dataframe(filtered_df, use_container_width=True)
    
    st.markdown("**Dataset Summary Statistics (`df.describe()`):**")
    st.dataframe(filtered_df.describe(), use_container_width=True)
