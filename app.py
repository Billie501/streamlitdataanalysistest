# analytics app 27/08/2025
# Advanced Incident Analytics & Safety Intelligence
# -------------------------------------------------
# Notes:
# - Time analysis now pulls hour from `incident_time` (not from the zeroed incident_date time)
# - `incident_time` is parsed as a datetime to safely extract hours
# - Optional `incident_datetime` combines date + time when both exist (non-breaking helper)
# - Everything else mirrors your existing logic and visuals

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# -----------------------------
# 0) Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Advanced Incident Analytics", layout="wide")

st.title("📊 Advanced Incident Analytics & Safety Intelligence")
st.markdown("*Comprehensive insights to improve workplace safety and prevent future incidents*")

# ============================================================
# 1) Helpers (parsing, safety, small utilities)
# ============================================================

def _safe_to_datetime(series: pd.Series) -> pd.Series:
    """
    Safely convert a pandas Series to datetime (coerce errors).
    """
    return pd.to_datetime(series, errors='coerce')

def _ensure_hour_from_incident_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure an 'hour' column exists and is derived from 'incident_time', NOT 'incident_date'.
    'incident_time' is parsed as a datetime to preserve .dt.hour extraction.
    """
    if 'incident_time' in df.columns:
        # Parse as datetime (not .dt.time), then extract hour
        parsed_time = pd.to_datetime(df['incident_time'], errors='coerce')
        df['hour'] = parsed_time.dt.hour
    else:
        # If no incident_time, ensure hour is absent to avoid misleading visuals
        if 'hour' in df.columns:
            df.drop(columns=['hour'], inplace=True)
    return df

def _add_date_parts_from_incident_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive day_of_week, month, year from incident_date.
    """
    if 'incident_date' in df.columns:
        dt = _safe_to_datetime(df['incident_date'])
        df['day_of_week'] = dt.dt.day_name()
        df['month'] = dt.dt.month
        df['year'] = dt.dt.year
    return df

def _make_incident_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional: combine incident_date + incident_time into a single incident_datetime
    to enable uniform temporal filtering. Non-breaking: only adds a column.
    """
    if 'incident_date' in df.columns and 'incident_time' in df.columns:
        date_part = _safe_to_datetime(df['incident_date'])
        time_part = pd.to_datetime(df['incident_time'], errors='coerce')
        # When both exist: replace date's time with time_part's time
        # time_part.dt.time gives python time; combine using dt.normalize + hours/min/sec
        # If time_part is NaT, fall back to incident_date at 00:00:00
        base = date_part.dt.normalize()
        # Extract hours/minutes/seconds from time_part safely
        hours = time_part.dt.hour.fillna(0).astype(int)
        minutes = time_part.dt.minute.fillna(0).astype(int)
        seconds = time_part.dt.second.fillna(0).astype(int)
        df['incident_datetime'] = base + pd.to_timedelta(hours, unit='h') \
                                        + pd.to_timedelta(minutes, unit='m') \
                                        + pd.to_timedelta(seconds, unit='s')
    return df

def _count_missing(df: pd.DataFrame) -> pd.Series:
    """
    Return missing counts per column.
    """
    return df.isnull().sum()

def _binary_injury(series: pd.Series) -> pd.Series:
    """
    Convert mixed-type injury column into binary 0/1.
    """
    return series.apply(
        lambda x: 1 if pd.notna(x) and str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured'] else 0
    )

def _nice_day_order():
    return ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# ============================================================
# 2) File Upload
# ============================================================
uploaded_file = st.file_uploader("Upload your incident CSV", type=["csv"])

if uploaded_file:
    # ============================================================
    # 3) Read and Clean
    # ============================================================
    df = pd.read_csv(uploaded_file)

    # --- Enhanced Data Cleaning ---
    # 3a) Handle various date formats (revert to working version)
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for col in date_columns:
        df[col] = _safe_to_datetime(df[col])  # coerces on error

    # 3b) Standardize time columns
    # IMPORTANT CHANGE: parse 'time' columns as datetime (NOT .dt.time), so we can use dt.hour later.
    time_columns = [col for col in df.columns if 'time' in col.lower()]
    for col in time_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')  # keep as datetime, not .dt.time

    # 3c) Extract date parts (from incident_date) and hour (from incident_time)
    df = _add_date_parts_from_incident_date(df)
    df = _ensure_hour_from_incident_time(df)

    # 3d) Optional unified timestamp (does not change your existing logic)
    df = _make_incident_datetime(df)

    # 3e) Clean text columns
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # ============================================================
    # 4) Data Overview
    # ============================================================
    st.subheader("🔎 Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df.head())
    with col2:
        st.write("**Data Quality Summary:**")
        missing_data = _count_missing(df)
        st.write(f"Total Records: {len(df)}")
        st.write(f"Columns: {len(df.columns)}")
        if missing_data.sum() > 0:
            st.write("Missing Values:")
            for col, missing in missing_data[missing_data > 0].items():
                st.write(f"  • {col}: {missing} ({missing/len(df)*100:.1f}%)")

    # ============================================================
    # 5) KPI Dashboard
    # ============================================================
    st.subheader("📈 Executive Dashboard")

    # Calculate advanced metrics
    total_incidents = len(df)
    current_month = datetime.now().month
    current_year = datetime.now().year

    # Filter for current period if date exists
    if 'incident_date' in df.columns:
        df_current_month = df[(df['month'] == current_month) & (df['year'] == current_year)]
        df_last_month = df[(df['month'] == current_month - 1) & (df['year'] == current_year)]

        # Calculate trends
        current_month_incidents = len(df_current_month)
        last_month_incidents = len(df_last_month)

        if last_month_incidents > 0:
            incident_trend = ((current_month_incidents - last_month_incidents) / last_month_incidents) * 100
        else:
            incident_trend = 0
    else:
        incident_trend = 0
        current_month_incidents = 0

    # Injury rate calculation - specifically for 'was_injured' column
    injury_rate = 0
    if 'was_injured' in df.columns:
        try:
            if df['was_injured'].dtype == 'bool':
                injury_rate = (df['was_injured'].sum() / len(df)) * 100 if len(df) > 0 else 0
            elif df['was_injured'].dtype == 'object':
                injured_count = df['was_injured'].dropna().apply(
                    lambda x: str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured']
                ).sum()
                injury_rate = (injured_count / len(df)) * 100 if len(df) > 0 else 0
            else:
                injury_rate = (df['was_injured'].fillna(0).astype(float).gt(0).sum() / len(df)) * 100 if len(df) > 0 else 0
        except Exception:
            injury_rate = 0

    # Display KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Incidents", total_incidents)
    col2.metric("This Month", current_month_incidents, f"{incident_trend:+.1f}%")
    col3.metric("Injury Rate", f"{injury_rate:.1f}%")
    col4.metric("Departments", df['department'].nunique() if 'department' in df.columns else 0)
    col5.metric("Locations", df['location'].nunique() if 'location' in df.columns else 0)

    # ============================================================
    # 6) Predictive Analytics (Moving Average)
    # ============================================================
    st.subheader("🔮 Predictive Analytics & Forecasting")

    if 'incident_date' in df.columns and len(df) > 10:
        col1, col2 = st.columns(2)

        with col1:
            st.write("**📊 Incident Trend Forecast**")

            # Create monthly aggregation for forecasting
            df_temp = df.copy()
            df_temp['year'] = df_temp['incident_date'].dt.year
            df_temp['month'] = df_temp['incident_date'].dt.month

            df_monthly = df_temp.groupby(['year', 'month']).size().reset_index(name='incident_count')
            # Create date column properly
            df_monthly['date'] = pd.to_datetime(df_monthly[['year', 'month']].assign(day=1))
            df_monthly = df_monthly.sort_values('date')

            if len(df_monthly) >= 3:
                # Simple moving average forecast
                window = min(3, len(df_monthly))
                df_monthly['forecast'] = df_monthly['incident_count'].rolling(window=window).mean().shift(1)

                # Create next 3 months prediction
                last_avg = df_monthly['incident_count'].tail(window).mean()
                future_dates = pd.date_range(
                    start=df_monthly['date'].max() + pd.DateOffset(months=1),
                    periods=3,
                    freq='MS'  # MS = month start
                )

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_monthly['date'], y=df_monthly['incident_count'],
                    mode='lines+markers', name='Actual Incidents'
                ))
                fig.add_trace(go.Scatter(
                    x=future_dates, y=[last_avg] * 3,
                    mode='lines+markers', name='Predicted',
                    line=dict(dash='dash')
                ))
                fig.update_layout(
                    title="3-Month Incident Forecast",
                    xaxis_title="Date", yaxis_title="Incidents"
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("**🎯 Risk Indicators**")

            # Calculate risk scores
            risk_factors = []

            # Department risk
            if 'department' in df.columns:
                dept_incidents = df['department'].value_counts()
                high_risk_dept = dept_incidents.index[0] if len(dept_incidents) > 0 else "Unknown"
                risk_factors.append(
                    f"🔴 High Risk Department: {high_risk_dept} ({dept_incidents.iloc[0]} incidents)"
                )

            # Time-based risk (NOW BASED ON incident_time → hour)
            if 'hour' in df.columns:
                hour_risk = df['hour'].value_counts()
                if len(hour_risk) > 0:
                    peak_hour = int(hour_risk.index[0])
                    risk_factors.append(
                        f"⏰ Peak Risk Time: {peak_hour}:00 ({hour_risk.iloc[0]} incidents)"
                    )

            # Day of week risk
            if 'day_of_week' in df.columns:
                day_risk = df['day_of_week'].value_counts()
                if len(day_risk) > 0:
                    risky_day = day_risk.index[0]
                    risk_factors.append(
                        f"📅 Highest Risk Day: {risky_day} ({day_risk.iloc[0]} incidents)"
                    )

            # Injury severity prediction
            if 'was_injured' in df.columns and 'department' in df.columns:
                try:
                    if df['was_injured'].dtype == 'bool':
                        injury_by_dept = df.groupby('department')['was_injured'] \
                                           .mean().sort_values(ascending=False)
                    else:
                        # For non-boolean, create a binary injury indicator
                        df['injury_binary'] = _binary_injury(df['was_injured'])
                        injury_by_dept = df.groupby('department')['injury_binary'] \
                                           .mean().sort_values(ascending=False)

                    if len(injury_by_dept) > 0 and injury_by_dept.iloc[0] > 0:
                        high_injury_dept = injury_by_dept.index[0]
                        injury_pct = injury_by_dept.iloc[0] * 100
                        risk_factors.append(
                            f"🏥 Injury Hotspot: {high_injury_dept} ({injury_pct:.1f}% injury rate)"
                        )
                except Exception:
                    pass

            for factor in risk_factors:
                st.write(factor)

    # ============================================================
    # 7) Advanced Time Analysis (uses hour from incident_time)
    # ============================================================
    if 'incident_date' in df.columns:
        st.subheader("⏰ Time Pattern Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Heatmap by hour and day (hour from incident_time)
            if 'hour' in df.columns and 'day_of_week' in df.columns:
                heatmap_data = df.groupby(['day_of_week', 'hour']) \
                                 .size().reset_index(name='incidents')
                heatmap_pivot = heatmap_data.pivot(
                    index='day_of_week', columns='hour', values='incidents'
                ).fillna(0)

                # Order days correctly
                day_order = _nice_day_order()
                heatmap_pivot = heatmap_pivot.reindex(day_order)

                fig = px.imshow(
                    heatmap_pivot,
                    title="Incident Heatmap: Day vs Hour",
                    labels=dict(x="Hour of Day", y="Day of Week", color="Incidents"),
                    color_continuous_scale="Reds"
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Seasonal trends (from month)
            if 'month' in df.columns:
                monthly_incidents = df.groupby('month').size().reset_index(name='incidents')
                monthly_incidents['month_name'] = monthly_incidents['month'].apply(
                    lambda x: pd.to_datetime(str(x), format='%m').strftime('%B')
                )

                fig = px.bar(
                    monthly_incidents, x='month_name', y='incidents',
                    title="Seasonal Incident Patterns"
                )
                st.plotly_chart(fig, use_container_width=True)

        with col3:
            # Weekly patterns (from day_of_week)
            if 'day_of_week' in df.columns:
                daily_incidents = df['day_of_week'].value_counts().reindex(
                    _nice_day_order()
                ).reset_index()
                daily_incidents.columns = ['day', 'incidents']

                fig = px.line(
                    daily_incidents, x='day', y='incidents',
                    markers=True, title="Weekly Incident Pattern"
                )
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # 8) Department & Location Insights
    # ============================================================
    st.subheader("🏢 Department & Location Intelligence")

    col1, col2 = st.columns(2)

    with col1:
        if 'department' in df.columns:
            # Create a safe aggregation function
            agg_dict = {df.columns[0]: 'count'}

            # Add injury rate if injury column exists
            if 'was_injured' in df.columns:
                if df['was_injured'].dtype == 'bool':
                    agg_dict['was_injured'] = 'mean'
                else:
                    # Create binary injury indicator for aggregation
                    df['injury_binary'] = _binary_injury(df['was_injured'])
                    agg_dict['injury_binary'] = 'mean'

            dept_analysis = df.groupby('department').agg(agg_dict).reset_index()

            if 'was_injured' in df.columns or 'injury_binary' in dept_analysis.columns:
                # Normalize column names for charting
                dept_analysis.columns = ['department', 'incident_count', 'injury_rate']
                dept_analysis['injury_rate'] *= 100

                # Create bubble chart
                fig = px.scatter(
                    dept_analysis, x='incident_count', y='injury_rate',
                    size='incident_count', hover_name='department',
                    title="Department Risk Matrix",
                    labels={'incident_count': 'Total Incidents',
                            'injury_rate': 'Injury Rate (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to simple bar chart
                fig = px.bar(
                    dept_analysis, x='department', y=dept_analysis.columns[1],
                    title="Incidents by Department"
                )
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'location' in df.columns:
            location_incidents = df['location'].value_counts().head(10).reset_index()
            location_incidents.columns = ['location', 'incidents']

            fig = px.bar(
                location_incidents, x='incidents', y='location',
                orientation='h', title="Top 10 Incident Locations"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # 9) Incident Categorization & Analysis
    # ============================================================
    if 'label' in df.columns or 'category' in df.columns:
        st.subheader("🏷️ Incident Category Analysis")

        category_col = 'label' if 'label' in df.columns else 'category'

        col1, col2 = st.columns(2)

        with col1:
            category_counts = df[category_col].value_counts().reset_index()
            category_counts.columns = ['category', 'count']

            fig = px.pie(
                category_counts, names='category', values='count',
                title="Incident Distribution by Category"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Category trends over time if date is available
            if 'incident_date' in df.columns:
                category_trends = df.groupby(
                    [df['incident_date'].dt.date, category_col]
                ).size().reset_index(name='count')

                # Rename for consistency with px.line usage
                category_trends.rename(columns={'incident_date': 'incident_date'}, inplace=True)

                fig = px.line(
                    category_trends, x='incident_date', y='count',
                    color=category_col, title="Category Trends Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # 10) Enhanced Text Analysis
    # ============================================================
    if 'incident_description' in df.columns:
        st.subheader("📝 Advanced Text Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**☁️ Incident Description Word Cloud**")
            text = " ".join(str(desc) for desc in df['incident_description'].dropna())
            if text.strip():
                # Remove common stop words
                stop_words = {
                    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'a', 'an'
                }
                words = text.lower().split()
                filtered_text = " ".join([
                    word for word in words if word not in stop_words and len(word) > 2
                ])

                wordcloud = WordCloud(width=400, height=300, background_color="white") \
                            .generate(filtered_text)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("No text available for analysis.")

        with col2:
            st.write("**🔍 Key Terms Frequency**")
            if text.strip():
                words = text.lower().split()
                word_freq = pd.Series(words).value_counts().head(10)
                word_freq = word_freq[word_freq.index.str.len() > 3]  # Filter short words

                fig = px.bar(
                    x=word_freq.values, y=word_freq.index,
                    orientation='h', title="Most Common Terms"
                )
                st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # 11) AI-Driven Recommendations
    # ============================================================
    st.subheader("💡 AI-Driven Recommendations")

    recommendations = []

    # Time-based recommendations (from incident_time → hour)
    if 'hour' in df.columns:
        peak_hours = df['hour'].value_counts().head(2)
        if len(peak_hours) > 0:
            top_hour = int(peak_hours.index[0])
            recommendations.append(
                f"🕐 **Peak Risk Hours**: Increase safety supervision during "
                f"{top_hour}:00-{(top_hour + 1) % 24}:00 (highest incident time)"
            )

    # Department recommendations
    if 'department' in df.columns:
        high_risk_depts = df['department'].value_counts().head(2)
        if len(high_risk_depts) > 0:
            recommendations.append(
                f"🏢 **Focus Area**: Prioritize safety training in "
                f"{high_risk_depts.index[0]} department ({high_risk_depts.iloc[0]} incidents)"
            )

    # Injury prevention
    if 'was_injured' in df.columns and injury_rate > 0:
        if injury_rate > 10:
            recommendations.append(
                f"🏥 **Critical**: {injury_rate:.1f}% injury rate requires immediate safety protocol review"
            )

        # Department-specific injury recommendations
        if 'department' in df.columns:
            try:
                if df['was_injured'].dtype == 'bool':
                    dept_injury_rates = df.groupby('department')['was_injured'] \
                                          .agg(['sum', 'count', 'mean'])
                else:
                    # Create binary injury indicator
                    df['injury_binary'] = _binary_injury(df['was_injured'])
                    dept_injury_rates = df.groupby('department')['injury_binary'] \
                                          .agg(['sum', 'count', 'mean'])

                # Only departments with 3+ incidents
                dept_injury_rates = dept_injury_rates[dept_injury_rates['count'] >= 3]
                if len(dept_injury_rates) > 0 and dept_injury_rates['mean'].max() > 0:
                    worst_dept = dept_injury_rates['mean'].idxmax()
                    worst_rate = dept_injury_rates.loc[worst_dept, 'mean'] * 100
                    recommendations.append(
                        f"🎯 **Targeted Intervention**: {worst_dept} has {worst_rate:.1f}% injury rate "
                        f"- implement enhanced safety measures"
                    )
            except Exception:
                pass

    # Trend-based recommendations (still uses incident_date for date-window logic)
    if 'incident_date' in df.columns and len(df) > 30:
        recent_30d = df[df['incident_date'] >= (datetime.now() - timedelta(days=30))]
        if len(recent_30d) > len(df) * 0.3:  # If 30% of incidents in last 30 days
            recommendations.append(
                "📈 **Trend Alert**: Recent surge in incidents detected - conduct immediate safety audit"
            )

    # Display recommendations
    if recommendations:
        for i, rec in enumerate(recommendations[:5], 1):  # Limit to top 5
            st.write(f"{i}. {rec}")
    else:
        st.write("📊 **Data Analysis**: Upload more comprehensive data for personalized recommendations")

    # ============================================================
    # 12) Export (placeholders)
    # ============================================================
    st.subheader("📤 Export Analytics")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Generate Report"):
            st.success("Analytics report generated! (Feature in development)")

    with col2:
        if st.button("📧 Email Insights"):
            st.success("Insights email sent! (Feature in development)")

    with col3:
        if st.button("📅 Schedule Reports"):
            st.success("Report scheduling configured! (Feature in development)")

else:
    # ============================================================
    # Empty State
    # ============================================================
    st.info("👆 Upload your incident CSV file to unlock powerful safety analytics and predictive insights")
    st.write("**Expected columns for optimal analysis:**")
    st.write("• Reporter Name, Person Involved, Incident Date & Time")
    st.write("• Department & Location, Incident Description")
    st.write("• Label/Category, Injury Information (was_injured: boolean)")
