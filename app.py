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

# Prophet forecasting
try:
    from prophet import Prophet
    prophet_installed = True
except ImportError:
    prophet_installed = False

st.set_page_config(page_title="Advanced Incident Analytics", layout="wide")

st.title("📊 Advanced Incident Analytics & Safety Intelligence")
st.markdown("*Comprehensive insights to improve workplace safety and prevent future incidents*")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload your incident CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # --- Enhanced Data Cleaning ---
    # Handle various date formats
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Standardize time columns
    time_columns = [col for col in df.columns if 'time' in col.lower()]
    for col in time_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.time
    
    # Extract time-based features
    if 'incident_date' in df.columns:
        df['day_of_week'] = pd.to_datetime(df['incident_date']).dt.day_name()
        df['month'] = pd.to_datetime(df['incident_date']).dt.month
        df['year'] = pd.to_datetime(df['incident_date']).dt.year

    if 'incident_time' in df.columns:
        df['hour'] = pd.to_datetime(df['incident_time'], errors='coerce').dt.hour
    
    # Clean text columns
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    st.subheader("🔎 Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df.head())
    with col2:
        st.write("**Data Quality Summary:**")
        missing_data = df.isnull().sum()
        st.write(f"Total Records: {len(df)}")
        st.write(f"Columns: {len(df.columns)}")
        if missing_data.sum() > 0:
            st.write("Missing Values:")
            for col, missing in missing_data[missing_data > 0].items():
                st.write(f"  • {col}: {missing} ({missing/len(df)*100:.1f}%)")

    # --- Enhanced KPI Dashboard ---
    st.subheader("📈 Executive Dashboard")
    
    total_incidents = len(df)
    current_month = datetime.now().month
    current_year = datetime.now().year
    
    if 'incident_date' in df.columns:
        df_current_month = df[(df['month'] == current_month) & (df['year'] == current_year)]
        df_last_month = df[(df['month'] == current_month-1) & (df['year'] == current_year)]
        
        current_month_incidents = len(df_current_month)
        last_month_incidents = len(df_last_month)
        
        if last_month_incidents > 0:
            incident_trend = ((current_month_incidents - last_month_incidents) / last_month_incidents) * 100
        else:
            incident_trend = 0
    else:
        incident_trend, current_month_incidents = 0, 0
    
    # Injury rate
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
        except:
            injury_rate = 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Incidents", total_incidents)
    col2.metric("This Month", current_month_incidents, f"{incident_trend:+.1f}%")
    col3.metric("Injury Rate", f"{injury_rate:.1f}%")
    col4.metric("Departments", df['department'].nunique() if 'department' in df.columns else 0)
    col5.metric("Locations", df['location'].nunique() if 'location' in df.columns else 0)

    # --- PREDICTIVE ANALYTICS ---
    st.subheader("🔮 Predictive Analytics & Forecasting")
    
    if 'incident_date' in df.columns and len(df) > 10:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Incident Trend Forecast**")
            df_temp = df.copy()
            df_temp['year'] = df_temp['incident_date'].dt.year
            df_temp['month'] = df_temp['incident_date'].dt.month
            df_monthly = df_temp.groupby(['year', 'month']).size().reset_index(name='incident_count')
            df_monthly['date'] = pd.to_datetime(df_monthly[['year', 'month']].assign(day=1))
            df_monthly = df_monthly.sort_values('date')
            
            if len(df_monthly) >= 3:
                if prophet_installed:
                    # Prophet Forecast
                    prophet_df = df_monthly[['date', 'incident_count']].rename(columns={'date': 'ds', 'incident_count': 'y'})
                    model = Prophet()
                    model.fit(prophet_df)
                    future = model.make_future_dataframe(periods=6, freq='MS')
                    forecast = model.predict(future)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'],
                                             mode='lines+markers', name='Actual'))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                                             mode='lines', name='Forecast'))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                                             mode='lines', line=dict(width=0),
                                             name='Upper CI', showlegend=False))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                                             mode='lines', fill='tonexty', line=dict(width=0),
                                             name='Lower CI', showlegend=False))
                    fig.update_layout(title="6-Month Prophet Forecast",
                                      xaxis_title="Date", yaxis_title="Incidents")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback: Moving average
                    window = min(3, len(df_monthly))
                    df_monthly['forecast'] = df_monthly['incident_count'].rolling(window=window).mean().shift(1)
                    last_avg = df_monthly['incident_count'].tail(window).mean()
                    future_dates = pd.date_range(start=df_monthly['date'].max() + pd.DateOffset(months=1), 
                                                 periods=3, freq='MS')
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_monthly['date'], y=df_monthly['incident_count'], 
                                             mode='lines+markers', name='Actual'))
                    fig.add_trace(go.Scatter(x=future_dates, y=[last_avg]*3, 
                                             mode='lines+markers', name='Predicted',
                                             line=dict(dash='dash')))
                    fig.update_layout(title="3-Month Incident Forecast (Moving Average)",
                                      xaxis_title="Date", yaxis_title="Incidents")
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**🎯 Risk Indicators**")
            risk_factors = []
            if 'department' in df.columns:
                dept_incidents = df['department'].value_counts()
                if len(dept_incidents) > 0:
                    risk_factors.append(f"🔴 High Risk Department: {dept_incidents.index[0]} ({dept_incidents.iloc[0]} incidents)")
            if 'hour' in df.columns:
                hour_risk = df['hour'].value_counts()
                if len(hour_risk) > 0:
                    peak_hour = hour_risk.index[0]
                    risk_factors.append(f"⏰ Peak Risk Time: {peak_hour}:00 ({hour_risk.iloc[0]} incidents)")
            if 'day_of_week' in df.columns:
                day_risk = df['day_of_week'].value_counts()
                if len(day_risk) > 0:
                    risky_day = day_risk.index[0]
                    risk_factors.append(f"📅 Highest Risk Day: {risky_day} ({day_risk.iloc[0]} incidents)")
            if 'was_injured' in df.columns and 'department' in df.columns:
                try:
                    df['injury_binary'] = df['was_injured'].apply(
                        lambda x: 1 if pd.notna(x) and str(x).strip().lower() in ['yes','true','1','y','injured'] else 0
                    )
                    injury_by_dept = df.groupby('department')['injury_binary'].mean().sort_values(ascending=False)
                    if len(injury_by_dept) > 0 and injury_by_dept.iloc[0] > 0:
                        high_injury_dept = injury_by_dept.index[0]
                        injury_pct = injury_by_dept.iloc[0] * 100
                        risk_factors.append(f"🏥 Injury Hotspot: {high_injury_dept} ({injury_pct:.1f}% injury rate)")
                except:
                    pass
            for factor in risk_factors:
                st.write(factor)

    # (rest of your app unchanged: time analysis, department/location insights, text analytics, recommendations, export)
else:
    st.info("👆 Upload your incident CSV file to unlock powerful safety analytics and predictive insights")
    st.write("**Expected columns for optimal analysis:**")
    st.write("• Reporter Name, Person Involved, Incident Date & Time")
    st.write("• Department & Location, Incident Description")
    st.write("• Label/Category, Injury Information (was_injured: boolean)")
