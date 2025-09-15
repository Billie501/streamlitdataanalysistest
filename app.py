# analytics app 27/08/2025 - Department-Level Enhanced Version

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Advanced Incident Analytics", layout="wide")

st.title("📊 Advanced Incident Analytics & Safety Intelligence")
st.markdown("*Comprehensive insights to improve workplace safety and prevent future incidents*")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload your incident CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # --- Enhanced Data Cleaning ---
    # Dates
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Times
    time_columns = [col for col in df.columns if 'time' in col.lower()]
    valid_time_column = None
    for col in time_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.time
        if col in df.columns:
            time_values = df[col].dropna()
            if len(time_values) > 0:
                non_midnight = sum(1 for t in time_values if t != pd.Timestamp('00:00:00').time())
                if non_midnight > len(time_values) * 0.1:
                    valid_time_column = col
                    break

    # Extract hour
    if 'incident_date' in df.columns:
        incident_datetime = pd.to_datetime(df['incident_date'])
        if valid_time_column and valid_time_column in df.columns:
            df['hour'] = df[valid_time_column].apply(lambda x: x.hour if pd.notna(x) else incident_datetime.dt.hour)
        else:
            df['hour'] = incident_datetime.dt.hour
        df['day_of_week'] = incident_datetime.dt.day_name()
        df['month'] = incident_datetime.dt.month
        df['year'] = incident_datetime.dt.year

    # Text cleaning
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()

    # --- Department Filter ---
    if 'department' in df.columns:
        departments = df['department'].dropna().unique()
        selected_dept = st.sidebar.selectbox("Select Department for Analytics", 
                                             options=["All Departments"] + list(departments))
        if selected_dept != "All Departments":
            df_dept = df[df['department'] == selected_dept].copy()
        else:
            df_dept = df.copy()
    else:
        df_dept = df.copy()

    # --- KPI Dashboard ---
    st.subheader("📈 Executive Dashboard")
    total_incidents = len(df_dept)
    current_month = datetime.now().month
    current_year = datetime.now().year
    df_current_month = df_dept[(df_dept['month'] == current_month) & (df_dept['year'] == current_year)]
    current_month_incidents = len(df_current_month)
    df_last_month = df_dept[(df_dept['month'] == current_month-1) & (df_dept['year'] == current_year)]
    last_month_incidents = len(df_last_month)
    incident_trend = ((current_month_incidents - last_month_incidents)/last_month_incidents*100) if last_month_incidents>0 else 0

    # Injury rate
    if 'was_injured' in df_dept.columns:
        df_dept['injury_binary'] = df_dept['was_injured'].apply(lambda x: 1 if str(x).strip().lower() in ['yes','true','1','y','injured'] else 0)
        injury_rate = df_dept['injury_binary'].mean() * 100
    else:
        injury_rate = 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Incidents", total_incidents)
    col2.metric("This Month", current_month_incidents, f"{incident_trend:+.1f}%")
    col3.metric("Injury Rate", f"{injury_rate:.1f}%")
    col4.metric("Departments", df_dept['department'].nunique() if 'department' in df_dept.columns else 0)
    col5.metric("Locations", df_dept['location'].nunique() if 'location' in df_dept.columns else 0)

    # --- Time Analysis ---
    if 'hour' in df_dept.columns and 'day_of_week' in df_dept.columns:
        st.subheader("⏰ Time Pattern Analysis")
        col1, col2, col3 = st.columns(3)
        # Heatmap
        heatmap_data = df_dept.groupby(['day_of_week','hour']).size().reset_index(name='incidents')
        heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='incidents').fillna(0)
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        heatmap_pivot = heatmap_pivot.reindex(day_order)
        fig = px.imshow(heatmap_pivot, title="Incident Heatmap: Day vs Hour", labels=dict(x="Hour",y="Day",color="Incidents"), color_continuous_scale="Reds")
        col1.plotly_chart(fig, use_container_width=True)
        # Monthly
        if 'month' in df_dept.columns:
            monthly_incidents = df_dept.groupby('month').size().reset_index(name='incidents')
            monthly_incidents['month_name'] = monthly_incidents['month'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
            fig = px.bar(monthly_incidents, x='month_name', y='incidents', title="Monthly Incidents")
            col2.plotly_chart(fig, use_container_width=True)
        # Weekly
        daily_incidents = df_dept['day_of_week'].value_counts().reindex(day_order).reset_index()
        daily_incidents.columns=['day','incidents']
        fig = px.line(daily_incidents, x='day', y='incidents', markers=True, title="Weekly Incident Pattern")
        col3.plotly_chart(fig, use_container_width=True)

    # --- Department Risk Matrix ---
    if 'department' in df.columns and len(df_dept)>0:
        st.subheader("🏢 Department Risk Matrix")
        agg_dict = {df.columns[0]:'count'}
        if 'injury_binary' in df_dept.columns:
            agg_dict['injury_binary']='mean'
        dept_analysis = df_dept.groupby('department').agg(agg_dict).reset_index()
        if 'injury_binary' in dept_analysis.columns:
            dept_analysis.columns=['department','incident_count','injury_rate']
            dept_analysis['injury_rate'] *= 100
            fig = px.scatter(dept_analysis, x='incident_count', y='injury_rate', size='incident_count',
                             color='injury_rate', hover_name='department', title="Department Risk Matrix",
                             color_continuous_scale='Reds', size_max=30)
            avg_injury_rate = dept_analysis['injury_rate'].mean()
            max_incidents = dept_analysis['incident_count'].max()
            fig.add_hline(y=avg_injury_rate, line_dash="dash", line_color="gray", annotation_text=f"Avg Injury Rate ({avg_injury_rate:.1f}%)")
            fig.add_vline(x=max_incidents/2, line_dash="dash", line_color="gray", annotation_text="Mid Incident Count")
            st.plotly_chart(fig, use_container_width=True)

    # --- Incident Category Analysis ---
    category_col = None
    for col_option in ['label','category','incident_category']:
        if col_option in df_dept.columns:
            category_col = col_option
            break
    if category_col:
        st.subheader("🏷️ Incident Category Analysis")
        category_counts = df_dept[category_col].value_counts().reset_index()
        category_counts.columns=['category','count']
        col1, col2 = st.columns(2)
        fig = px.pie(category_counts, names='category', values='count', title="Incident Distribution")
        col1.plotly_chart(fig, use_container_width=True)
        # Trends
        if 'incident_date' in df_dept.columns:
            category_trends = df_dept.groupby([df_dept['incident_date'].dt.date, category_col]).size().reset_index(name='count')
            fig = px.line(category_trends, x='incident_date', y='count', color=category_col, title="Category Trends Over Time")
            col2.plotly_chart(fig, use_container_width=True)

    # --- Text Analytics ---
    if 'incident_description' in df_dept.columns:
        st.subheader("📝 Incident Description Word Cloud")
        text = " ".join(str(desc) for desc in df_dept['incident_description'].dropna())
        if text.strip():
            stop_words = {'the','and','or','but','in','on','at','to','for','of','with','by','a','an'}
            words = text.lower().split()
            filtered_text = " ".join([w for w in words if w not in stop_words and len(w)>2])
            wordcloud = WordCloud(width=400,height=300,background_color="white").generate(filtered_text)
            fig, ax = plt.subplots(figsize=(8,6))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

    st.subheader("💡 Recommendations")
    recommendations = []
    if 'hour' in df_dept.columns:
        peak_hours = df_dept['hour'].value_counts().head(3)
        if len(peak_hours)>0:
            recommendations.append(f"🕐 Peak Risk Hours: {', '.join(str(h) for h in peak_hours.index)}")
    if 'department' in df_dept.columns:
        high_risk_depts = df_dept['department'].value_counts().head(2)
        recommendations.append(f"🏢 Focus Area: {', '.join(high_risk_depts.index)}")
    if recommendations:
        for rec in recommendations:
            st.write(f"• {rec}")

else:
    st.info("👆 Upload your incident CSV file to unlock powerful analytics")
