# analytics_app.py
# This script provides an interactive dashboard for incident analytics,
# with added features for predictive analysis and risk forecasting.

# --- Core Libraries ---
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore') # Suppress warnings

# Import the ARIMA model from statsmodels
from statsmodels.tsa.arima.model import ARIMA

# --- Page Configuration ---
st.set_page_config(page_title="Advanced Incident Analytics", layout="wide")

st.title("📊 Advanced Incident Analytics & Safety Intelligence")
st.markdown("*Comprehensive insights to improve workplace safety and prevent future incidents*")

# --- Upload CSV Section ---
uploaded_file = st.file_uploader("Upload your incident CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # --- Enhanced Data Cleaning and Feature Engineering ---
    # Convert date-like columns to datetime objects
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Extract temporal features if 'incident_date' exists
    if 'incident_date' in df.columns:
        df['hour'] = pd.to_datetime(df['incident_date']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['incident_date']).dt.day_name()
        df['month'] = pd.to_datetime(df['incident_date']).dt.month
        df['year'] = pd.to_datetime(df['incident_date']).dt.year
    
    # Clean and standardize text columns
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
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
    
    # Calculate trends based on incident_date
    if 'incident_date' in df.columns:
        df_current_month = df[(df['month'] == current_month) & (df['year'] == current_year)]
        
        # Determine previous month for comparison
        last_month = current_month - 1 if current_month > 1 else 12
        last_month_year = current_year if current_month > 1 else current_year - 1
        df_last_month = df[(df['month'] == last_month) & (df['year'] == last_month_year)]
        
        current_month_incidents = len(df_current_month)
        last_month_incidents = len(df_last_month)
        
        if last_month_incidents > 0:
            incident_trend = ((current_month_incidents - last_month_incidents) / last_month_incidents) * 100
        else:
            incident_trend = 0
    else:
        incident_trend = 0
        current_month_incidents = 0
    
    # Injury rate calculation
    injury_rate = 0
    if 'was_injured' in df.columns:
        try:
            injured_count = 0
            # Handle various data types for 'was_injured'
            if df['was_injured'].dtype == 'bool':
                injured_count = df['was_injured'].sum()
            elif df['was_injured'].dtype == 'object':
                injured_count = df['was_injured'].dropna().apply(
                    lambda x: str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured']
                ).sum()
            else:
                injured_count = df['was_injured'].fillna(0).astype(float).gt(0).sum()
            
            injury_rate = (injured_count / len(df)) * 100 if len(df) > 0 else 0
        except:
            injury_rate = 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Incidents", total_incidents)
    col2.metric("This Month", current_month_incidents, f"{incident_trend:+.1f}%")
    col3.metric("Injury Rate", f"{injury_rate:.1f}%")
    col4.metric("Departments", df['department'].nunique() if 'department' in df.columns else 0)
    col5.metric("Locations", df['location'].nunique() if 'location' in df.columns else 0)

    # --- ADVANCED PREDICTIVE ANALYTICS ---
    st.subheader("🔮 Predictive Analytics & Forecasting")
    st.markdown("These charts and indicators predict future risks based on historical data patterns.")
    
    if 'incident_date' in df.columns and len(df) > 10:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 📊 Incident Trend Forecast (ARIMA)")
            
            # Aggregate data by month for time-series analysis
            df_temp = df.copy()
            df_temp['year_month'] = df_temp['incident_date'].dt.to_period('M')
            df_monthly = df_temp.groupby('year_month').size().reset_index(name='incident_count')
            df_monthly['date'] = df_monthly['year_month'].dt.to_timestamp()
            df_monthly = df_monthly.sort_values('date')
            df_monthly = df_monthly.set_index('date') # ARIMA requires a DatetimeIndex
            
            # Check for sufficient data for ARIMA
            if len(df_monthly) >= 15: # Recommend at least 15 months of data
                try:
                    # Fit an ARIMA(1,1,1) model. These parameters (p,d,q) are a common starting point.
                    model = ARIMA(df_monthly['incident_count'], order=(1, 1, 1))
                    model_fit = model.fit()
                    
                    # Generate a 3-month forecast
                    forecast = model_fit.forecast(steps=3)
                    future_dates = pd.date_range(start=df_monthly.index.max() + pd.DateOffset(months=1), periods=3, freq='MS')
                    forecast_series = pd.Series(forecast, index=future_dates)
                    
                    # Plot the actual and predicted data
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_monthly.index, y=df_monthly['incident_count'], 
                                             mode='lines+markers', name='Actual Incidents'))
                    fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series.values, 
                                             mode='lines+markers', name='ARIMA Predicted Incidents', 
                                             line=dict(dash='dash', color='red')))
                    fig.update_layout(title="3-Month Incident Forecast (ARIMA)", 
                                      xaxis_title="Date", 
                                      yaxis_title="Incidents")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"ARIMA model could not be fitted: {e}. Falling back to simple forecast.")
                    # Simple moving average fallback
                    window = min(3, len(df_monthly))
                    last_avg = df_monthly['incident_count'].tail(window).mean()
                    future_dates = pd.date_range(start=df_monthly.index.max() + pd.DateOffset(months=1), periods=3, freq='MS')
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_monthly.index, y=df_monthly['incident_count'], mode='lines+markers', name='Actual Incidents'))
                    fig.add_trace(go.Scatter(x=future_dates, y=[last_avg]*3, mode='lines+markers', name='Predicted Incidents', line=dict(dash='dash', color='red')))
                    fig.update_layout(title="3-Month Incident Forecast (Simple Moving Average)", xaxis_title="Date", yaxis_title="Incidents")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ARIMA requires more data. Please upload at least 15 months of data for a reliable forecast.")
        
        with col2:
            st.write("### 🎯 Predictive Risk Indicators")
            st.markdown("These indicators predict where and when future incidents are most likely to occur.")
            
            risk_factors = []
            
            # Predict high-risk departments based on incident count
            if 'department' in df.columns:
                dept_incidents = df['department'].value_counts()
                if not dept_incidents.empty:
                    high_risk_dept = dept_incidents.index[0]
                    risk_factors.append(f"🔴 **High Risk Department**: **{high_risk_dept}** ({dept_incidents.iloc[0]} incidents)")
            
            # Predict peak risk time based on historical data
            if 'hour' in df.columns:
                hour_risk = df['hour'].value_counts()
                if not hour_risk.empty:
                    peak_hour = hour_risk.index[0]
                    risk_factors.append(f"⏰ **Peak Risk Time**: **{peak_hour}:00** ({hour_risk.iloc[0]} incidents)")
            
            # Predict highest risk day of the week
            if 'day_of_week' in df.columns:
                day_risk = df['day_of_week'].value_counts()
                if not day_risk.empty:
                    risky_day = day_risk.index[0]
                    risk_factors.append(f"📅 **Highest Risk Day**: **{risky_day}** ({day_risk.iloc[0]} incidents)")
            
            # Predict injury hotspots based on department injury rate
            if 'was_injured' in df.columns and 'department' in df.columns:
                try:
                    # Create a binary injury indicator for calculation
                    df['injury_binary'] = df['was_injured'].apply(
                        lambda x: 1 if pd.notna(x) and str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured'] else 0
                    )
                    injury_by_dept = df.groupby('department')['injury_binary'].mean().sort_values(ascending=False)
                    
                    if not injury_by_dept.empty and injury_by_dept.iloc[0] > 0:
                        high_injury_dept = injury_by_dept.index[0]
                        injury_pct = injury_by_dept.iloc[0] * 100
                        risk_factors.append(f"🏥 **Injury Hotspot**: **{high_injury_dept}** has a high injury rate of {injury_pct:.1f}%")
                except Exception as e:
                    st.warning(f"Could not calculate injury hotspot: {e}")
                    
            for factor in risk_factors:
                st.markdown(factor)
                
    # --- Other sections (unchanged for brevity) ---
    st.subheader("⏰ Time Pattern Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'hour' in df.columns and 'day_of_week' in df.columns:
            heatmap_data = df.groupby(['day_of_week', 'hour']).size().reset_index(name='incidents')
            heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='incidents').fillna(0)
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_pivot = heatmap_pivot.reindex(day_order)
            fig = px.imshow(heatmap_pivot, title="Incident Heatmap: Day vs Hour", labels=dict(x="Hour of Day", y="Day of Week", color="Incidents"), color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'month' in df.columns:
            monthly_incidents = df.groupby('month').size().reset_index(name='incidents')
            monthly_incidents['month_name'] = monthly_incidents['month'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
            fig = px.bar(monthly_incidents, x='month_name', y='incidents', title="Seasonal Incident Patterns")
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        if 'day_of_week' in df.columns:
            daily_incidents = df['day_of_week'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index()
            daily_incidents.columns = ['day', 'incidents']
            fig = px.line(daily_incidents, x='day', y='incidents', markers=True, title="Weekly Incident Pattern")
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
    st.subheader("🏢 Department & Location Intelligence")
    col1, col2 = st.columns(2)
    with col1:
        if 'department' in df.columns:
            agg_dict = {df.columns[0]: 'count'}
            if 'was_injured' in df.columns:
                if df['was_injured'].dtype == 'bool':
                    agg_dict['was_injured'] = 'mean'
                else:
                    df['injury_binary'] = df['was_injured'].apply(lambda x: 1 if pd.notna(x) and str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured'] else 0)
                    agg_dict['injury_binary'] = 'mean'
            
            dept_analysis = df.groupby('department').agg(agg_dict).reset_index()
            if 'was_injured' in df.columns or 'injury_binary' in dept_analysis.columns:
                injury_rate_col = 'was_injured' if 'was_injured' in dept_analysis.columns else 'injury_binary'
                dept_analysis.columns = ['department', 'incident_count', 'injury_rate']
                dept_analysis['injury_rate'] *= 100
                fig = px.scatter(dept_analysis, x='incident_count', y='injury_rate', size='incident_count', hover_name='department', title="Department Risk Matrix", labels={'incident_count': 'Total Incidents', 'injury_rate': 'Injury Rate (%)'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.bar(dept_analysis, x='department', y=dept_analysis.columns[1], title="Incidents by Department")
                st.plotly_chart(fig, use_container_width=True)
    with col2:
        if 'location' in df.columns:
            location_incidents = df['location'].value_counts().head(10).reset_index()
            location_incidents.columns = ['location', 'incidents']
            fig = px.bar(location_incidents, x='incidents', y='location', orientation='h', title="Top 10 Incident Locations")
            st.plotly_chart(fig, use_container_width=True)
    
    if 'incident_description' in df.columns:
        st.subheader("📝 Advanced Text Analytics")
        col1, col2 = st.columns(2)
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        with col1:
            st.write("**☁️ Incident Description Word Cloud**")
            text = " ".join(str(desc) for desc in df['incident_description'].dropna())
            if text.strip():
                stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
                words = text.lower().split()
                filtered_text = " ".join([word for word in words if word not in stop_words and len(word) > 2])
                wordcloud = WordCloud(width=400, height=300, background_color="white").generate(filtered_text)
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
                word_freq = word_freq[word_freq.index.str.len() > 3]
                fig = px.bar(x=word_freq.values, y=word_freq.index, orientation='h', title="Most Common Terms")
                st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("💡 AI-Driven Recommendations")
    recommendations = []
    if 'hour' in df.columns:
        peak_hours = df['hour'].value_counts().head(2)
        if len(peak_hours) > 0:
            recommendations.append(f"🕐 **Peak Risk Hours**: Increase safety supervision during **{peak_hours.index[0]}:00-{peak_hours.index[0]+1}:00** (highest incident time)")
    if 'department' in df.columns:
        high_risk_depts = df['department'].value_counts().head(2)
        if len(high_risk_depts) > 0:
            recommendations.append(f"🏢 **Focus Area**: Prioritize safety training in **{high_risk_depts.index[0]}** department ({high_risk_depts.iloc[0]} incidents)")
    if 'was_injured' in df.columns and injury_rate > 0:
        if injury_rate > 10:
            recommendations.append(f"🏥 **Critical**: **{injury_rate:.1f}%** injury rate requires immediate safety protocol review")
        if 'department' in df.columns:
            try:
                if df['was_injured'].dtype == 'bool':
                    dept_injury_rates = df.groupby('department')['was_injured'].agg(['sum', 'count', 'mean'])
                else:
                    df['injury_binary'] = df['was_injured'].apply(lambda x: 1 if pd.notna(x) and str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured'] else 0)
                    dept_injury_rates = df.groupby('department')['injury_binary'].agg(['sum', 'count', 'mean'])
                dept_injury_rates = dept_injury_rates[dept_injury_rates['count'] >= 3]
                if len(dept_injury_rates) > 0 and dept_injury_rates['mean'].max() > 0:
                    worst_dept = dept_injury_rates['mean'].idxmax()
                    worst_rate = dept_injury_rates.loc[worst_dept, 'mean'] * 100
                    recommendations.append(f"🎯 **Targeted Intervention**: **{worst_dept}** has **{worst_rate:.1f}%** injury rate - implement enhanced safety measures")
            except:
                pass
    if 'incident_date' in df.columns and len(df) > 30:
        recent_30d = df[df['incident_date'] >= (datetime.now() - timedelta(days=30))]
        if len(recent_30d) > len(df) * 0.3:
            recommendations.append("📈 **Trend Alert**: Recent surge in incidents detected - conduct immediate safety audit")
    
    if recommendations:
        for i, rec in enumerate(recommendations[:5], 1):
            st.write(f"{i}. {rec}")
    else:
        st.write("📊 **Data Analysis**: Upload more comprehensive data for personalized recommendations")
    
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
    st.info("👆 Upload your incident CSV file to unlock powerful safety analytics and predictive insights")
    st.write("**Expected columns for optimal analysis:**")
    st.write("• Reporter Name, Person Involved, Incident Date & Time")
    st.write("• Department & Location, Incident Description")
    st.write("• Label/Category, Injury Information (was_injured: boolean)")