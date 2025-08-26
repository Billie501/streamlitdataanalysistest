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

st.set_page_config(page_title="Advanced Incident Analytics", layout="wide")

st.title("📊 Advanced Incident Analytics & Safety Intelligence")
st.markdown("*Comprehensive insights to improve workplace safety and prevent future incidents*")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload your incident CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # --- Enhanced Data Cleaning ---
    # Handle various date formats (revert to working version)
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Standardize time columns
    time_columns = [col for col in df.columns if 'time' in col.lower()]
    for col in time_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.time
    
    # Extract hour from datetime for time analysis
    if 'incident_date' in df.columns:
        df['day_of_week'] = pd.to_datetime(df['incident_date']).dt.day_name()
        df['month'] = pd.to_datetime(df['incident_date']).dt.month
        df['year'] = pd.to_datetime(df['incident_date']).dt.year
    
    # Extract hour from the TIME column for proper time analysis
    if 'incident_time' in df.columns:
        try:
            # Handle time format like 06:45:00
            df['hour'] = pd.to_datetime(df['incident_time'], errors='coerce').dt.hour
        except:
            # Fallback for different time formats
            df['hour'] = pd.to_datetime(df['incident_time'].astype(str), format='%H:%M:%S', errors='coerce').dt.hour
    
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
    
    # Calculate advanced metrics
    total_incidents = len(df)
    current_month = datetime.now().month
    current_year = datetime.now().year
    
    # Filter for current period if date exists
    if 'incident_date' in df.columns:
        df_current_month = df[(df['month'] == current_month) & (df['year'] == current_year)]
        df_last_month = df[(df['month'] == current_month-1) & (df['year'] == current_year)]
        
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
    
    # Injury rate calculation - now specifically for 'was_injured' column
    injury_rate = 0
    if 'was_injured' in df.columns:
        # Handle different data types (boolean, text, numbers)
        try:
            if df['was_injured'].dtype == 'bool':
                injury_rate = (df['was_injured'].sum() / len(df)) * 100 if len(df) > 0 else 0
            elif df['was_injured'].dtype == 'object':
                # Count 'Yes', 'True', '1', etc. as injuries
                injured_count = df['was_injured'].dropna().apply(
                    lambda x: str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured']
                ).sum()
                injury_rate = (injured_count / len(df)) * 100 if len(df) > 0 else 0
            else:
                # Numeric - count values > 0
                injury_rate = (df['was_injured'].fillna(0).astype(float).gt(0).sum() / len(df)) * 100 if len(df) > 0 else 0
        except:
            injury_rate = 0
    
    # Display KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Incidents", total_incidents)
    col2.metric("This Month", current_month_incidents, f"{incident_trend:+.1f}%")
    col3.metric("Injury Rate", f"{injury_rate:.1f}%")
    col4.metric("Departments", df['department'].nunique() if 'department' in df.columns else 0)
    col5.metric("Locations", df['location'].nunique() if 'location' in df.columns else 0)

    # --- PREDICTIVE ANALYTICS ---
    st.subheader("🔮 Advanced Predictive Analytics & Forecasting")
    
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
                future_dates = pd.date_range(start=df_monthly['date'].max() + pd.DateOffset(months=1), 
                                           periods=3, freq='MS')  # MS = month start
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_monthly['date'], y=df_monthly['incident_count'], 
                                       mode='lines+markers', name='Actual Incidents'))
                fig.add_trace(go.Scatter(x=future_dates, y=[last_avg]*3, 
                                       mode='lines+markers', name='Predicted', 
                                       line=dict(dash='dash')))
                fig.update_layout(title="3-Month Incident Forecast", xaxis_title="Date", yaxis_title="Incidents")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**🎯 Risk Indicators**")
            
            # Calculate risk scores
            risk_factors = []
            
            # Department risk
            if 'department' in df.columns:
                dept_incidents = df['department'].value_counts()
                high_risk_dept = dept_incidents.index[0] if len(dept_incidents) > 0 else "Unknown"
                risk_factors.append(f"🔴 High Risk Department: {high_risk_dept} ({dept_incidents.iloc[0]} incidents)")
            
            # Time-based risk
            if 'hour' in df.columns:
                hour_risk = df['hour'].value_counts()
                if len(hour_risk) > 0:
                    peak_hour = hour_risk.index[0]
                    risk_factors.append(f"⏰ Peak Risk Time: {peak_hour}:00 ({hour_risk.iloc[0]} incidents)")
            
            # Day of week risk
            if 'day_of_week' in df.columns:
                day_risk = df['day_of_week'].value_counts()
                if len(day_risk) > 0:
                    risky_day = day_risk.index[0]
                    risk_factors.append(f"📅 Highest Risk Day: {risky_day} ({day_risk.iloc[0]} incidents)")
            
            # Injury severity prediction
            if 'was_injured' in df.columns and 'department' in df.columns:
                try:
                    if df['was_injured'].dtype == 'bool':
                        injury_by_dept = df.groupby('department')['was_injured'].mean().sort_values(ascending=False)
                    else:
                        # For non-boolean, create a binary injury indicator
                        df['injury_binary'] = df['was_injured'].apply(
                            lambda x: 1 if pd.notna(x) and str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured'] else 0
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

    # --- ADVANCED FORECASTING MODELS ---
    st.subheader("📈 Advanced Forecasting Models")
    
    if 'incident_date' in df.columns and len(df) > 20:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**🔄 Seasonal Pattern Forecast**")
            # Weekly pattern forecasting
            if 'day_of_week' in df.columns:
                weekly_pattern = df.groupby('day_of_week').size()
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekly_pattern = weekly_pattern.reindex(days_order, fill_value=0)
                
                # Predict next week based on historical average
                avg_weekly = weekly_pattern.mean()
                next_week_prediction = weekly_pattern.copy()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=weekly_pattern.index, y=weekly_pattern.values, name='Historical Average'))
                fig.add_trace(go.Scatter(x=weekly_pattern.index, y=[avg_weekly]*7, 
                                       mode='lines', name='Weekly Average', line=dict(dash='dash')))
                fig.update_layout(title="Weekly Pattern Forecast", xaxis_title="Day", yaxis_title="Expected Incidents")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**⏰ Hourly Risk Forecast**")
            if 'hour' in df.columns:
                hourly_pattern = df.groupby('hour').size()
                
                # Create risk zones
                risk_zones = []
                for hour, count in hourly_pattern.items():
                    if count >= hourly_pattern.quantile(0.75):
                        risk_zones.append(('High Risk', hour, count))
                    elif count >= hourly_pattern.quantile(0.25):
                        risk_zones.append(('Medium Risk', hour, count))
                    else:
                        risk_zones.append(('Low Risk', hour, count))
                
                # Create risk zone chart
                risk_df = pd.DataFrame(risk_zones, columns=['Risk Level', 'Hour', 'Incidents'])
                color_map = {'High Risk': 'red', 'Medium Risk': 'orange', 'Low Risk': 'green'}
                
                fig = px.bar(risk_df, x='Hour', y='Incidents', color='Risk Level',
                           color_discrete_map=color_map, title="Hourly Risk Zones")
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.write("**📊 Department Risk Trajectory**")
            if 'department' in df.columns and 'incident_date' in df.columns:
                # Calculate department trends over time
                dept_trends = df.groupby(['department', df['incident_date'].dt.date]).size().reset_index(name='daily_incidents')
                dept_trends['incident_date'] = pd.to_datetime(dept_trends['incident_date'])
                
                # Get top 3 departments by incident count
                top_depts = df['department'].value_counts().head(3).index.tolist()
                dept_trends_top = dept_trends[dept_trends['department'].isin(top_depts)]
                
                if len(dept_trends_top) > 0:
                    fig = px.line(dept_trends_top, x='incident_date', y='daily_incidents', 
                                color='department', title="Department Incident Trends")
                    st.plotly_chart(fig, use_container_width=True)

    # --- PREDICTIVE RISK SCORING ---
    st.subheader("🎯 Predictive Risk Scoring System")
    
    if len(df) > 10:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🏢 Department Risk Score**")
            if 'department' in df.columns:
                dept_risk = df.groupby('department').agg({
                    df.columns[0]: 'count',  # incident frequency
                    'was_injured': 'mean' if 'was_injured' in df.columns else lambda x: 0  # severity
                }).reset_index()
                
                if 'was_injured' in df.columns:
                    dept_risk.columns = ['department', 'frequency', 'severity']
                    # Calculate composite risk score (frequency * severity * 100)
                    dept_risk['risk_score'] = (dept_risk['frequency'] * dept_risk['severity'] * 100).round(1)
                else:
                    dept_risk.columns = ['department', 'frequency']
                    dept_risk['risk_score'] = dept_risk['frequency']
                
                dept_risk = dept_risk.sort_values('risk_score', ascending=False)
                
                # Display as progress bars
                for _, row in dept_risk.head(5).iterrows():
                    max_score = dept_risk['risk_score'].max()
                    progress = row['risk_score'] / max_score if max_score > 0 else 0
                    st.write(f"**{row['department']}**")
                    st.progress(progress)
                    st.write(f"Risk Score: {row['risk_score']}")
        
        with col2:
            st.write("**⏰ Time-Based Risk Prediction**")
            if 'hour' in df.columns:
                # Create risk probability by hour
                hourly_risk = df.groupby('hour').size()
                total_incidents = hourly_risk.sum()
                hourly_prob = (hourly_risk / total_incidents * 100).round(1)
                
                # Next 8 hours prediction
                current_hour = datetime.now().hour
                next_hours = [(current_hour + i) % 24 for i in range(8)]
                
                predictions = []
                for hour in next_hours:
                    prob = hourly_prob.get(hour, 0)
                    predictions.append({
                        'Hour': f"{hour:02d}:00",
                        'Risk Probability': f"{prob}%",
                        'Risk Level': 'High' if prob > hourly_prob.quantile(0.75) else 
                                   'Medium' if prob > hourly_prob.quantile(0.25) else 'Low'
                    })
                
                pred_df = pd.DataFrame(predictions)
                st.dataframe(pred_df, hide_index=True)

    # --- INCIDENT IMPACT FORECASTING ---
    st.subheader("💼 Business Impact Forecasting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**💰 Cost Impact Projection**")
        # Estimate costs based on incident types
        if 'was_injured' in df.columns:
            injured_count = df['was_injured'].sum() if df['was_injured'].dtype == 'bool' else \
                          df['was_injured'].apply(lambda x: 1 if pd.notna(x) and str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured'] else 0).sum()
            
            # Industry average cost estimates (these would be customizable)
            avg_injury_cost = 45000  # Average workplace injury cost
            avg_incident_cost = 3000  # Average non-injury incident cost
            
            monthly_incidents = len(df) / max(1, df['month'].nunique()) if 'month' in df.columns else len(df)
            monthly_injuries = injured_count / max(1, df['month'].nunique()) if 'month' in df.columns else injured_count
            
            projected_monthly_cost = (monthly_injuries * avg_injury_cost) + ((monthly_incidents - monthly_injuries) * avg_incident_cost)
            projected_annual_cost = projected_monthly_cost * 12
            
            st.metric("Projected Monthly Cost", f"${projected_monthly_cost:,.0f}")
            st.metric("Projected Annual Cost", f"${projected_annual_cost:,.0f}")
            
            # Cost reduction potential
            if monthly_incidents > 0:
                reduction_20pct = projected_annual_cost * 0.2
                st.write(f"**💡 Potential Savings**: 20% incident reduction = **${reduction_20pct:,.0f}** annually")
    
    with col2:
        st.write("**📈 Resource Allocation Forecast**")
        
        # Calculate required safety resources based on risk patterns
        if 'department' in df.columns:
            dept_incidents = df['department'].value_counts()
            total_incidents = dept_incidents.sum()
            
            resource_needs = []
            for dept, count in dept_incidents.head(5).items():
                percentage = (count / total_incidents * 100)
                if percentage > 30:
                    priority = "🔴 Critical"
                    resources = "Full-time safety officer + weekly audits"
                elif percentage > 15:
                    priority = "🟡 High"  
                    resources = "Part-time safety oversight + monthly reviews"
                else:
                    priority = "🟢 Standard"
                    resources = "Standard safety protocols"
                
                resource_needs.append({
                    'Department': dept,
                    'Incident %': f"{percentage:.1f}%",
                    'Priority': priority,
                    'Recommended Resources': resources
                })
            
            resource_df = pd.DataFrame(resource_needs)
            st.dataframe(resource_df, hide_index=True)

    # --- ADVANCED TIME ANALYSIS ---
    if 'incident_date' in df.columns:
        st.subheader("⏰ Time Pattern Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Heatmap by hour and day
            if 'hour' in df.columns and 'day_of_week' in df.columns:
                heatmap_data = df.groupby(['day_of_week', 'hour']).size().reset_index(name='incidents')
                heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='incidents').fillna(0)
                
                # Order days correctly
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heatmap_pivot = heatmap_pivot.reindex(day_order)
                
                fig = px.imshow(heatmap_pivot, 
                              title="Incident Heatmap: Day vs Hour",
                              labels=dict(x="Hour of Day", y="Day of Week", color="Incidents"),
                              color_continuous_scale="Reds")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Seasonal trends
            if 'month' in df.columns:
                monthly_incidents = df.groupby('month').size().reset_index(name='incidents')
                monthly_incidents['month_name'] = monthly_incidents['month'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
                
                fig = px.bar(monthly_incidents, x='month_name', y='incidents',
                           title="Seasonal Incident Patterns")
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Weekly patterns
            if 'day_of_week' in df.columns:
                daily_incidents = df['day_of_week'].value_counts().reindex(
                    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                ).reset_index()
                daily_incidents.columns = ['day', 'incidents']
                
                fig = px.line(daily_incidents, x='day', y='incidents', 
                            markers=True, title="Weekly Incident Pattern")
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

    # --- DEPARTMENT & LOCATION INSIGHTS ---
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
                    df['injury_binary'] = df['was_injured'].apply(
                        lambda x: 1 if pd.notna(x) and str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured'] else 0
                    )
                    agg_dict['injury_binary'] = 'mean'
            
            dept_analysis = df.groupby('department').agg(agg_dict).reset_index()
            
            if 'was_injured' in df.columns or 'injury_binary' in dept_analysis.columns:
                injury_rate_col = 'was_injured' if 'was_injured' in dept_analysis.columns else 'injury_binary'
                dept_analysis.columns = ['department', 'incident_count', 'injury_rate']
                dept_analysis['injury_rate'] *= 100
                
                # Create bubble chart
                fig = px.scatter(dept_analysis, x='incident_count', y='injury_rate', 
                               size='incident_count', hover_name='department',
                               title="Department Risk Matrix",
                               labels={'incident_count': 'Total Incidents', 
                                     'injury_rate': 'Injury Rate (%)'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.bar(dept_analysis, x='department', y=dept_analysis.columns[1],
                           title="Incidents by Department")
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'location' in df.columns:
            location_incidents = df['location'].value_counts().head(10).reset_index()
            location_incidents.columns = ['location', 'incidents']
            
            fig = px.bar(location_incidents, x='incidents', y='location', 
                        orientation='h', title="Top 10 Incident Locations")
            st.plotly_chart(fig, use_container_width=True)

    # --- INCIDENT CATEGORIZATION & ANALYSIS ---
    if 'label' in df.columns or 'category' in df.columns:
        st.subheader("🏷️ Incident Category Analysis")
        
        category_col = 'label' if 'label' in df.columns else 'category'
        
        col1, col2 = st.columns(2)
        
        with col1:
            category_counts = df[category_col].value_counts().reset_index()
            category_counts.columns = ['category', 'count']
            
            fig = px.pie(category_counts, names='category', values='count',
                        title="Incident Distribution by Category")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category trends over time if date is available
            if 'incident_date' in df.columns:
                category_trends = df.groupby([df['incident_date'].dt.date, category_col]).size().reset_index(name='count')
                
                fig = px.line(category_trends, x='incident_date', y='count', 
                            color=category_col, title="Category Trends Over Time")
                st.plotly_chart(fig, use_container_width=True)

    # --- ENHANCED TEXT ANALYSIS ---
    if 'incident_description' in df.columns:
        st.subheader("📝 Advanced Text Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**☁️ Incident Description Word Cloud**")
            text = " ".join(str(desc) for desc in df['incident_description'].dropna())
            if text.strip():
                # Remove common stop words
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
                # Extract key terms
                words = text.lower().split()
                word_freq = pd.Series(words).value_counts().head(10)
                word_freq = word_freq[word_freq.index.str.len() > 3]  # Filter short words
                
                fig = px.bar(x=word_freq.values, y=word_freq.index, 
                           orientation='h', title="Most Common Terms")
                st.plotly_chart(fig, use_container_width=True)

    # --- BUSINESS RECOMMENDATIONS ---
    st.subheader("💡 AI-Driven Recommendations")
    
    recommendations = []
    
    # Time-based recommendations
    if 'hour' in df.columns:
        peak_hours = df['hour'].value_counts().head(2)
        if len(peak_hours) > 0:
            recommendations.append(f"🕐 **Peak Risk Hours**: Increase safety supervision during {peak_hours.index[0]}:00-{peak_hours.index[0]+1}:00 (highest incident time)")
    
    # Department recommendations
    if 'department' in df.columns:
        high_risk_depts = df['department'].value_counts().head(2)
        if len(high_risk_depts) > 0:
            recommendations.append(f"🏢 **Focus Area**: Prioritize safety training in {high_risk_depts.index[0]} department ({high_risk_depts.iloc[0]} incidents)")
    
    # Injury prevention
    if 'was_injured' in df.columns and injury_rate > 0:
        if injury_rate > 10:
            recommendations.append(f"🏥 **Critical**: {injury_rate:.1f}% injury rate requires immediate safety protocol review")
        
        # Department-specific injury recommendations
        if 'department' in df.columns:
            try:
                if df['was_injured'].dtype == 'bool':
                    dept_injury_rates = df.groupby('department')['was_injured'].agg(['sum', 'count', 'mean'])
                else:
                    # Create binary injury indicator
                    df['injury_binary'] = df['was_injured'].apply(
                        lambda x: 1 if pd.notna(x) and str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured'] else 0
                    )
                    dept_injury_rates = df.groupby('department')['injury_binary'].agg(['sum', 'count', 'mean'])
                
                dept_injury_rates = dept_injury_rates[dept_injury_rates['count'] >= 3]  # Only departments with 3+ incidents
                if len(dept_injury_rates) > 0 and dept_injury_rates['mean'].max() > 0:
                    worst_dept = dept_injury_rates['mean'].idxmax()
                    worst_rate = dept_injury_rates.loc[worst_dept, 'mean'] * 100
                    recommendations.append(f"🎯 **Targeted Intervention**: {worst_dept} has {worst_rate:.1f}% injury rate - implement enhanced safety measures")
            except:
                pass
    
    # Trend-based recommendations
    if 'incident_date' in df.columns and len(df) > 30:
        recent_30d = df[df['incident_date'] >= (datetime.now() - timedelta(days=30))]
        if len(recent_30d) > len(df) * 0.3:  # If 30% of incidents in last 30 days
            recommendations.append("📈 **Trend Alert**: Recent surge in incidents detected - conduct immediate safety audit")
    
    # Display recommendations
    if recommendations:
        for i, rec in enumerate(recommendations[:5], 1):  # Limit to top 5
            st.write(f"{i}. {rec}")
    else:
        st.write("📊 **Data Analysis**: Upload more comprehensive data for personalized recommendations")

    # --- EXPORT INSIGHTS ---
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