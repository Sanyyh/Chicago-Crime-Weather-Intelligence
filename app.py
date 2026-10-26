import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Chicago Crime & Weather Intelligence",
    page_icon="🚨",
    layout="wide"
)


@st.cache_resource
def load_model():
    model = joblib.load("final_crime_model.joblib")
    encoder = joblib.load("final_label_encoder.joblib")
    return model, encoder


@st.cache_data
def load_heatmap_data(filepath="crime_weather_merged.csv", sample_size=60000):
    cols = [
        'Primary Type', 'Latitude', 'Longitude', 'District',
        'temperature_2m_mean', 'rain_sum', 'snowfall_sum', 'hour'
    ]
    df = pd.read_csv(filepath, usecols=lambda c: c in cols)
    df = df.dropna(subset=['Primary Type', 'Latitude', 'Longitude', 'temperature_2m_mean'])
    
    
    df = df[(df['Latitude'].between(41.6, 42.1)) & (df['Longitude'].between(-87.95, -87.5))]
    
    
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    return df

try:
    model, encoder = load_model()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure 'final_crime_model.joblib' is present.")
    st.stop()

# Load mapping data
with st.spinner("Loading heatmap dataset..."):
    df_history = load_heatmap_data()


DISTRICT_DETAILS = {
    1: {"name": "Central (Downtown / The Loop)", "lat": 41.8674, "lon": -87.6288},
    2: {"name": "Wentworth (Bronzeville)", "lat": 41.8174, "lon": -87.6188},
    3: {"name": "Grand Crossing (South Shore)", "lat": 41.7798, "lon": -87.6046},
    4: {"name": "South Chicago", "lat": 41.7360, "lon": -87.5684},
    5: {"name": "Calumet (Roseland / Pullman)", "lat": 41.6927, "lon": -87.6043},
    6: {"name": "Gresham (Chatham)", "lat": 41.7486, "lon": -87.6527},
    7: {"name": "Englewood", "lat": 41.7788, "lon": -87.6640},
    8: {"name": "Chicago Lawn", "lat": 41.7797, "lon": -87.7262},
    9: {"name": "Deering (Bridgeport)", "lat": 41.8374, "lon": -87.6558},
    10: {"name": "Ogden (Little Village / Lawndale)", "lat": 41.8565, "lon": -87.7082},
    11: {"name": "Harrison (Garfield Park / Humboldt)", "lat": 41.8797, "lon": -87.7275},
    12: {"name": "Near West (West Loop / UIC)", "lat": 41.8631, "lon": -87.6749},
    14: {"name": "Shakespeare (Wicker Park / Logan Sq)", "lat": 41.9213, "lon": -87.6974},
    15: {"name": "Austin", "lat": 41.8893, "lon": -87.7663},
    16: {"name": "Jefferson Park (O'Hare area)", "lat": 41.9740, "lon": -87.7660},
    17: {"name": "Albany Park / Irving Park", "lat": 41.9660, "lon": -87.7281},
    18: {"name": "Near North (River North / Gold Coast)", "lat": 41.9032, "lon": -87.6433},
    19: {"name": "Town Hall (Lakeview / Lincoln Park)", "lat": 41.9470, "lon": -87.6534},
    20: {"name": "Lincoln (Edgewater / Andersonville)", "lat": 41.9790, "lon": -87.6702},
    22: {"name": "Morgan Park / Beverly", "lat": 41.7161, "lon": -87.6698},
    24: {"name": "Rogers Park", "lat": 42.0097, "lon": -87.6734},
    25: {"name": "Grand Central (Belmont Cragin)", "lat": 41.9186, "lon": -87.7656}
}

# Navigation Tabs
tab_map, tab_matrix, tab_predict = st.tabs([
    "🗺️ Chicago Spatial Heatmap",
    "📈 Weather-Crime Correlation Matrix",
    "🔮 Real-Time Crime Prediction"
])

# TAB 1: GEOGRAPHIC CRIME HEATMAP (PyDeck)

with tab_map:
    st.subheader("🔥 Geographic Crime Density Under Specific Weather Conditions")
    st.markdown("Filter by crime type and weather conditions to visualize how crime patterns shift across Chicago neighborhoods.")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        crime_types = ["ALL"] + sorted(df_history['Primary Type'].unique().tolist())
        selected_crime = st.selectbox("Filter Crime Type", crime_types, index=0)
        
    with col_f2:
        temp_filter = st.selectbox(
            "Temperature Category",
            ["All Temperatures", "Freezing (< 0°C)", "Cold (0°C - 12°C)", "Pleasant (12°C - 24°C)", "Hot (> 24°C)"]
        )
        
    with col_f3:
        rain_filter = st.selectbox(
            "Precipitation",
            ["All Weather", "Dry Days Only (Rain = 0)", "Rainy / Snowy Days (> 1mm)"]
        )
        
    # Apply filters
    filtered_df = df_history.copy()
    
    if selected_crime != "ALL":
        filtered_df = filtered_df[filtered_df['Primary Type'] == selected_crime]
        
    if temp_filter == "Freezing (< 0°C)":
        filtered_df = filtered_df[filtered_df['temperature_2m_mean'] < 0]
    elif temp_filter == "Cold (0°C - 12°C)":
        filtered_df = filtered_df[filtered_df['temperature_2m_mean'].between(0, 12)]
    elif temp_filter == "Pleasant (12°C - 24°C)":
        filtered_df = filtered_df[filtered_df['temperature_2m_mean'].between(12, 24)]
    elif temp_filter == "Hot (> 24°C)":
        filtered_df = filtered_df[filtered_df['temperature_2m_mean'] > 24]
        
    if rain_filter == "Dry Days Only (Rain = 0)":
        filtered_df = filtered_df[(filtered_df['rain_sum'] == 0) & (filtered_df['snowfall_sum'] == 0)]
    elif rain_filter == "Rainy / Snowy Days (> 1mm)":
        filtered_df = filtered_df[(filtered_df['rain_sum'] > 1.0) | (filtered_df['snowfall_sum'] > 0.5)]

    st.write(f"Displaying **{len(filtered_df):,}** incidents matching current filters.")

    # PyDeck Spatial Heatmap Layer
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=filtered_df,
        get_position=["Longitude", "Latitude"],
        aggregation=pdk.types.String("MEAN"),
        radius_pixels=35,
        intensity=1.2,
        threshold=0.03
    )

    view_state = pdk.ViewState(
        latitude=41.8781,
        longitude=-87.6298,
        zoom=10,
        pitch=20
    )

    st.pydeck_chart(pdk.Deck(
        layers=[heatmap_layer],
        initial_view_state=view_state,
        map_style="road"
    ))


# TAB 2: WEATHER VS CRIME FREQUENCY HEATMAP MATRIX

with tab_matrix:
    st.subheader("📊 Weather Brackets vs. Crime Occurrence Matrix")
    st.markdown("This heatmap shows the relationship between temperature brackets and the distribution of primary crime types.")
    
    # Bucket temperatures
    temp_bins = [-30, 0, 10, 20, 30, 50]
    temp_labels = ['Freezing (<0°C)', 'Cold (0-10°C)', 'Moderate (10-20°C)', 'Warm (20-30°C)', 'Hot (>30°C)']
    
    matrix_df = df_history.copy()
    matrix_df['Temp_Bracket'] = pd.cut(matrix_df['temperature_2m_mean'], bins=temp_bins, labels=temp_labels)
    
    # Filter top 8 crime classes
    top_crimes = matrix_df['Primary Type'].value_counts().nlargest(8).index
    matrix_df = matrix_df[matrix_df['Primary Type'].isin(top_crimes)]
    
    # Pivot Table
    pivot_table = pd.crosstab(matrix_df['Primary Type'], matrix_df['Temp_Bracket'], normalize='columns') * 100
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': '% of Crimes within Temp Bracket'}, ax=ax)
    ax.set_title("Crime Type Proportion Across Temperature Ranges (%)", fontsize=12)
    ax.set_xlabel("Temperature Range", fontsize=10)
    ax.set_ylabel("Crime Category", fontsize=10)
    st.pyplot(fig)

# TAB 3: REAL-TIME CRIME PREDICTION
with tab_predict:
    st.subheader("🔮 Run Individual Crime Predictions")
    
    col_input, col_output = st.columns(2)
    
    with col_input:
        st.markdown("#### Scenario Configuration")
        sel_dist = st.selectbox(
            "Target Police District",
            options=list(DISTRICT_DETAILS.keys()),
            format_func=lambda d: f"District {d} - {DISTRICT_DETAILS[d]['name']}",
            index=0
        )
        sim_hour = st.slider("Hour (0-23)", 0, 23, 21)
        sim_month = st.selectbox("Month", list(range(1, 13)), index=6,
                                 format_func=lambda m: pd.to_datetime(f"2026-{m}-01").strftime("%B"))
        sim_day = st.selectbox("Day of Week", list(range(7)), index=4,
                               format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d])
        sim_temp = st.slider("Temperature (°C)", -15.0, 38.0, 24.0, 0.5)
        sim_rain = st.slider("Rainfall (mm)", 0.0, 40.0, 0.0, 0.5)
        sim_snow = st.slider("Snowfall (cm)", 0.0, 25.0, 0.0, 0.5)
        sim_wind = st.slider("Max Wind Speed (km/h)", 0.0, 60.0, 15.0, 1.0)
        sim_sun = st.slider("Sunshine Duration (seconds)", 0.0, 50000.0, 20000.0, 1000.0)
        
    with col_output:
        st.markdown("#### Model Predictions")
        sim_lat = DISTRICT_DETAILS[sel_dist]["lat"]
        sim_lon = DISTRICT_DETAILS[sel_dist]["lon"]
        sim_weekend = 1 if sim_day in [5, 6] else 0
        
        sample_row = pd.DataFrame([{
            'hour': sim_hour,
            'month': sim_month,
            'day_of_week': sim_day,
            'is_weekend': sim_weekend,
            'Latitude': sim_lat,
            'Longitude': sim_lon,
            'District': sel_dist,
            'temperature_2m_mean': sim_temp,
            'precipitation_sum': sim_rain + (sim_snow * 10),
            'rain_sum': sim_rain,
            'snowfall_sum': sim_snow,
            'windspeed_10m_max': sim_wind,
            'sunshine_duration': sim_sun
        }])
        
        pred_idx = model.predict(sample_row)[0]
        predicted_crime = encoder.inverse_transform([pred_idx])[0]
        probs = model.predict_proba(sample_row)[0]
        
        st.metric("Predicted Crime", predicted_crime, f"Confidence: {probs.max()*100:.1f}%")
        
        df_probs = pd.DataFrame({
            'Crime': encoder.classes_,
            'Probability (%)': probs * 100
        }).sort_values(by='Probability (%)', ascending=True)
        
        fig_prob, ax_prob = plt.subplots(figsize=(7, 4))
        ax_prob.barh(df_probs['Crime'], df_probs['Probability (%)'], color='#2b5c8f')
        ax_prob.set_xlabel("Probability (%)")
        st.pyplot(fig_prob)