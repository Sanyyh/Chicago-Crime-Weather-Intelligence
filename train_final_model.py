import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

def train_final():
    filepath = "crime_weather_merged.csv"
    start_year, end_year = 2015, 2026
    
    print(f"Loading full dataset for final training ({start_year}-{end_year})...")
    df = pd.read_csv(filepath)
    
    # Parse date and filter year range
    df['parsed_date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['parsed_date'])
    df['year'] = df['parsed_date'].dt.year
    
    df_full = df[(df['year'] >= start_year) & (df['year'] <= end_year)].copy()
    print(f"Total historical records found: {len(df_full)}")
    
    # Feature engineering
    df_full['month'] = df_full['parsed_date'].dt.month
    df_full['day_of_week'] = df_full['parsed_date'].dt.dayofweek
    df_full['is_weekend'] = df_full['day_of_week'].isin([5, 6]).astype(int)
    
    num_cols = [
        'hour', 'month', 'day_of_week', 'is_weekend',
        'Latitude', 'Longitude',
        'temperature_2m_mean', 'precipitation_sum',
        'rain_sum', 'snowfall_sum', 'windspeed_10m_max', 'sunshine_duration'
    ]
    cat_cols = ['District']
    
    # Drop rows missing crucial values
    df_full = df_full.dropna(subset=['Primary Type', 'Latitude', 'Longitude'])
    
    # Keep the top 8 crime classes
    top_crimes = df_full['Primary Type'].value_counts().nlargest(8).index
    df_full = df_full[df_full['Primary Type'].isin(top_crimes)].copy()
    print("Classes being trained:", list(top_crimes))
    
    X = df_full[num_cols + cat_cols]
    y = df_full['Primary Type']
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    
    # Final Model: Best performing architecture (XGBoost)
    final_estimator = XGBClassifier(
        n_estimators=250,
        max_depth=7,
        learning_rate=0.08,
        tree_method='hist',       # Optimized for larger multi-year datasets
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', final_estimator)
    ])
    
    print("\nTraining final XGBoost model on full 2015-2026 data...")
    final_pipeline.fit(X, y_encoded)
    
    # Save the pipeline and label encoder
    joblib.dump(final_pipeline, "final_crime_model.joblib")
    joblib.dump(label_encoder, "final_label_encoder.joblib")
    
    print("\nModel training complete.")
    print("Files saved:")
    print(" - final_crime_model.joblib")
    print(" - final_label_encoder.joblib")

if __name__ == "__main__":
    train_final()