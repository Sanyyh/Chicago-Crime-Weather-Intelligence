import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_preprocess(filepath, start_year=2025, end_year=2026, top_n_classes=8):
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Parse date and filter year range
    df['parsed_date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['parsed_date'])
    df['year'] = df['parsed_date'].dt.year
    
    df_filtered = df[(df['year'] >= start_year) & (df['year'] <= end_year)].copy()
    print(f"Records found for {start_year}-{end_year}: {len(df_filtered)}")
    
    # 2. Extract temporal features
    df_filtered['month'] = df_filtered['parsed_date'].dt.month
    df_filtered['day_of_week'] = df_filtered['parsed_date'].dt.dayofweek
    df_filtered['is_weekend'] = df_filtered['day_of_week'].isin([5, 6]).astype(int)
    
    # 3. Clean Location & Weather columns
    num_cols = [
        'hour', 'month', 'day_of_week', 'is_weekend',
        'Latitude', 'Longitude',
        'temperature_2m_mean', 'precipitation_sum',
        'rain_sum', 'snowfall_sum', 'windspeed_10m_max', 'sunshine_duration'
    ]
    
    cat_cols = ['District']
    
    # Drop rows where critical location or target data is missing
    df_filtered = df_filtered.dropna(subset=['Primary Type', 'Latitude', 'Longitude'])
    
    # Keep top N classes to maintain class balance and model stability
    top_crimes = df_filtered['Primary Type'].value_counts().nlargest(top_n_classes).index
    df_filtered = df_filtered[df_filtered['Primary Type'].isin(top_crimes)].copy()
    print(f"Top {top_n_classes} crime classes kept: {list(top_crimes)}")

    X = df_filtered[num_cols + cat_cols]
    y = df_filtered['Primary Type']
    
    return X, y, num_cols, cat_cols

def evaluate_models():
    filepath = "crime_weather_merged.csv"
    X, y, num_cols, cat_cols = load_and_preprocess(filepath, start_year=2025, end_year=2026)
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split 80% train, 20% test (stratified by crime type)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    # Preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    
    # Define 4 candidate models
    models = {
        "Multinomial Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1),
        "HistGradientBoosting Classifier": HistGradientBoostingClassifier(max_iter=150, random_state=42),
        "XGBoost Classifier": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
    }
    
    results = []
    
    print("\n" + "="*60)
    print("STARTING MODEL BENCHMARKING (2025 - 2026)")
    print("="*60)
    
    best_score = -1
    best_model_name = None
    
    for name, clf in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Weighted F1': weighted_f1,
            'Macro F1': macro_f1
        })
        
        print(f"Results for {name}:")
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")
        print(f"  Macro F1:    {macro_f1:.4f}")
        
        if weighted_f1 > best_score:
            best_score = weighted_f1
            best_model_name = name
            
    # Summary Table
    df_results = pd.DataFrame(results).sort_values(by='Weighted F1', ascending=False)
    print("\n" + "="*60)
    print("LEADERBOARD SUMMARY")
    print("="*60)
    print(df_results.to_string(index=False))
    print(f"\nBest Selected Model: {best_model_name} (Weighted F1: {best_score:.4f})")
    
    # Save label encoder for future reference
    joblib.dump(label_encoder, "label_encoder.joblib")
    
    return best_model_name

if __name__ == "__main__":
    evaluate_models()