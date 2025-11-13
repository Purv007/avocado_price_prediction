import matplotlib
matplotlib.use('Agg')  # Fix for Matplotlib threading issue

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64
import os
import holidays
from flask import send_from_directory
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load models and data
def load_models():
    try:
        xgb_model = joblib.load('models/xgboost_model.pkl')
        return xgb_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_data():
    try:
        df = pd.read_csv('data/avocado_features.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Load data and model
df = load_data()
xgb_model = load_models()

# US holidays for feature engineering
us_holidays = holidays.US()




@app.route('/price_over_time.png')
def serve_price_chart():
    return send_from_directory(os.getcwd(), 'price_over_time.png')

@app.route('/')
def index():
    if df is None or xgb_model is None:
        return render_template('error.html', message="Failed to load model or data")
    
    regions = sorted(df['region'].unique()) if df is not None else []
    types = df['type'].unique() if df is not None else []
    
    # Set default date range
    min_date = datetime.now().date() + timedelta(days=7)
    max_date = min_date + timedelta(weeks=52)
    
    # Get available years from data
    available_years = sorted(df['year'].unique()) if df is not None else []
    
    return render_template('index.html', 
                         regions=regions, 
                         types=types,
                         min_date=min_date,
                         max_date=max_date,
                         years=available_years)

@app.route('/predict', methods=['POST'])
def predict():
    if df is None or xgb_model is None:
        return render_template('error.html', message="Failed to load model or data")
    
    try:
        region = request.form['region']
        avocado_type = request.form['type']
        date_str = request.form['date']
        total_volume = float(request.form.get('total_volume', 100000))
        small_bags = float(request.form.get('small_bags', 50000))
        large_bags = float(request.form.get('large_bags', 30000))
        xlarge_bags = float(request.form.get('xlarge_bags', 20000))
        
        selected_date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Prepare features with additional attributes
        features = prepare_enhanced_features(
            region, avocado_type, selected_date, 
            total_volume, small_bags, large_bags, xlarge_bags
        )
        
        if features is None:
            return render_template('results.html', 
                                 error="No historical data available for the selected region and type.")
        
        # Convert to DataFrame with correct column order
        feature_cols = [
            'year', 'month', 'week', 'dayofweek', 'is_month_start', 'is_month_end',
            'month_sin', 'month_cos', 'price_lag_1', 'price_lag_4', 'price_lag_8', 'price_lag_12',
            'price_roll_mean_4', 'price_roll_mean_8', 'price_roll_mean_12',
            'price_roll_std_4', 'price_roll_std_8', 'price_roll_std_12',
            'volume_lag_1', 'volume_roll_mean_4', 'region_encoded', 'type_encoded',
            'is_holiday', 'season', 'total_volume', 'small_bags_ratio', 'large_bags_ratio',
            'xlarge_bags_ratio', 'bag_size_diversity'
        ]
        
        features_df = pd.DataFrame([features], columns=feature_cols)
        
        # Make prediction
        prediction = xgb_model.predict(features_df)[0]
        
        # Get historical data for chart
        region_data = df[(df['region'] == region) & (df['type'] == avocado_type)]
        
        # Create historical price chart
        chart_url = create_chart(region_data, region, avocado_type)
        
        # Get model performance metrics
        metrics = get_model_metrics()
        
        # Get feature importance
        feature_importance = get_feature_importance()
        
        return render_template('results.html', 
                             prediction=prediction,
                             region=region,
                             avocado_type=avocado_type,
                             date=selected_date.strftime('%Y-%m-%d'),
                             chart_url=chart_url,
                             metrics=metrics,
                             feature_importance=feature_importance,
                             input_data=request.form)
    
    except Exception as e:
        return render_template('results.html', error=str(e))

def prepare_enhanced_features(region, avocado_type, date, total_volume, small_bags, large_bags, xlarge_bags):
    # Filter data for the selected region and type
    region_data = df[(df['region'] == region) & (df['type'] == avocado_type)].copy()
    
    if region_data.empty:
        return None
    
    # Get the latest data point
    latest = region_data.iloc[-1].copy()
    
    # Create features for the prediction date
    prediction_features = latest[['year', 'month', 'week', 'dayofweek', 'is_month_start', 
                                 'is_month_end', 'month_sin', 'month_cos', 'region_encoded', 
                                 'type_encoded']].copy()
    
    # Update temporal features for the prediction date
    prediction_date = pd.to_datetime(date)
    prediction_features['year'] = prediction_date.year
    prediction_features['month'] = prediction_date.month
    prediction_features['week'] = prediction_date.isocalendar().week
    prediction_features['dayofweek'] = prediction_date.dayofweek
    prediction_features['is_month_start'] = int(prediction_date.is_month_start)
    prediction_features['is_month_end'] = int(prediction_date.is_month_end)
    
    # Update cyclical encoding for month
    prediction_features['month_sin'] = np.sin(2 * np.pi * prediction_date.month/12)
    prediction_features['month_cos'] = np.cos(2 * np.pi * prediction_date.month/12)
    
    # Holiday information
    prediction_features['is_holiday'] = int(date in us_holidays)
    
    # Season (1: Winter, 2: Spring, 3: Summer, 4: Fall)
    month = prediction_date.month
    if month in [12, 1, 2]:
        prediction_features['season'] = 1
    elif month in [3, 4, 5]:
        prediction_features['season'] = 2
    elif month in [6, 7, 8]:
        prediction_features['season'] = 3
    else:
        prediction_features['season'] = 4
    
    # Use the latest available lag features
    for col in ['price_lag_1', 'price_lag_4', 'price_lag_8', 'price_lag_12',
                'price_roll_mean_4', 'price_roll_mean_8', 'price_roll_mean_12',
                'price_roll_std_4', 'price_roll_std_8', 'price_roll_std_12',
                'volume_lag_1', 'volume_roll_mean_4']:
        prediction_features[col] = latest[col]
    
    # Add new features from user input
    prediction_features['total_volume'] = total_volume
    prediction_features['small_bags_ratio'] = small_bags / total_volume if total_volume > 0 else 0
    prediction_features['large_bags_ratio'] = large_bags / total_volume if total_volume > 0 else 0
    prediction_features['xlarge_bags_ratio'] = xlarge_bags / total_volume if total_volume > 0 else 0
    
    # Bag size diversity (entropy-like measure)
    ratios = [small_bags/total_volume, large_bags/total_volume, xlarge_bags/total_volume]
    ratios = [r for r in ratios if r > 0]
    bag_size_diversity = -sum(r * np.log(r) for r in ratios) if ratios else 0
    prediction_features['bag_size_diversity'] = bag_size_diversity
    
    return prediction_features

def create_chart(region_data, region, avocado_type):
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Price chart
    ax1.plot(region_data['date'], region_data['AveragePrice'], 
             color='#4CAF50', linewidth=3, marker='o', markersize=4)
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Price ($)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Historical Avocado Prices in {region} ({avocado_type})', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Volume chart
    ax2.plot(region_data['date'], region_data['Total Volume'], 
             color='#2196F3', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Volume', fontsize=12, fontweight='bold')
    ax2.set_title('Sales Volume Over Time', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot to bytes
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100, facecolor='#f8f9fa')
    img.seek(0)
    
    # Encode to base64
    chart_url = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{chart_url}"

def get_model_metrics():
    # Load metrics from JSON files if they exist
    metrics = {
        'baseline': {'mae': 0.3482, 'rmse': 0.4533},
        'prophet': {'mae': 0.3035, 'rmse': 0.3880},
        'xgboost': {'mae': 0.0611, 'rmse': 0.0931}
    }
    return metrics

def get_feature_importance():
    # Get feature importance from the model
    if hasattr(xgb_model, 'feature_importances_'):
        feature_names = xgb_model.feature_names_in_
        importance_scores = xgb_model.feature_importances_
        
        # Create a list of (feature, importance) pairs
        feature_importance = list(zip(feature_names, importance_scores))
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 10 features
        return feature_importance[:10]
    
    return []

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)