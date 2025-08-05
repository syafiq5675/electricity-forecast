"""
Feature Engineering Pipeline
Professional implementation for electricity consumption forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime
import holidays

def load_raw_datasets():
    """Load and clean raw energy and weather datasets"""
    print("Loading raw datasets...")
    
    # Load energy data
    energy_df = pd.read_csv('datasets/energy_dataset.csv')
    energy_df['time'] = pd.to_datetime(energy_df['time'], utc=True).dt.tz_localize(None)
    
    # Load weather data  
    weather_df = pd.read_csv('datasets/weather_features.csv')
    weather_df['dt_iso'] = pd.to_datetime(weather_df['dt_iso'], utc=True).dt.tz_localize(None)
    
    print(f"Energy data: {len(energy_df):,} rows")
    print(f"Weather data: {len(weather_df):,} rows")
    
    return energy_df, weather_df

def create_national_weather(weather_df):
    """Create national weather representation by averaging across cities"""
    print("Creating national weather representation...")
    
    # Group by timestamp and calculate mean across all cities
    national_weather = weather_df.groupby('dt_iso').agg({
        'temp': 'mean',
        'humidity': 'mean', 
        'pressure': 'mean'
    }).reset_index()
    
    national_weather.rename(columns={
        'dt_iso': 'time',
        'temp': 'temp_celsius'
    }, inplace=True)
    
    # Convert temperature from Kelvin to Celsius
    national_weather['temp_celsius'] = national_weather['temp_celsius'] - 273.15
    
    print(f"National weather created: {len(national_weather):,} timestamps")
    print(f"Temperature range: {national_weather['temp_celsius'].min():.1f}°C to {national_weather['temp_celsius'].max():.1f}°C")
    
    return national_weather

def merge_energy_weather(energy_df, weather_df):
    """Merge energy and weather data with temporal alignment"""
    print("Merging energy and weather data...")
    
    # Remove data leakage - forecast columns contain future information
    leakage_columns = [col for col in energy_df.columns if 'forecast' in col.lower()]
    energy_clean = energy_df.drop(columns=leakage_columns)
    
    # Merge using time-based join with tolerance
    merged_df = pd.merge_asof(
        energy_clean.sort_values('time'),
        weather_df.sort_values('time'), 
        on='time',
        tolerance=pd.Timedelta('1 hour'),
        direction='nearest'
    )
    
    # Clean merged data
    merged_df = merged_df.dropna(subset=['total load actual', 'temp_celsius'])
    merged_df = merged_df.sort_values('time').reset_index(drop=True)
    
    print(f"Merged dataset: {len(merged_df):,} rows")
    
    return merged_df

def create_target_and_features(df):
    """Create target variable and engineered features"""
    print("Engineering features...")
    
    # Create target variable (consumption in kWh)
    df['consumption_kWh'] = df['total load actual'] * 1000  # MW to kWh
    
    # Create datetime features
    df['datetime'] = df['time']
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    df['is_peak_hours'] = ((df['hour'] >= 17) & (df['hour'] <= 21)).astype(int)
    
    # Create holiday feature
    spain_holidays = holidays.Spain(years=range(2015, 2019))
    df['is_holiday'] = df['time'].dt.date.apply(lambda x: x in spain_holidays).astype(int)
    
    # Create weather intelligence features
    df['temp_hot'] = (df['temp_celsius'] >= 25).astype(int)
    
    print(f"Target range: {df['consumption_kWh'].min():,.0f} to {df['consumption_kWh'].max():,.0f} kWh")
    print(f"Holiday hours: {df['is_holiday'].sum():,}")
    
    return df

def create_lag_features(df):
    """Create historical lag features for time series forecasting"""
    print("Creating lag features...")
    
    # Sort by time to ensure proper lag calculation
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Create lag features (essential for time series)
    df['consumption_lag_24h'] = df['consumption_kWh'].shift(24)   # Previous day same hour
    df['consumption_lag_168h'] = df['consumption_kWh'].shift(168) # Previous week same hour
    
    # Remove rows with missing lag values
    df = df.dropna(subset=['consumption_lag_24h', 'consumption_lag_168h'])
    
    print(f"Dataset after lag creation: {len(df):,} rows")
    
    return df

def select_final_features(df):
    """Select final feature set for model training"""
    print("Selecting final features...")
    
    # Define the final feature set (12 features used in model training)
    final_features = [
        'datetime',
        'consumption_kWh',           # Target variable
        'hour',                      # Time features
        'day_of_week',
        'month', 
        'temp_celsius',              # Weather features
        'humidity',
        'is_holiday',                # Binary features
        'is_weekend',
        'is_business_hours',
        'is_peak_hours',
        'temp_hot',
        'consumption_lag_24h',       # Lag features  
        'consumption_lag_168h'
    ]
    
    # Select only final features
    final_df = df[final_features].copy()
    
    print(f"Final dataset: {len(final_df):,} rows, {len(final_features):,} features")
    print(f"Date range: {final_df['datetime'].min()} to {final_df['datetime'].max()}")
    
    return final_df

def save_processed_dataset(df, filepath):
    """Save processed dataset to CSV"""
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to: {filepath}")

def main():
    """Main feature engineering pipeline"""
    print("ELECTRICITY CONSUMPTION FORECASTING - FEATURE ENGINEERING")
    print("=" * 65)
    
    try:
        # Load raw data
        energy_df, weather_df = load_raw_datasets()
        
        # Create national weather representation
        national_weather = create_national_weather(weather_df)
        
        # Merge datasets
        merged_df = merge_energy_weather(energy_df, national_weather)
        
        # Create features
        featured_df = create_target_and_features(merged_df)
        
        # Create lag features
        lagged_df = create_lag_features(featured_df)
        
        # Select final features
        final_df = select_final_features(lagged_df)
        
        # Save processed dataset
        save_processed_dataset(final_df, 'datasets/processed_data_fixed.csv')
        
        print("=" * 65)
        print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
        print("=" * 65)
        print(f"Final dataset: {len(final_df):,} rows × {len(final_df.columns):,} columns")
        print(f"Target variable: consumption_kWh")
        print(f"Feature count: {len(final_df.columns) - 2}")  # Exclude datetime and target
        print(f"Ready for machine learning training")
        print()
        print("Next step: python src/model.py")
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()