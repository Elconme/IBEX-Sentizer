import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import glob
import os
import tensorflow as tf
import random

LOOKBACK_VALUES = [1, 5, 8, 10, 16, 20]
BASE_MODELS_DIR = 'models'
BASE_PLOTS_OUTPUT_DIR = 'plots'
TARGET_COL = 'TRDPRC_1'

# Sets seed for reproducibility
def set_seed(seed_value):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

# Create sequences of data for time series prediction
def create_sequences(X, y, lookback=1):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# Prepare full data and scale it
def prepare_full_data_and_scale(data_df, lookback, use_sentiment, sentiment_model_type):
    data = data_df.copy()

    # --- Feature Engineering (MUST be IDENTICAL to training script) ---
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data['day_of_week'] = data['Date'].dt.dayofweek
        data['month'] = data['Date'].dt.month
        data = data.sort_values(by='Date').reset_index(drop=True)
    else:
        raise ValueError("Missing 'Date' column for calendar feature extraction. Please ensure your CSV has a 'Date' column.")

    data['EMA_9'] = data[TARGET_COL].ewm(span=9, adjust=False).mean().shift(1)
    data['SMA_5'] = data[TARGET_COL].rolling(window=5).mean().shift(1)
    data['SMA_10'] = data[TARGET_COL].rolling(window=10).mean().shift(1)
    data['SMA_15'] = data[TARGET_COL].rolling(window=15).mean().shift(1)
    data['SMA_30'] = data[TARGET_COL].rolling(window=30).mean().shift(1)
    data['SMA_50'] = data[TARGET_COL].rolling(window=50).mean().shift(1)
    data['Returns'] = data[TARGET_COL].pct_change()
    data['Open_Lag1'] = data['OPEN_PRC'].shift(1)
    data['High_Lag1'] = data['HIGH_1'].shift(1)
    data['Low_Lag1'] = data['LOW_1'].shift(1)
    data['Volume_Lag1'] = data['ACVOL_UNS'].shift(1)

    data['Open_Close_Ratio_Lag1'] = (data[TARGET_COL].shift(1) - data['Open_Lag1']) / data['Open_Lag1']
    data['High_Low_Range_Lag1'] = data['High_Lag1'] - data['Low_Lag1']
    data['Close_High_Ratio_Lag1'] = (data['High_Lag1'] - data[TARGET_COL].shift(1)) / (data['High_Lag1'] - data['Low_Lag1'] + 1e-6)
    data['Volume_Change_Lag1'] = data['ACVOL_UNS'].pct_change().shift(1)

    data['EMA_9_Lag2'] = data['EMA_9'].shift(1)
    data['Returns_Lag2'] = data['Returns'].shift(1)
    data['High_Lag2'] = data['HIGH_1'].shift(2)
    data['Low_Lag2'] = data['LOW_1'].shift(2)
    data['Open_Lag2'] = data['OPEN_PRC'].shift(2)
    data['Volume_Lag2'] = data['ACVOL_UNS'].shift(2)

    data['Volatility_5'] = data[TARGET_COL].rolling(window=5).std().shift(1)
    data['Time_Index'] = np.arange(len(data)) / len(data)

    if use_sentiment:
        if sentiment_model_type == 'finbert':
            sentiment_cols = ['daily_sentiment_score']
            for col in sentiment_cols:
                if col not in data.columns:
                    raise ValueError(f"Missing '{col}' column for FinBERT sentiment. Cannot use sentiment analysis. Please ensure your CSV has a 'daily_sentiment_score' column for sentiment models.")
            data['daily_sentiment_score_Lag1'] = data['daily_sentiment_score'].shift(1)
            data['sentiment_trend_3'] = data['daily_sentiment_score'].rolling(3).mean().shift(1)
            data['sentiment_volatility_3'] = data['daily_sentiment_score'].rolling(3).std().shift(1)
            data['sentiment_ma_5'] = data['daily_sentiment_score'].rolling(5).mean().shift(1)
            data['sentiment_diff_1'] = data['daily_sentiment_score'].diff().shift(1)
        else:
            raise ValueError(f"Unexpected sentiment model type '{sentiment_model_type}' when use_sentiment is True. Expected 'finbert'.")

        main_sentiment_lag1_col = None
        if sentiment_model_type == 'finbert' and 'daily_sentiment_score_Lag1' in data.columns:
            main_sentiment_lag1_col = 'daily_sentiment_score_Lag1'

        if main_sentiment_lag1_col and not data[main_sentiment_lag1_col].isnull().all():
            data['volatility_x_sentiment_lag1'] = data['Volatility_5'] * data[main_sentiment_lag1_col]
            data['returns_x_sentiment_lag1'] = data['Returns'] * data[main_sentiment_lag1_col]
            data['volume_change_x_sentiment_lag1'] = data['Volume_Change_Lag1'] * data[main_sentiment_lag1_col]
        else:
            if use_sentiment:
                print(f"Warning: Main sentiment column '{main_sentiment_lag1_col}' not found or all NaN for interaction terms. Skipping interaction terms. Ensure your data has 'daily_sentiment_score' if using sentiment.")

    data['Target'] = data[TARGET_COL].shift(-1)

    dates_full = data['Date'].copy()

    data = data.dropna()
    dates_full = dates_full[data.index]

    if len(data) < lookback:
        raise ValueError(f"Not enough data after dropping NaNs for lookback {lookback}. Data length: {len(data)}")

    feature_cols = ['EMA_9', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_30', 'SMA_50', 'Returns',
                    'Open_Lag1', 'High_Lag1', 'Low_Lag1', 'Volume_Lag1',
                    'Open_Close_Ratio_Lag1', 'High_Low_Range_Lag1', 'Close_High_Ratio_Lag1', 'Volume_Change_Lag1',
                    'EMA_9_Lag2', 'Returns_Lag2', 'High_Lag2', 'Low_Lag2',
                    'Open_Lag2', 'Volume_Lag2', 'Volatility_5', 'Time_Index',
                    'day_of_week', 'month']

    if use_sentiment:
        if sentiment_model_type == 'finbert':
            feature_cols += ['daily_sentiment_score_Lag1', 'sentiment_trend_3', 'sentiment_volatility_3',
                             'sentiment_ma_5', 'sentiment_diff_1']

        if 'volatility_x_sentiment_lag1' in data.columns and not data['volatility_x_sentiment_lag1'].isnull().all():
            feature_cols += ['volatility_x_sentiment_lag1', 'returns_x_sentiment_lag1', 'volume_change_x_sentiment_lag1']
        else:
            if use_sentiment:
                missing_interaction_features = [f for f in ['volatility_x_sentiment_lag1', 'returns_x_sentiment_lag1', 'volume_change_x_sentiment_lag1'] if f in feature_cols and f not in data.columns]
                if missing_interaction_features:
                    print(f"WARNING: Interaction terms {missing_interaction_features} were specified but not found in processed data. This could be due to missing daily_sentiment_score. Check data consistency.")

    missing_cols = [col for col in feature_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}. Please check your data and ensure all necessary columns are present for the chosen model type.")

    scaler = StandardScaler()
    X_scaled_full = scaler.fit_transform(data[feature_cols])
    y_full = data['Target'].values

    dates_seq_full = dates_full[lookback:len(data)].reset_index(drop=True)
    X_seq_full, y_seq_full = create_sequences(X_scaled_full, y_full, lookback=lookback)

    if len(X_seq_full) == 0:
        raise ValueError(f"No sequences created after scaling for lookback {lookback}. Check data length and lookback value.")

    return X_seq_full, y_seq_full, dates_seq_full

# Function to plot predictions
def plot_predictions(model, X_test, y_test, test_dates, stock_name, lookback, use_sentiment, sentiment_model_type, seed, output_dir):
    y_pred = model.predict(X_test).flatten()

    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_test, label='Actual Price', color='blue', alpha=0.8)
    plt.plot(test_dates, y_pred, label='Predicted Price', color='red', linestyle='--', alpha=0.7)

    clean_stock_name = stock_name.split('_')[0]

    sentiment_str = f"with FinBERT Sentiment" if use_sentiment else "No Sentiment"
    plot_title = f"{clean_stock_name} Stock Price Prediction\n(Lookback={lookback}, {sentiment_str})"

    plt.title(plot_title)
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    sentiment_file_str = f"_finbert_sentiment" if use_sentiment else "_no_sentiment"
    plot_filename = f"{stock_name}_lookback{lookback}{sentiment_file_str}_predictions.png"
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()
    print(f"Plot saved to: {os.path.join(output_dir, plot_filename)}")


if __name__ == "__main__":
    for LOOKBACK_VALUE in LOOKBACK_VALUES:
        MODELS_BASE_DIR = os.path.join(BASE_MODELS_DIR, f'lookback{LOOKBACK_VALUE}')
        PLOTS_OUTPUT_DIR = os.path.join(BASE_PLOTS_OUTPUT_DIR, f'lookback{LOOKBACK_VALUE}')

        # Check if the models directory for the current lookback value exists
        if not os.path.exists(MODELS_BASE_DIR):
            print(f"Skipping lookback value {LOOKBACK_VALUE}: Models directory '{MODELS_BASE_DIR}' not found.")
            continue # Skip to the next LOOKBACK_VALUE

        os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
        print(f"\n--- Processing models for LOOKBACK VALUE: {LOOKBACK_VALUE} ---")
        print(f"Models directory: {MODELS_BASE_DIR}")
        print(f"Plots output directory: {PLOTS_OUTPUT_DIR}")

        saved_stock_dirs = [d for d in os.listdir(MODELS_BASE_DIR) if os.path.isdir(os.path.join(MODELS_BASE_DIR, d))]

        if not saved_stock_dirs:
            print(f"No stock directories found in {MODELS_BASE_DIR}. Skipping this lookback value.")
            continue

        for stock_name in saved_stock_dirs:
            stock_model_path = os.path.join(MODELS_BASE_DIR, stock_name)
            print(f"\n--- Processing saved models for {stock_name} (Lookback {LOOKBACK_VALUE}) ---")

            csv_file_path_list = glob.glob(os.path.join('data', f'{stock_name}.csv'))

            if not csv_file_path_list:
                csv_file_path_list = glob.glob(os.path.join('data', f'{stock_name}_*.csv'))

            if not csv_file_path_list:
                print(f"Error: Original data file for {stock_name} not found in 'data/'. Skipping this stock.")
                continue
            original_data = pd.read_csv(csv_file_path_list[0])
            print(f"Loaded original data for {stock_name}. Shape: {original_data.shape}")

            keras_model_files = glob.glob(os.path.join(stock_model_path, '*.keras'))

            if not keras_model_files:
                print(f"No .keras models found in {stock_model_path}. Skipping this stock.")
                continue

            for model_file in keras_model_files:
                model_basename = os.path.basename(model_file)
                print(f"Attempting to process model: {model_basename}")

                model_basename_lower = model_basename.lower()

                if "no_sentiment" in model_basename_lower:
                    use_sentiment = False
                    sentiment_type_to_pass = "N/A"
                else:
                    use_sentiment = True
                    sentiment_type_to_pass = "finbert"

                seed_match = [part for part in model_basename.split('_') if 'seed' in part]
                if seed_match:
                    seed = int(seed_match[0].replace('seed', ''))
                else:
                    print(f"WARNING: Could not extract seed from filename: {model_basename}. Using a default of 42.")
                    seed = 42

                print(f"DEBUG: Calling prepare_full_data_and_scale with:")
                print(f"       use_sentiment={use_sentiment}")
                print(f"       sentiment_model_type={sentiment_type_to_pass}")

                try:
                    X_seq_full, y_seq_full, dates_seq_full = prepare_full_data_and_scale(
                        original_data.copy(), LOOKBACK_VALUE, use_sentiment, sentiment_type_to_pass
                    )
                    print(f"Full scaled sequences created for {model_basename}. Total sequences: {len(X_seq_full)}")

                    if len(X_seq_full) == 0:
                        print(f"WARNING: No sequences available for {model_basename} after data prep. Skipping.")
                        continue

                    set_seed(seed)
                    split_point = int(len(X_seq_full) * 0.8)
                    recreated_X_test = X_seq_full[split_point:]
                    recreated_y_test = y_seq_full[split_point:]

                    recreated_test_dates = dates_seq_full[split_point:]

                    if len(recreated_X_test) == 0:
                        print(f"WARNING: Recreated test set is empty for {model_basename}. Skipping.")
                        continue

                    loaded_model = load_model(model_file)
                    print(f"Loaded model from: {os.path.basename(model_file)}")

                    plot_predictions(loaded_model, recreated_X_test, recreated_y_test,
                                     recreated_test_dates,
                                     stock_name, LOOKBACK_VALUE, use_sentiment, sentiment_type_to_pass, seed, PLOTS_OUTPUT_DIR)

                except Exception as e:
                    print(f"Error processing model {model_basename}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    print("\n--- All plotting complete ---")