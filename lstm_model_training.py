import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
import glob
import os
import tensorflow as tf
import random
import itertools
import time

def set_seed(seed_value):
    """
    Sets the random seed for reproducibility across different libraries.
    """
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

def create_sequences(X, y, lookback=1):
    """
    Creates sequences of data for LSTM input.
    """
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def build_lstm_model(input_shape, units, dropout_rate, optimizer_name, activation_function, learning_rate):
    """
    Builds a Keras LSTM model with dynamic hyperparameters.
    """
    if optimizer_name == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'Nadam':
        optimizer = Nadam(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape, activation=activation_function),
        Dropout(dropout_rate),
        LSTM(units // 2 if units > 32 else units, activation=activation_function),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mse')
    return model

def prepare_and_train_lstm(data, lookback=1, use_sentiment=False, sentiment_model_type='finbert',
                           units=64, dropout_rate=0.2, optimizer_name='Adam',
                           activation_function='tanh', learning_rate=0.001,
                           epochs=100, batch_size=16, verbose=0, plot_results=False):
    """
    Prepares data, trains an LSTM model with specified hyperparameters, and evaluates its performance.
    
    Returns:
        tuple: (rmse, mae, r2, val_loss, model) evaluation metrics and the trained model.
    """
    data = data.copy()

    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data['day_of_week'] = data['Date'].dt.dayofweek
        data['month'] = data['Date'].dt.month
        data = data.sort_values(by='Date').reset_index(drop=True)
    else:
        raise ValueError("Missing 'Date' column for calendar feature extraction. Please ensure your CSV has a 'Date' column.")

    data['EMA_9'] = data['TRDPRC_1'].ewm(span=9, adjust=False).mean().shift(1)
    data['SMA_5'] = data['TRDPRC_1'].rolling(window=5).mean().shift(1)
    data['SMA_10'] = data['TRDPRC_1'].rolling(window=10).mean().shift(1)
    data['SMA_15'] = data['TRDPRC_1'].rolling(window=15).mean().shift(1)
    data['SMA_30'] = data['TRDPRC_1'].rolling(window=30).mean().shift(1)
    data['SMA_50'] = data['TRDPRC_1'].rolling(window=50).mean().shift(1)
    data['Returns'] = data['TRDPRC_1'].pct_change()
    data['Open_Lag1'] = data['OPEN_PRC'].shift(1)
    data['High_Lag1'] = data['HIGH_1'].shift(1)
    data['Low_Lag1'] = data['LOW_1'].shift(1)
    data['Volume_Lag1'] = data['ACVOL_UNS'].shift(1)

    data['Open_Close_Ratio_Lag1'] = (data['TRDPRC_1'].shift(1) - data['Open_Lag1']) / data['Open_Lag1']
    data['High_Low_Range_Lag1'] = data['High_Lag1'] - data['Low_Lag1']
    data['Close_High_Ratio_Lag1'] = (data['High_Lag1'] - data['TRDPRC_1'].shift(1)) / (data['High_Lag1'] - data['Low_Lag1'] + 1e-6)
    data['Volume_Change_Lag1'] = data['ACVOL_UNS'].pct_change().shift(1)

    data['EMA_9_Lag2'] = data['EMA_9'].shift(1)
    data['Returns_Lag2'] = data['Returns'].shift(1)
    data['High_Lag2'] = data['HIGH_1'].shift(2)
    data['Low_Lag2'] = data['LOW_1'].shift(2)
    data['Open_Lag2'] = data['OPEN_PRC'].shift(2)
    data['Volume_Lag2'] = data['ACVOL_UNS'].shift(2)

    data['Volatility_5'] = data['TRDPRC_1'].rolling(window=5).std().shift(1)
    data['Time_Index'] = np.arange(len(data)) / len(data)

    if use_sentiment:
        if sentiment_model_type == 'finbert':
            sentiment_cols = ['daily_sentiment_score']
            for col in sentiment_cols:
                if col not in data.columns:
                    raise ValueError(f"Missing '{col}' column for FinBERT sentiment. Cannot use sentiment analysis.")
            data['daily_sentiment_score_Lag1'] = data['daily_sentiment_score'].shift(1)
            data['sentiment_trend_3'] = data['daily_sentiment_score'].rolling(3).mean().shift(1)
            data['sentiment_volatility_3'] = data['daily_sentiment_score'].rolling(3).std().shift(1)
            data['sentiment_ma_5'] = data['daily_sentiment_score'].rolling(5).mean().shift(1)
            data['sentiment_diff_1'] = data['daily_sentiment_score'].diff().shift(1)
        elif sentiment_model_type == 'fingpt':
            sentiment_cols = ['positive_sentiment_fingpt', 'negative_sentiment_fingpt', 'neutral_sentiment_fingpt']
            for col in sentiment_cols:
                if col not in data.columns:
                    raise ValueError(f"Missing '{col}' column for FinGPT sentiment. Cannot use sentiment analysis.")
            data['positive_sentiment_fingpt_Lag1'] = data['positive_sentiment_fingpt'].shift(1)
            data['negative_sentiment_fingpt_Lag1'] = data['negative_sentiment_fingpt'].shift(1)
            data['neutral_sentiment_fingpt_Lag1'] = data['neutral_sentiment_fingpt'].shift(1)
            data['fingpt_pos_neg_diff_Lag1'] = (data['positive_sentiment_fingpt'] - data['negative_sentiment_fingpt']).shift(1)
            data['fingpt_pos_neutral_ratio_Lag1'] = (data['positive_sentiment_fingpt'] / (data['neutral_sentiment_fingpt'] + 1e-6)).shift(1)
        else:
            raise ValueError(f"Unknown sentiment model: {sentiment_model_type}. Choose 'finbert' or 'fingpt'.")

        main_sentiment_lag1_col = None
        if sentiment_model_type == 'finbert' and 'daily_sentiment_score_Lag1' in data.columns:
            main_sentiment_lag1_col = 'daily_sentiment_score_Lag1'
        elif sentiment_model_type == 'fingpt' and 'positive_sentiment_fingpt_Lag1' in data.columns:
            main_sentiment_lag1_col = 'positive_sentiment_fingpt_Lag1'

        if main_sentiment_lag1_col and not data[main_sentiment_lag1_col].isnull().all():
            data['volatility_x_sentiment_lag1'] = data['Volatility_5'] * data[main_sentiment_lag1_col]
            data['returns_x_sentiment_lag1'] = data['Returns'] * data[main_sentiment_lag1_col]
            data['volume_change_x_sentiment_lag1'] = data['Volume_Change_Lag1'] * data[main_sentiment_lag1_col]
        else:
            if use_sentiment:
                print(f"Warning: Main sentiment column '{main_sentiment_lag1_col}' not found or all NaN for interaction terms. Skipping interaction terms.")

    data['Target'] = data['TRDPRC_1'].shift(-1)
    data = data.dropna()

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
        elif sentiment_model_type == 'fingpt':
            feature_cols += ['positive_sentiment_fingpt_Lag1', 'negative_sentiment_fingpt_Lag1',
                             'neutral_sentiment_fingpt_Lag1', 'fingpt_pos_neg_diff_Lag1', 'fingpt_pos_neutral_ratio_Lag1']
        if 'volatility_x_sentiment_lag1' in data.columns and not data['volatility_x_sentiment_lag1'].isnull().all():
            feature_cols += ['volatility_x_sentiment_lag1', 'returns_x_sentiment_lag1', 'volume_change_x_sentiment_lag1']

    missing_cols = [col for col in feature_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}. Please check your data.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[feature_cols])
    y = data['Target'].values

    X_seq, y_seq = create_sequences(X_scaled, y, lookback=lookback)

    if len(X_seq) == 0:
        raise ValueError(f"No sequences created for lookback {lookback}. Check data length and lookback value.")

    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(f"Train or test set is empty after splitting. X_train: {len(X_train)}, X_test: {len(X_test)}")

    model = build_lstm_model(
        input_shape=(lookback, X_train.shape[2]),
        units=units,
        dropout_rate=dropout_rate,
        optimizer_name=optimizer_name,
        activation_function=activation_function,
        learning_rate=learning_rate
    )

    early_stop = EarlyStopping(patience=7, restore_best_weights=True, monitor='val_loss')
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0, min_lr=1e-6)

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epochs, batch_size=batch_size, verbose=verbose,
                        callbacks=[early_stop, lr_scheduler])

    y_pred = model.predict(X_test).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    if verbose > 0:
        print(f"\n MAE: {mae:.4f}")
        print(f" MSE: {mse:.4f}")
        print(f"√MSE (RMSE): {rmse:.4f}")
        print(f" R²:  {r2:.4f}")

    if plot_results:
        plt.figure(figsize=(12, 5))
        plt.plot(y_test, label='Actual Price')
        plt.plot(y_pred, label='Predicted Price')
        plt.legend()
        plt.title(f"LSTM - Actual vs Predicted Stock Price\nLookback: {lookback}, Sentiment: {use_sentiment}, Model: {sentiment_model_type}\nUnits: {units}, Dropout: {dropout_rate}, Optimizer: {optimizer_name}, LR: {learning_rate}, Batch: {batch_size}")
        plt.grid()
        plt.tight_layout()
        plt.show()

    return rmse, mae, r2, history.history['val_loss'][-1], model

# Function to run random search for hyperparameter tuning
def run_random_search(data, num_iterations, lookback_value, use_sentiment, sentiment_model_type, seed):
    set_seed(seed)
    
    param_grid = {
        'units': [32, 64, 128, 256],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
        'optimizer_name': ['Adam', 'Nadam', 'RMSprop', 'SGD'],
        'activation_function': ['relu', 'tanh', 'selu', 'elu', 'swish'],
        'learning_rate': [0.001, 0.01, 0.1],
        'epochs': [50, 100, 150],
        'batch_size': [16, 32, 64]
    }

    param_combinations = []
    for _ in range(num_iterations):
        combo = {k: random.choice(v) for k, v in param_grid.items()}
        param_combinations.append(combo)

    results = []
    best_model_for_search = None
    best_r2_for_search = -np.inf

    print(f"\n--- Starting Random Search for {num_iterations} iterations (Lookback: {lookback_value}, Sentiment: {use_sentiment}, Model: {sentiment_model_type}) ---")
    
    for i, params in enumerate(param_combinations):
        print(f"\n    Iteration {i+1}/{num_iterations} with params: {params}")
        try:
            rmse, mae, r2, val_loss, current_model = prepare_and_train_lstm(
                data=data.copy(),
                lookback=lookback_value,
                use_sentiment=use_sentiment,
                sentiment_model_type=sentiment_model_type,
                units=params['units'],
                dropout_rate=params['dropout_rate'],
                optimizer_name=params['optimizer_name'],
                activation_function=params['activation_function'],
                learning_rate=params['learning_rate'],
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                verbose=0,
                plot_results=False
            )
            result = {
                'Units': params['units'],
                'Dropout': params['dropout_rate'],
                'Optimizer': params['optimizer_name'],
                'Activation': params['activation_function'],
                'Learning Rate': params['learning_rate'],
                'Epochs': params['epochs'],
                'Batch Size': params['batch_size'],
                'RMSE': rmse,
                'MAE': mae,
                'R^2': r2,
                'Validation Loss': val_loss
            }
            results.append(result)
            
            # Keep track of the best model found during this random search
            if r2 > best_r2_for_search:
                best_r2_for_search = r2
                best_model_for_search = current_model
                
            print(f"    Iteration {i+1} completed. R^2: {r2:.4f}, RMSE: {rmse:.4f}")
        except ValueError as ve:
            print(f"    Iteration {i+1} skipped due to data error: {ve}")
        except Exception as e:
            print(f"    Iteration {i+1} skipped due to unexpected error: {e}")
    
    return pd.DataFrame(results), best_model_for_search

if __name__ == "__main__":
    fixed_lookback = 8
    num_random_search_iterations = 5
    seeds_to_test = [10,20,30,40]
    sentiment_models_to_test = ['finbert']

    final_experiment_results = []
    
    models_base_dir = f'models/lookback{fixed_lookback}'
    os.makedirs(models_base_dir, exist_ok=True)

    csv_files = glob.glob('data/*_10*.csv')

    if not csv_files:
        print("No CSV files found in the 'data/' directory matching '*_10*.csv'. Please ensure your data files are present.")
    else:
        for file_path in csv_files:
            stock_name = os.path.basename(file_path).replace('.csv', '')
            print(f"\n===== Processing stock: {stock_name} =====")

            try:
                data = pd.read_csv(file_path)
                print(f"Successfully loaded {file_path}. Shape: {data.shape}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}. Skipping this file.")
                continue

            for seed in seeds_to_test:
                print(f"\n--- Running experiments for {stock_name} with Seed: {seed} ---")

                # Scenario 1: No Sentiment
                print(f"\nScenario: No Sentiment (Lookback={fixed_lookback})")
                no_sentiment_rs_results, best_no_sentiment_model = run_random_search(
                    data=data.copy(),
                    num_iterations=num_random_search_iterations,
                    lookback_value=fixed_lookback,
                    use_sentiment=False,
                    sentiment_model_type='N/A',
                    seed=seed
                )

                if not no_sentiment_rs_results.empty:
                    best_no_sentiment_run = no_sentiment_rs_results.loc[no_sentiment_rs_results['R^2'].idxmax()]
                    final_experiment_results.append({
                        'Stock Name': stock_name,
                        'Lookback': fixed_lookback,
                        'Sentiment Used': "No",
                        'Sentiment Model': "N/A",
                        'Best R^2': best_no_sentiment_run['R^2'],
                        'Best RMSE': best_no_sentiment_run['RMSE'],
                        'Best MAE': best_no_sentiment_run['MAE'],
                        'Seed': seed,
                        'Best Hyperparameters': best_no_sentiment_run[['Units', 'Dropout', 'Optimizer', 'Activation', 'Learning Rate', 'Epochs', 'Batch Size']].to_dict(),
                        'Trained Model': best_no_sentiment_model # Store the model object
                    })
                    print(f"    Best No Sentiment R^2 for {stock_name} (Seed {seed}): {best_no_sentiment_run['R^2']:.4f}")
                else:
                    print(f"    No valid results for No Sentiment for {stock_name} (Seed {seed}).")


                # Scenario 2: With Sentiment (FinBERT)
                for sentiment_type in sentiment_models_to_test:
                    print(f"\nScenario: With Sentiment ({sentiment_type}, Lookback={fixed_lookback})")
                    sentiment_rs_results, best_sentiment_model = run_random_search(
                        data=data.copy(),
                        num_iterations=num_random_search_iterations,
                        lookback_value=fixed_lookback,
                        use_sentiment=True,
                        sentiment_model_type=sentiment_type,
                        seed=seed
                    )

                    if not sentiment_rs_results.empty:
                        best_sentiment_run = sentiment_rs_results.loc[sentiment_rs_results['R^2'].idxmax()]
                        final_experiment_results.append({
                            'Stock Name': stock_name,
                            'Lookback': fixed_lookback,
                            'Sentiment Used': "Yes",
                            'Sentiment Model': sentiment_type,
                            'Best R^2': best_sentiment_run['R^2'],
                            'Best RMSE': best_sentiment_run['RMSE'],
                            'Best MAE': best_sentiment_run['MAE'],
                            'Seed': seed,
                            'Best Hyperparameters': best_sentiment_run[['Units', 'Dropout', 'Optimizer', 'Activation', 'Learning Rate', 'Epochs', 'Batch Size']].to_dict(),
                            'Trained Model': best_sentiment_model # Store the model object
                        })
                        print(f"    Best {sentiment_type} Sentiment R^2 for {stock_name} (Seed {seed}): {best_sentiment_run['R^2']:.4f}")
                    else:
                        print(f"    No valid results for {sentiment_type} Sentiment for {stock_name} (Seed {seed}).")

    # Final Results Aggregation and Model Saving
    final_df = pd.DataFrame(final_experiment_results)
    if not final_df.empty:
        final_df_sorted = final_df.sort_values(by=['Stock Name', 'Sentiment Used', 'Sentiment Model', 'Seed'])

    print("\n" + "="*80)
    print("           ANÁLISIS: ¿EL SENTIMIENTO MEJORA EL RENDIMIENTO? (MEJOR R^2 POR CONFIGURACIÓN)            ")
    print("="*80)

    if not final_df.empty:
        improvement_summary = []
        for (stock_name, lookback, seed), group in final_df.groupby(['Stock Name', 'Lookback', 'Seed']):
            no_sentiment_row = group[group['Sentiment Used'] == 'No']

            if not no_sentiment_row.empty:
                no_sentiment_r2 = no_sentiment_row['Best R^2'].iloc[0]
                no_sentiment_rmse = no_sentiment_row['Best RMSE'].iloc[0]
                no_sentiment_mae = no_sentiment_row['Best MAE'].iloc[0]
                no_sentiment_model = no_sentiment_row['Trained Model'].iloc[0]

                for sentiment_type in sentiment_models_to_test:
                    sentiment_row = group[(group['Sentiment Used'] == 'Yes') & (group['Sentiment Model'] == sentiment_type)]

                    if not sentiment_row.empty:
                        sentiment_r2 = sentiment_row['Best R^2'].iloc[0]
                        sentiment_rmse = sentiment_row['Best RMSE'].iloc[0]
                        sentiment_mae = sentiment_row['Best MAE'].iloc[0]
                        sentiment_model = sentiment_row['Trained Model'].iloc[0]

                        if sentiment_r2 > no_sentiment_r2:
                            improvement_summary.append({
                                'Stock Name': stock_name,
                                'Lookback': lookback,
                                'Sentiment Model': sentiment_type,
                                'Baseline R^2 (No Sentiment)': no_sentiment_r2,
                                'Sentiment R^2': sentiment_r2,
                                'R^2 Improvement': sentiment_r2 - no_sentiment_r2,
                                'Baseline RMSE (No Sentiment)': no_sentiment_rmse,
                                'Sentiment RMSE': sentiment_rmse,
                                'RMSE Improvement': no_sentiment_rmse - sentiment_rmse,
                                'Baseline MAE (No Sentiment)': no_sentiment_mae,
                                'Sentiment MAE': sentiment_mae,
                                'MAE Improvement': no_sentiment_mae - sentiment_mae,
                                'Seed': seed,
                                'Baseline Hyperparameters': no_sentiment_row['Best Hyperparameters'].iloc[0],
                                'Sentiment Hyperparameters': sentiment_row['Best Hyperparameters'].iloc[0]
                            })
                            
                            model_save_path = os.path.join(models_base_dir, stock_name)
                            os.makedirs(model_save_path, exist_ok=True) # Create stock-specific folder

                            # Save baseline model
                            if no_sentiment_model is not None:
                                baseline_model_filename = f"baseline_no_sentiment_seed{seed}_R2_{no_sentiment_r2:.4f}.keras"
                                baseline_model_path = os.path.join(model_save_path, baseline_model_filename)
                                no_sentiment_model.save(baseline_model_path)
                                print(f"Saved baseline model for {stock_name} (Seed {seed}) to: {baseline_model_path}")
                            else:
                                print(f"Warning: Baseline model for {stock_name} (Seed {seed}) was None, skipping save.")

                            # Save sentiment model
                            if sentiment_model is not None:
                                sentiment_model_filename = f"{sentiment_type}_sentiment_seed{seed}_R2_{sentiment_r2:.4f}.keras"
                                sentiment_model_path = os.path.join(model_save_path, sentiment_model_filename)
                                sentiment_model.save(sentiment_model_path)
                                print(f"Saved {sentiment_type} sentiment model for {stock_name} (Seed {seed}) to: {sentiment_model_path}")
                            else:
                                print(f"Warning: Sentiment model for {stock_name} ({sentiment_type}, Seed {seed}) was None, skipping save.")
        
        improvement_df_8 = pd.DataFrame(improvement_summary)
        
        if not improvement_df_8.empty:
            improvement_df_8_sorted = improvement_df_8.sort_values(by=['Stock Name', 'Sentiment Model', 'R^2 Improvement'], ascending=[True, True, False])
            print(improvement_df_8_sorted.to_string())
        else:
            print("No se encontraron mejoras significativas en R^2 al usar análisis de sentimiento.")
    else:
        print("No hay resultados para analizar mejoras.")

    print("\n" + "="*80)