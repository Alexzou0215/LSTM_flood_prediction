import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import os
import joblib
import openpyxl

class MCDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)  # Always use dropout for training and prediction

def selective_quantile_loss(tau=0.7, threshold=0.95):
    def loss(y_true, y_pred):
        # Mask: 1 if y_true > threshold (in standard scaled data), else 0
        mask = K.cast(y_true > threshold, dtype='float32')
        
        # Quantile loss (asymmetric) applied only to peaks
        error = y_true - y_pred
        quantile = K.maximum(tau * error, (tau - 1) * error)
        peak_loss = mask * quantile

        # Optional: apply MSE to non-peak values
        mse_loss = (1.0 - mask) * K.square(error)

        return K.mean(peak_loss + mse_loss)
    return loss

# Load Data with Hourly Timestamps
def load_data(file_path, name=None):
    df = pd.read_csv(file_path, usecols=["Timestamp", "Varibale Value"], low_memory=False)

    # Convert timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %H:%M")
    df = df.sort_values(by='Timestamp') 

    # Ensure variable column is numeric
    df["Varibale Value"] = pd.to_numeric(df["Varibale Value"], errors="coerce")

    # Rename columns for clarity
    if name is not None:
        df = df.rename(columns={"Varibale Value": name})

    return df

def load_GloFAS_data(file_path, name='GloFAS_error_In'):
    df = pd.read_csv(file_path, usecols=["Timestamp", "spread"], low_memory=False)

    # Convert timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %H:%M")
    df = df.sort_values(by='Timestamp') 

    # Ensure variable column is numeric
    df["spread"] = pd.to_numeric(df["spread"], errors="coerce")

    # Rename columns for clarity
    if name is not None:
        df = df.rename(columns={"spread": name})

    return df

def data_clean_align_date_3(df1, df2, df3):
    """
    Align df1, df2, and df3 on a common hourly timeline, and fill missing values using ffill, then bfill.
    Assumes all have a 'Timestamp' column (datetime).
    Returns: df1_aligned, df2_aligned, df3_aligned
    """
    # Ensure Timestamp is datetime
    for df in [df1, df2, df3]:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Build full hourly timeline from min to max of all dfs
    start = max(df1['Timestamp'].min(), df2['Timestamp'].min(), df3['Timestamp'].min())
    end   = min(df1['Timestamp'].max(), df2['Timestamp'].max(), df3['Timestamp'].max())
    full_timeline = pd.DataFrame({'Timestamp': pd.date_range(start=start, end=end, freq='H')})

    # Merge to align each df on the hourly timeline
    df1_aligned = pd.merge(full_timeline, df1, on='Timestamp', how='left')
    df2_aligned = pd.merge(full_timeline, df2, on='Timestamp', how='left')
    df3_aligned = pd.merge(full_timeline, df3, on='Timestamp', how='left')

    # Fill missing values: forward fill, then backward fill for leading NaNs
    df1_aligned = df1_aligned.ffill().bfill()
    df2_aligned = df2_aligned.ffill().bfill()
    df3_aligned = df3_aligned.ffill().bfill()

    return df1_aligned, df2_aligned, df3_aligned

# Aggregate hourly data into assigned time step, for target column and input column
def aggregate_to_nhourly_last_timestamp(df, aggregation_period=3):
    # Sort and reset index
    df = df.sort_values(by='Timestamp').reset_index(drop=True)

    # Group by the specified aggregation period
    group_size = aggregation_period
    num_groups = len(df) // group_size

    result = []
    for i in range(num_groups):
        group = df.iloc[i*group_size : (i+1)*group_size]
        avg_1 = group.iloc[:, 1].mean()  # First value column
        timestamp = group.iloc[-1]["Timestamp"]  # Use last timestamp in the group
        result.append({
            "Timestamp": timestamp,
            df.columns[1]: avg_1,
        })

    return pd.DataFrame(result)

# Merge Two DataFrames with lagged time step 
def merge_with_context_lag(
    df_target, df_context, lag_steps=7, 
    target_col='419012_target', context_col='419001_context', lagged_col = "419001_context_lag"):
    # Ensure sorted by timestamp
    df_target = df_target.sort_values('Timestamp').reset_index(drop=True)
    df_context = df_context.sort_values('Timestamp').reset_index(drop=True)
    
    # Create dummy (lagged context)
    dummy = df_context[['Timestamp']].copy()
    dummy[lagged_col] = df_context[context_col].shift(lag_steps)
    dummy = dummy.dropna(subset=[lagged_col]).reset_index(drop=True)
    
    # Merge all on Timestamp (only rows present in dummy will be kept)
    merged = (
        df_target
        .merge(df_context, on='Timestamp', how='inner', suffixes=('', '_context'))
        .merge(dummy, on='Timestamp', how='inner')
    )
    
    # Rename columns for clarity if needed
    merged = merged.rename(columns={
        target_col: target_col,
        context_col: context_col,
        lagged_col: lagged_col
    })
    
    return merged

def merge_with_GloFAS(merged_df, df3):
    # Identify non-Timestamp column (optional, for clarity)
    data_col = [col for col in df3.columns if col != 'Timestamp'][0]
    # Merge directly, no renaming
    merged = merged_df.merge(df3, on='Timestamp', how='inner')
    return merged

# include labels (train/vali/test) if any, otherwise it follows the simplest preprocessing
def create_sequences(data, window_size, lead_time, labels=None):
    X, Y, seq_labels = [], [], []
    n = len(data)
    for i in range(n - window_size - lead_time + 1):
        # Check input window and output (target) labels
        if labels is not None:
            input_labels = labels[i : i + window_size]
            output_labels = labels[i + window_size : i + window_size + lead_time]
            all_labels = np.concatenate([input_labels, output_labels])
            if not np.all(all_labels == all_labels[0]):
                continue  # Only keep "pure" sequences
            this_label = all_labels[0]
        else:
            this_label = None

        X.append(data[i:i+window_size])
        Y.append(data[i+window_size:i+window_size+lead_time].flatten())
        if labels is not None:
            seq_labels.append(this_label)

    if labels is not None:
        return np.array(X), np.array(Y), np.array(seq_labels)
    else:
        return np.array(X), np.array(Y)

# Create Input-Output Sequences
def preprocess_data_inference_3col(
    df, scaler, target_column, context_column1, context_column2,
    window_size=48, lead_time=16
):
    """
    Preprocess new data for inference using an existing, already-fitted scaler for 3 columns.
    Returns LSTM-ready X, dummy Y, unscaled targets, unscaled context2, date sequences, and the scaler.
    """
    # Transform with the provided, already-fitted scaler (on all three columns)
    input_features = [target_column, context_column1, context_column2]
    scaled_data = scaler.transform(df[input_features])

    # Create LSTM input sequences using all three features
    X, _ = create_sequences(scaled_data, window_size, lead_time)
    X = X.reshape((X.shape[0], X.shape[1], 3))  # (samples, window_size, 3 features)

    # Dummy Y (all zeros, for inference; shape matches (samples, lead_time))
    Y = np.zeros((X.shape[0], lead_time))

    # Unscaled target (raw values) from target_column
    _, Y_unscaled = create_sequences(df[target_column].values.reshape(-1, 1), window_size, lead_time)

    # Unscaled context2 (Glo_error_unscale)
    _, Glo_error_unscale = create_sequences(df[context_column2].values.reshape(-1, 1), window_size, lead_time)

    # Date sequences
    _, date_sequences = create_sequences(df["Timestamp"].values.reshape(-1, 1), window_size, lead_time)

    return X, Y, Y_unscaled, Glo_error_unscale, date_sequences


def build_lstm_model(window_size, lead_time, dropout_rate=0.05):
    model = Sequential([
        LSTM(100, activation='tanh', return_sequences=True, input_shape=(window_size, 3)),
        MCDropout(dropout_rate),
        LSTM(100, activation='tanh'),
        MCDropout(dropout_rate),
        Dense(lead_time)
    ])
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=selective_quantile_loss(tau=0.7, threshold=0.95))
    return model

# Forecasting
def forecast_mc_dropout(model, X_test, scaler, Glofas_raw_y_error, T=100):
    """
    MC Dropout forecast with correct inverse scaling for a scaler fit on 2 columns.
    Args:
        model: Trained Keras model with MCDropout layers.
        X_test: Test input, shape (samples, timesteps, features).
        scaler: Scaler fit on 2 columns (the first is the target).
        T: Number of MC samples.
    Returns:
        mean_prediction, std_prediction: Both are (samples, n_steps) arrays in original scale.
    """
    preds_scaled = []

    for _ in range(T):
        # MC Dropout prediction (dropout ON at test time)
        pred = model(X_test, training=True).numpy()  # shape: (samples, n_steps)
        preds_scaled.append(pred)
    preds_scaled = np.array(preds_scaled)  # shape: (T, samples, n_steps)

    # Compute mean and std along the MC sample axis (axis=0)
    mean_scaled = np.mean(preds_scaled, axis=0)   # shape: (samples, n_steps)
    std_scaled = np.std(preds_scaled, axis=0)     # shape: (samples, n_steps)

    # Inverse scale, assuming scaler was fit on 3 columns and target is column 0
    def inverse_scale(pred_scaled):
        n_samples, n_steps = pred_scaled.shape
        dummy_input = np.zeros((n_samples * n_steps, 3))
        dummy_input[:, 0] = pred_scaled.flatten()
        pred_inverse = scaler.inverse_transform(dummy_input)[:, 0]
        return pred_inverse.reshape(n_samples, n_steps)

    mean_prediction = inverse_scale(mean_scaled)
    # For std: only multiply by the target column's std scale (no shift for std)
    std_scale = scaler.scale_[0]
    std_prediction = std_scaled * std_scale

    # Add GloFAS error to the mean prediction if provided
    std_prediction_all = std_prediction + Glofas_raw_y_error

    return mean_prediction, std_prediction_all

# Evaluation Metrics
def calculate_nse(actual, predicted):
    return 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))

def calculate_kge(actual, predicted):
    r = np.corrcoef(actual.squeeze(), predicted.squeeze())[0, 1]
    beta = np.mean(predicted) / np.mean(actual)
    gamma = np.std(predicted) / np.std(actual)
    return 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)

def root_mean_squared_error(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

def plot_results_with_uncertainty(actual, predicted, uncertainty, dates, lead_step=-1, save_path=None):
    """
    Plot for a chosen lead step (default: last lead step).
    """
    import numpy as np
    # allow users to choose lead step (1-based, if negative, use last)
    if lead_step < 0:
        lead_index = actual.shape[1] - 1
    else:
        lead_index = lead_step if lead_step < actual.shape[1] else actual.shape[1] - 1
    actual_plot = actual[:, lead_index]
    predicted_plot = predicted[:, lead_index]
    uncertainty_plot = uncertainty[:, lead_index]
    dates_plot = dates[:, lead_index]

    rmse      = root_mean_squared_error(actual_plot, predicted_plot)
    nse_value = calculate_nse(actual_plot, predicted_plot)
    kge_value = calculate_kge(actual_plot, predicted_plot)

    plt.style.use('ggplot')
    plt.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Arial'],
        'font.size':         14,
        'axes.titlesize':    18,
        'axes.labelsize':    16,
        'xtick.labelsize':   14,
        'ytick.labelsize':   14,
        'legend.fontsize':   18,
        'figure.titlesize':  20
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('none')    
    ax.set_facecolor('none') 
    ax.grid(False)

    lower = predicted_plot - 1.96 * uncertainty_plot
    upper = predicted_plot + 1.96 * uncertainty_plot
    lower = np.where(lower < 0, 1, lower)

    ax.plot(dates_plot, actual_plot, label='Actual Flow Rate (ML/day)', linewidth=2)
    ax.plot(dates_plot, predicted_plot, label='Predicted Flow Rate (ML/day)', linestyle='--', linewidth=2)
    ax.fill_between(dates_plot, lower, upper, color='gray', alpha=0.3, label="95% Prediction Interval")
    ax.set_title(f"Lead: {lead_index+1} | RMSE: {rmse:.1f}, NSE: {nse_value:.2f}, KGE: {kge_value:.2f}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Flow Rate (ML/day)")
    ax.legend()
    fig.autofmt_xdate(rotation=45, ha='right')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.show()
    plt.close(fig)

def save_results_excel_per_lead_with_uncertainty(actual, predicted, uncertainty, dates, save_path="prediction_results_by_lead.xlsx"):
    """
    Save results for each lead time to a separate sheet in an Excel file.
    Each sheet has columns: Timestamp, Actual_FlowRate, Predicted_FlowRate, Uncertainty
    """
    import openpyxl
    n_samples, n_leads = actual.shape
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        for lead in range(n_leads):
            df_lead = pd.DataFrame({
                "Timestamp": pd.to_datetime(dates[:, lead]),
                "Actual_FlowRate": actual[:, lead],
                "Predicted_FlowRate": predicted[:, lead],
                "Uncertainty": uncertainty[:, lead]
            })
            df_lead.to_excel(writer, sheet_name=f"Lead_{lead+1}", index=False)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# Make sure you have already defined:
#   load_data
#   create_sequences
#   build_lstm_model
#   forecast
#   preprocess_data_inference (as above)

if __name__ == "__main__":
    # File paths for the two locations
    file1 = 'test_419012.csv'  # Boggabri  (target)
    file2 = 'test_419001.csv'  # Gunnedah (context)
    file3 = 'test_GloFAS_spread_419012.csv'  # GloFAS error

    # Rename the columns for clarity
    name1 = "419012_target"
    name2 = "419001_context"
    name3 = "419001_context_lag"
    name4 = "GloFAS_error_In"
    
    # Set parameters
    window_size = 48 # Change as needed
    lead_time = 16  # Change as needed
    timescale = 3
    lag_steps = 5  # Number of steps to lag context column

    # Load and merge the datasets
    df1 = load_data(file1,name1)
    df2 = load_data(file2,name2)
    df3 = load_GloFAS_data(file3, name4)
    df1,df2,df3 = data_clean_align_date_3(df1,df2,df3)

    # Aggregate into 3 hourly
    df1 = aggregate_to_nhourly_last_timestamp(df1, timescale)
    df2 = aggregate_to_nhourly_last_timestamp(df2, timescale)
    df3 = aggregate_to_nhourly_last_timestamp(df3, timescale)

    #Merge with lagged contex
    df = merge_with_context_lag(df1, df2, lag_steps, 
                                 target_col= name1, context_col=name2,lagged_col=name3)
    df = merge_with_GloFAS(df, df3)

    #Cut df to ensure floods event in all three periods
    start_date = pd.to_datetime("2022-07-01")
    end_date = pd.to_datetime("2023-01-31")
    df = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]

        # --- 2. Load the saved scaler ---
    # --- Load Model and scaler names (consistent naming) ---
    model_path = "lstm_flood_pro10_model5.h5"
    weights_path = model_path + "_best_val.weights.h5"
    scaler_path = model_path.replace('.h5', '_scaler.pkl')

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Cannot find scaler file: {scaler_path}")
    scaler = joblib.load(scaler_path)

    # --- 3. Preprocess new data (use inference version) ---
    X_new, Y_new_scaled, Y_new_unscaled, Glofas_new_raw_y_error, new_dates = preprocess_data_inference_3col(
        df = df,
        scaler = scaler,
        target_column = name1,
        context_column1= name3,
        context_column2= name4,  # GloFAS error
        window_size=window_size,
        lead_time=lead_time
    )

    # --- 4. Build model architecture (must match trained model) ---
    model = build_lstm_model(window_size=window_size, lead_time=lead_time)

    # --- 5. Load trained weights ---
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Cannot find weights file: {weights_path}")
    model.load_weights(weights_path)

    # --- 6. Make predictions ---
    Y_new_prediction_mean, Y_new_prediction_std = forecast_mc_dropout(model, X_new, scaler,Glofas_new_raw_y_error)

        # --- 7. Plot and save results for a selected lead time (e.g., lead 5) ---
    plot_results_with_uncertainty(
        actual = Y_new_unscaled,
        predicted = Y_new_prediction_mean,
        uncertainty = Y_new_prediction_std,
        dates = new_dates,
        leadsteps = 5,  # change this to plot a different lead step (1-based)
        save_path = "uncertainty_vs_actual_lead5.png"
    )

    # --- 8. Save all lead times to an Excel file with uncertainty ---
    save_results_excel_per_lead_with_uncertainty(
        actual = Y_new_unscaled,
        predicted = Y_new_prediction_mean,
        uncertainty = Y_new_prediction_std,
        dates = new_dates,
        save_path = "prediction_results_by_lead_with_uncertainty.xlsx"
    )