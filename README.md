# LSTM Flood Prediction Models

This repository provides three LSTM-based models for predicting flow rate at Boggabri.

## Model Overview

- **Model 1:** Baseline.  
  - Input: Boggabri flow rate  
  - Lead time: 16 hours  
  - Lookback window: 48 hours  
  - Uncertainty: None

- **Model 2:** Extended lead time.  
  - Input: Boggabri and Gunnedah flow rates (Gunnedah lagged by 20 hours)  
  - Lead time: 48 hours  
  - Lookback window: 144 hours  
  - Uncertainty: MC dropout

- **Model 3:** Enhanced uncertainty characterization.  
  - Input: Boggabri and Gunnedah flow rates (Gunnedah lagged by 20 hours), GloFAS standard deviation  
  - Lead time: 48 hours  
  - Lookback window: 144 hours  
  - Uncertainty: MC dropout and GloFAS

Each model predicts the flow rate at Boggabri.

## Usage

- Model files, training and testing notebooks, and data are organized in:
  - `models/lstm_model1/`
  - `models/lstm_model2/`
  - `models/lstm_model3/`

See each subfolder for details and Colab-ready notebooks.
