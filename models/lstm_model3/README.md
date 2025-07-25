# Model 3 – LSTM Flood Prediction (Pro10, Contextual + GloFAS Lag)

Predicts hourly river flow at Boggabri using an LSTM network with contextual input from Gunnedah and GloFAS ensemble spread as lagged features.

**Input:**  
- `test_419012.csv` — hourly flow data from Boggabri (`timestamp`, `flowrate`)
- `test_419001.csv` — hourly flow data from Gunnedah (`timestamp`, `flowrate`)
- `test_GloFAS_spread_419012.csv` — hourly GloFAS ensemble spread features

**Output:**  
- Predicted hourly flow rates for Boggabri (as CSV, 16 pages for 16 lead times)
- Evaluation metrics and/or comparison plots

**Files:**  
- `lstm_flood_pro10_model3_scaler.pkl`: Scaler for input preprocessing  
- `lstm_flood_pro10_model3.h5_best_val.weights`: Best model weights  
- `model_testing(if host for API).py`: Script for local/API testing  
- `model_testing(if host on colab).ipynb`: Notebook for Colab  
- `test_419012.csv`: Boggabri input data  
- `test_419001.csv`: Gunnedah input data  
- `test_GloFAS_spread_419012.csv`: GloFAS ensemble spread data

**Usage:**  
1. Put all files in the same folder.
2. Install dependencies:  
   `!pip install numpy==2.0.2 pandas==2.2.3 matplotlib==3.9.2 scikit-learn==1.6.1 tensorflow==2.19.0 joblib==1.4.2`
3. **For Colab:**  
   [![Open In Colab](https://colab.research.google.com/github/Alexzou0215/LSTM_flood_prediction/blob/main/models/lstm_model3/model_testing(if%20host%20on%20colab).ipynb)](https://colab.research.google.com/github/Alexzou0215/LSTM_flood_prediction/blob/main/models/lstm_model3/model_testing(if%20host%20on%20colab).ipynb)  
   Click the badge above to launch the notebook directly in Colab.  
   After launching, upload all required files (data, weights, scaler) using the upload cell at the top of the notebook:
   ```python
   # Upload required files: test_419012.csv, test_419001.csv, test_GloFAS_spread_419012.csv, lstm_flood_pro10_model3_scaler.pkl, lstm_flood_pro10_model3.h5_best_val.weights
   from google.colab import files
   uploaded = files.upload()
4. **For local/API:**
    Run python model_testing(if host for API).py.

All data and outputs use hourly time steps. File paths are relative (e.g., ./test_419012.csv, ./test_419001.csv，./test_GloFAS_spread_419012.csv).