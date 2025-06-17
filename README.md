# 🇪🇸 IBEX-Sentizer: Forecasting IBEX 35 Banking Sector Stock Prices with FinBERT Sentiment and LSTM Deep Learning

This repository contains the code for a project focused on forecasting stock prices within the IBEX 35 banking sector. It leverages both historical price data and FinBERT-derived sentiment from financial news headlines, processed through Long Short-Term Memory (LSTM) deep learning models.

## 📁 Repository Structure and Usage

This section outlines each file's role and provides instructions on how to execute them. It is highly recommended to run the scripts in the order presented below to ensure proper data generation and model training.

### 1. `rdp_data_downloader.py`

This script is responsible for the extraction of historical stock price data and news headlines from **Refinitiv Workspace**.

-   **Key Features:**
    * 📊 **Historical Price Data:** Extracts daily historical stock prices.
    * 📰 **News Headlines:** Fetches relevant news headlines.
-   **Important Note:** The Refinitiv API for news extraction has a limitation and **only allows extraction of news from the past 15 months**. This means that if your project requires a longer history of sentiment data, you might need to supplement it with data from other sources.

**How to Execute:**
```bash
python rdp_data_downloader.py
```
### 2. `finbert_sentiment_analysis.py`

This script processes the news headlines extracted by `rdp_data_downloader.py` to derive sentiment scores using the **FinBERT** transformer model. These sentiment scores are then integrated into your historical stock price dataset as a new feature.

-   **Key Features:**
    * 🧠 **Sentiment Extraction:** Applies FinBERT to quantify sentiment from news.
    * ➕ **Feature Engineering:** Adds the sentiment score as a new column to the historical price data, ready for model training.

**How to Execute:**
```bash
python finbert_sentiment_analysis.py
```
### 3. `lstm_model_training.py`

This script handles the training and parameter configuration of the LSTM deep learning models. It builds and trains the predictive models using the combined technical indicators and FinBERT sentiment data.

-   **Key Features:**
    * 📈 **LSTM Model Training:** Configures and trains LSTM models for stock price prediction.
-   **Data Period Consideration:** This design is optimized for a **five-year period of historical data**. If the news sentiment data is not enhanced with other sources to cover this full period, the predictive results, especially for models incorporating sentiment, may not be as promising due to the 15-month Refinitiv news API limitation.

**How to Execute:**
```bash
python lstm_model_training.py
```
### 4. `prediction_plot.py`

This script visualizes the performance of the trained LSTM models. It generates plots comparing the predicted stock prices against the actual historical values, allowing for a clear assessment of the model's accuracy.

-   **Key Features:**
    * 📊 **Prediction Visualization:** Draws plots of predicted vs. real stock prices.
    * 🔍 **Performance Insight:** Helps in understanding how well the models are performing over time.

**How to Execute:**
```bash
python prediction_plot.py
```
