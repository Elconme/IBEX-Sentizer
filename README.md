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
