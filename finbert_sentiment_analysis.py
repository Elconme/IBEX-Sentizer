import pandas as pd
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import os

# Function to Process News Sentiment
def process_news_sentiment(news_csv_path, finbert_model, label_to_polarity=None):
    if label_to_polarity is None:
        label_to_polarity = {'positive': 1, 'neutral': 0, 'negative': -1}

    def analyze_sentiment(text):
        try:
            result = finbert_model(text)[0]
            return result['label'], float(result['score'])
        except Exception as e:
            print(f"Error analyzing sentiment for text (truncated to 50 chars): '{text[:50]}...' Error: {e}")
            return 'ERROR', 0.0

    try:
        compNews = pd.read_csv(news_csv_path)
        
        if 'headline' not in compNews.columns:
            print(f"Error: 'headline' column not found in {news_csv_path}. Skipping this file.")
            return

        print(f"Analyzing sentiment for {len(compNews)} headlines in {os.path.basename(news_csv_path)}...")
        sentiments = compNews['headline'].apply(analyze_sentiment)
        compNews['sentiment_label'] = sentiments.apply(lambda x: x[0])
        compNews['sentiment_score'] = sentiments.apply(lambda x: x[1])
        compNews['polarity'] = compNews['sentiment_label'].map(label_to_polarity)
        compNews['weighted_score'] = compNews['polarity'] * compNews['sentiment_score']
        
        compNews.to_csv(news_csv_path, index=False)
        print(f"Processed sentiment for {os.path.basename(news_csv_path)}. Added sentiment columns.")
    except Exception as e:
        print(f"Error processing news sentiment for {os.path.basename(news_csv_path)}: {e}")

# Function to Merge Sentiment with Price Data
def merge_sentiment_with_price(price_file, news_file):
    print(f"Attempting to merge {os.path.basename(news_file)} with {os.path.basename(price_file)}...")
    try:
        price_df = pd.read_csv(price_file, parse_dates=['Date'])
    except Exception as e:
        print(f"Error reading price file {price_file}: {e}. Skipping merge for this pair.")
        return

    try:
        news_df = pd.read_csv(news_file)
    except Exception as e:
        print(f"Error reading news file {news_file}: {e}. Skipping merge for this pair.")
        return

    if 'versionCreated' not in news_df.columns:
        print(f"Error: 'versionCreated' column not found in {os.path.basename(news_file)}. Cannot merge sentiment by date. Skipping this pair.")
        return
    
    news_df['date'] = pd.to_datetime(
        news_df['versionCreated'],
        errors='coerce',
        dayfirst=True,
        format='mixed'
    ).dt.date

    news_df = news_df.dropna(subset=['date'])

    if 'weighted_score' not in news_df.columns:
        print(f"Error: 'weighted_score' column not found in {os.path.basename(news_file)}. Please ensure sentiment processing was successful. Skipping this pair.")
        return

    daily_sentiment = news_df.groupby('date')['weighted_score'].mean().reset_index()
    daily_sentiment.columns = ['Date', 'daily_sentiment_score']
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    
    price_df = price_df.merge(daily_sentiment, on='Date', how='left')
    
    first_valid_idx = price_df['daily_sentiment_score'].first_valid_index()
    if first_valid_idx is not None:
        price_df = price_df.loc[first_valid_idx:].reset_index(drop=True)
        print(f"Trimmed price data to start from first sentiment date: {price_df['Date'].min().strftime('%Y-%m-%d')}")
    else:
        print("Warning: No valid sentiment dates found in news data to merge with price data. Proceeding without trimming.")

    missing_sentiment_count = price_df['daily_sentiment_score'].isnull().sum()
    if missing_sentiment_count > 0:
        print(f"Interpolating {missing_sentiment_count} missing daily sentiment values in {os.path.basename(price_file)} using 'nearest' method...")
        price_df['daily_sentiment_score'] = price_df['daily_sentiment_score'].interpolate(method='nearest')
    else:
        print("No missing sentiment values to interpolate.")

    price_df.to_csv(price_file, index=False)
    print(f"Updated {os.path.basename(price_file)} with 'daily_sentiment_score' column.")


if __name__ == "__main__":
    DATA_FOLDER = 'data'
    FINBERT_MODEL_NAME = "ProsusAI/finbert"
    LABEL_TO_POLARITY = {'positive': 1, 'neutral': 0, 'negative': -1}

    print("--- Initializing FinBERT Model ---")
    try:
        tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        model = BertForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
        finbert_sentiment = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        print("FinBERT model loaded successfully.")
    except Exception as e:
        print(f"Error loading FinBERT model: {e}")
        print("Please ensure you have an active internet connection and the 'transformers' library is installed.")
        exit("Exiting due to FinBERT model loading error.") 

    news_files = []
    price_files = []

    if not os.path.exists(DATA_FOLDER):
        print(f"Error: Data folder '{DATA_FOLDER}' not found. Please create it and place your CSV files inside.")
        exit("Exiting due to missing data folder.")

    print(f"\n--- Scanning '{DATA_FOLDER}' for CSV files ---")
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith('.csv'):
            full_path = os.path.join(DATA_FOLDER, filename)
            if '_news' in filename:
                news_files.append(full_path)
            elif '_data' in filename:
                price_files.append(full_path)
    
    if not news_files:
        print(f"No CSV files containing '_news' found in '{DATA_FOLDER}'.")
    else:
        print(f"Found {len(news_files)} news files.")

    if not price_files:
        print(f"No CSV files containing '_data' found in '{DATA_FOLDER}'.")
    else:
        print(f"Found {len(price_files)} price files.")
    
    if not news_files or not price_files:
        print("\nWarning: Not enough news or price files found to perform merges. Ensure your files are named correctly (e.g., 'STOCK_news.csv', 'STOCK_data.csv').")

    # Process News Sentiment for all identified news files
    print("\n--- Processing Sentiment for All News Files ---")
    for news_path in news_files:
        process_news_sentiment(news_path, finbert_sentiment, LABEL_TO_POLARITY)

    # Merge Sentiment with Price Data for corresponding pairs
    print("\n--- Merging Sentiment with Price Data for Corresponding Pairs ---")
    processed_pairs = 0
    for price_path in price_files:
        stock_identifier = os.path.basename(price_path).replace('_data.csv', '')
        
        corresponding_news_file = None
        for news_path in news_files:
            if stock_identifier + '_news' in os.path.basename(news_path):
                corresponding_news_file = news_path
                break
        
        if corresponding_news_file:
            merge_sentiment_with_price(price_path, corresponding_news_file)
            processed_pairs += 1
        else:
            print(f"Warning: No corresponding news file found for {os.path.basename(price_path)}. Skipping merge for this price file.")

    if processed_pairs == 0 and news_files and price_files:
        print("\nNo matching news/price file pairs found for merging. Ensure your stock identifiers in filenames match (e.g., 'ABC_news.csv' and 'ABC_data.csv').")
    elif processed_pairs == 0:
        print("\nNo files processed. Ensure your data folder contains CSVs with '_news' and '_data' in their names.")

    print("\n--- Process Complete ---")