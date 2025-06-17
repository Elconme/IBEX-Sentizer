import pandas as pd
from datetime import datetime, timedelta
import os
import warnings
import refinitiv.data as rdp

warnings.filterwarnings("ignore")

# Read the app key from the file
try:
    app_key = open("app_key.txt", "r").read().strip()
except FileNotFoundError:
    raise FileNotFoundError("The file 'app_key.txt' was not found. Please ensure it exists and contains a valid app key.")

# Open the RDP session
try:
    rdp.open_session()
    print("Session opened successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to open session with the provided app key. Error: {e}")

ric_index = '.IBEX'
start_date = datetime(2025, 5, 22).date()

# Function to Get Constituents
def get_constituents_as_of(ric_code, date_obj):
    date_str = date_obj.strftime('%Y-%m-%d') # Format date to string here
    print(f"Fetching constituents for {ric_code} as of {date_str}...")
    const_df = rdp.get_data(
        universe=[f"0#{ric_code}({date_str.replace('-', '')})"],
        fields=["TR.PriceClose", "TR.CompanyName", "TR.RIC"],
        parameters={"SDATE": f"{date_str}", "EDATE": f"{date_str}"}
    )
    if 'RIC' in const_df.columns and 'Instrument' not in const_df.columns:
        const_df.rename(columns={'RIC': 'Instrument'}, inplace=True)
    return const_df

# Function to Fetch Historical Prices
def fetch_historical_prices(ric_code, d_now_obj, days_back=365 * 5):
    s_date = d_now_obj - timedelta(days=days_back)
    print(f"Fetching historical data from {s_date} to {d_now_obj} for RIC: {ric_code}")

    historical_data = rdp.get_history(
        universe=ric_code,
        interval="daily",
        fields=["OPEN_PRC", "TRDPRC_1", "HIGH_1", "LOW_1", "ACVOL_UNS"],
        start=str(s_date), # Convert to string for the API call
        end=str(d_now_obj) # Convert to string for the API call
    )

    historical_data = historical_data.dropna()
    return historical_data

# Function to Fetch Financial News
def fetch_financial_news(riclist, d_now_news_obj, out_dir='data'):
    os.makedirs(out_dir, exist_ok=True)

    news_start_date = d_now_news_obj - timedelta(days=365 * 2)
    comp_news = pd.DataFrame()

    for ric_code in riclist:
        print(f"Fetching news for RIC: {ric_code} from {news_start_date} to {d_now_news_obj}...")
        try:
            c_headlines = rdp.news.get_headlines(
                f"R:{ric_code} AND Language:LEN AND Topic:SIGNWS",
                start=str(news_start_date), # Convert to string for the API call
                end=str(d_now_news_obj),     # Convert to string for the API call
                count=10000
            )
            c_headlines['cRIC'] = ric_code
            if not comp_news.empty:
                comp_news = pd.concat([comp_news, c_headlines])
            else:
                comp_news = c_headlines
        except Exception as e:
            print(f"Could not fetch news for {ric_code}. Error: {e}")
            c_headlines = pd.DataFrame()
            pass

        file_path = os.path.join(out_dir, f'{ric_code}_news.csv')
        if not c_headlines.empty:
            c_headlines.to_csv(file_path, index=True)
            print(f"Saved {ric_code} news data to {file_path}")
        else:
            print(f"No news fetched for {ric_code}, skipping CSV save.")
    return comp_news

if __name__ == "__main__":
    # Get initial constituents for .IBEX on the specified start_date
    initial_constituents_df = get_constituents_as_of(ric_index, start_date)

    # Filter for financial institutions
    search_terms = r'(bank|banco)'
    financial_df = initial_constituents_df[initial_constituents_df['Company Name'].str.contains(search_terms, case=False, regex=True)]
    initial_constituents_rics = financial_df['Instrument'].to_list()
    print("\nFinancial Constituents identified:")
    print(financial_df)

    os.makedirs('data', exist_ok=True)

    # Retrieve and save historical data for each financial constituent
    print("\n--- Fetching Historical Price Data ---")
    for ric_item in initial_constituents_rics:
        # Pass the datetime.date object for start_date
        historical_data = fetch_historical_prices(ric_item, start_date)
        file_path = os.path.join('data', f'{ric_item}_data.csv')
        historical_data.to_csv(file_path, index=True)
        print(f"Saved {ric_item} historical data to {file_path}")

    # Retrieve and save news for each financial constituent
    print("\n--- Fetching Financial News ---")
    fetch_financial_news(initial_constituents_rics, start_date)

    # Close the RDP session when done
    try:
        rdp.close_session()
        print("\nSession closed successfully.")
    except Exception as e:
        print(f"Error closing session: {e}")
