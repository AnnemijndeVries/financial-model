import yfinance as yf
import pandas as pd

def add_economic_indicators(data: pd.DataFrame):
    """
    Haalt economische indicatoren op via Yahoo Finance en voegt ze toe aan de bestaande dataset.
    
    Args:
    data (pd.DataFrame): De bestaande dataset waaraan de indicatoren toegevoegd worden.
    
    Returns:
    pd.DataFrame: De originele DataFrame met de toegevoegde indicatoren.
    """
    # Economische indicatoren via FRED en andere bronnen
    tickers = {
        "^VIX": "VIX",    # Volatiliteitsindex
        "XLK": "Technology",
        "XLE": "Energy",
        "XLF": "Financials",
        "XLV": "Healthcare",
        "^TNX": "10Yr_Treasury_Rate",  # 10-jaars rente
    }

#        "DJI": "DowJones",  # Dow Jones


    # Loop door de tickers om data op te halen en toe te voegen aan de dataset
    for ticker, column_name in tickers.items():
        # Haal de data op voor de huidige ticker
        ticker_data = yf.download(ticker, start="2010-01-01", progress=False)
        
        # Controleer of er een MultiIndex is en maak deze plat
        if isinstance(ticker_data.columns, pd.MultiIndex):
            ticker_data.columns = ticker_data.columns.get_level_values(0)
        
        # Gebruik alleen de slotkoers en hernoem de kolom
        ticker_data = ticker_data[['Close']].rename(columns={'Close': column_name})
        
        # Formatteer de index naar 'YYYY-MM-DD'
        ticker_data.index = ticker_data.index.strftime('%Y-%m-%d')
        ticker_data.index = pd.to_datetime(ticker_data.index)
        
        # Voeg de nieuwe kolom toe aan de dataset
        data = data.join(ticker_data, how='left')
        
        # Vul eventuele NaN-waarden in de kolom in
        # data[column_name] = data[column_name].ffill()  # Forward fill missing values

    return data
