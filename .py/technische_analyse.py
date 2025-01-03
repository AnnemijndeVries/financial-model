import pandas as pd
import yfinance as yf
import numpy as np

def add_indicators(data, lagg=30, window=20):
    # Voeg de 'Change' kolom toe voor de procentuele verandering
    data['Change'] = data['Close'].pct_change()

    # Gebruik een dictionary om nieuwe kolommen op te slaan
    new_columns = {}

    # Target (of de prijs stijgt morgen)
    new_columns['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    new_columns['Target'] = np.where(data['Change'] > 0, 1, 0)

    # Moving Averages
    new_columns['MA_5'] = data['Close'].rolling(window=5).mean()
    new_columns['MA_10'] = data['Close'].rolling(window=10).mean()
    new_columns['MA_20'] = data['Close'].rolling(window=20).mean()

    # EMA en MACD
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    new_columns['EMA_12'] = ema_12
    new_columns['EMA_26'] = ema_26
    new_columns['MACD'] = ema_12 - ema_26

    # Bollinger Bands
    bollinger_mid = data['Close'].rolling(window=20).mean()
    bollinger_std = data['Close'].rolling(window=20).std()
    new_columns['Bollinger_Mid'] = bollinger_mid
    new_columns['Bollinger_Upper'] = bollinger_mid + 2 * bollinger_std
    new_columns['Bollinger_Lower'] = bollinger_mid - 2 * bollinger_std

    # Stochastic Oscillator
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    new_columns['Stochastic'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))

    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    new_columns['RSI'] = 100 - (100 / (1 + rs))

    # Volume Weighted Average Price (VWAP)
    new_columns['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()

    # Volatiliteit en kalender features
    new_columns['Volatility'] = (data['High'] - data['Low']) / data['Open']
    new_columns['Day_of_Week'] = pd.to_datetime(data.index).dayofweek
    new_columns['Month'] = pd.to_datetime(data.index).month
    new_columns['Quarter'] = pd.to_datetime(data.index).quarter

    # Feestdagen in de VS (zoals Thanksgiving, Kerstmis, etc.)
    from pandas.tseries.holiday import USFederalHolidayCalendar
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=data.index.min(), end=data.index.max())
    new_columns['Is_Holiday'] = data.index.isin(holidays).astype(int)

    # Lag-variabelen toevoegen voor een aantal dagen
    for lag in range(1, lagg):
        new_columns[f'Prev_Close_{lag}'] = data['Close'].shift(lag)
        new_columns[f'Prev_Change_{lag}'] = data['Change'].shift(lag)
        new_columns[f'Prev_Open_{lag}'] = data['Open'].shift(lag)

    # Resistance en Support niveaus
    new_columns['Resistance'] = data['High'].rolling(window=window).max()
    new_columns['Support'] = data['Low'].rolling(window=window).min()

    # Voeg alle nieuwe kolommen in één keer toe
    new_columns_df = pd.DataFrame(new_columns, index=data.index)
    data = pd.concat([data, new_columns_df], axis=1)

    # Voeg 'Days_Since_Up' toe
    days_since_up = []
    last_up_day = None  # Houd de index bij van de laatste keer dat Target == 1 was

    for idx, target in enumerate(data['Target']):
        if target == 1:
            last_up_day = idx  # Sla de huidige index op
            days_since_up.append(0)  # 0 dagen geleden als het vandaag omhoog ging
        else:
            if last_up_day is None:
                days_since_up.append(np.nan)  # Nog nooit omhoog gegaan
            else:
                days_since_up.append(idx - last_up_day)  # Verschil in dagen

    # Voeg de kolom toe aan de dataset
    data['Days_Since_Up'] = days_since_up

    # Verwijder eventuele null-waardes die door de lag-variabelen en indicatoren zijn ontstaan
    # data.dropna(inplace=True)

    return data


def add_change_features(data):
    """
    Voegt aanvullende Change-gebaseerde variabelen toe aan de dataset.

    Args:
    data (pd.DataFrame): De dataset waaraan nieuwe features worden toegevoegd.

    Returns:
    pd.DataFrame: De dataset met extra Change-gerelateerde variabelen.
    """
    # Cumulatieve procentuele verandering over een venster
    data['Cumulative_Change_5'] = data['Change'].rolling(window=5).sum()
    data['Cumulative_Change_10'] = data['Change'].rolling(window=10).sum()

    # Aantal Up Days en Down Days binnen een venster
    data['Up_Days_Count_10'] = (data['Change'] > 0).rolling(window=10).sum()
    data['Down_Days_Count_10'] = (data['Change'] < 0).rolling(window=10).sum()

    # Standaarddeviatie van procentuele veranderingen
    data['Change_Std_Dev_5'] = data['Change'].rolling(window=5).std()
    data['Change_Std_Dev_10'] = data['Change'].rolling(window=10).std()

    # Relatie tussen Change en Volume
    data['Change_to_Volume_Ratio'] = data['Change'] / data['Volume']

    # Relatie tussen Change en Volatiliteit
    data['Change_to_Volatility'] = data['Change'] / data['Volatility']

    # Up en Down Streaks
    data['Streak_Up'] = (
        data['Target']
        .groupby((data['Target'] != data['Target'].shift()).cumsum())
        .cumsum()
        * data['Target']
    )
    data['Streak_Down'] = (
        ((data['Target'] == 0).astype(int))
        .groupby((data['Target'] != data['Target'].shift()).cumsum())
        .cumsum()
    )

    # Verwijder eventuele null-waardes door het gebruik van rolling-vensters
    # data.dropna(inplace=True)

    return data

from fredapi import Fred
from dotenv import load_dotenv
import os

def add_economic_indicators_from_fred(data: pd.DataFrame, start_date: str = "2000-01-01") -> pd.DataFrame:
    """
    Haalt economische indicatoren op via de FRED API en voegt ze toe aan een bestaande dataset.
    
    Args:
    data (pd.DataFrame): De bestaande dataset waaraan de indicatoren toegevoegd worden.
    start_date (str): Startdatum voor het ophalen van de economische indicatoren.
    
    Returns:
    pd.DataFrame: De originele DataFrame met de toegevoegde economische indicatoren.
    """
    # Converteer altijd de index naar een DatetimeIndex
    data.index = pd.to_datetime(data.index, errors='coerce')

    # Laad de .env-variabelen
    load_dotenv()

    # Haal de API-sleutel op uit de omgeving
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("API-sleutel voor FRED ontbreekt. Controleer je .env-bestand.")

    # Maak een Fred-object
    fred = Fred(api_key=api_key)

    # Definieer de tickers en bijbehorende kolomnamen
    tickers = {
        "UNRATE": "Unemployment Rate",
        "CPIAUCSL": "CPI (Inflation)",
        "GDP": "GDP Growth",
        "DGS10": "10-Year Treasury Rate",
        "FEDFUNDS": "Effective Federal Funds Rate",
        "PCE": "Personal Consumption Expenditures",
        "INDPRO": "Industrial Production Index",
        "M2SL": "Money Stock (M2)"
    }

    # Loop door de tickers om data op te halen en toe te voegen aan de dataset
    for ticker, column_name in tickers.items():
        try:
            # Haal de tijdreeks op via de FRED API
            series_data = fred.get_series(ticker, observation_start=start_date)
            
            # Zet de tijdreeks om in een DataFrame
            indicator_df = pd.DataFrame(series_data, columns=[column_name])
            indicator_df.index.name = "date"
            
            # Zorg ervoor dat de index een datetime is
            indicator_df.index = pd.to_datetime(indicator_df.index)
            
            # Voeg de nieuwe kolom toe aan de bestaande dataset
            data = data.join(indicator_df, how="left")
            
            # Vul eventuele missende waarden
            data[column_name] = data[column_name].ffill()  # Forward fill missing values
            
            # print(f"Succesvol toegevoegd: {column_name} ({ticker})")
        
        except Exception as e:
            print(f"Kon {column_name} ({ticker}) niet ophalen: {e}")

    return data
