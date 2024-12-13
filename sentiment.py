# from urllib.request import urlopen, Request
# from bs4 import BeautifulSoup
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import pandas as pd

# def fetch_and_analyze_sentiment(ticker: str):
#     """
#     Haalt nieuwsartikelen op van Finviz voor een gegeven ticker, voert sentimentanalyse uit en retourneert een DataFrame.
    
#     Args:
#     ticker (str): De ticker voor de aandelen (bijv. 'SPY').
    
#     Returns:
#     pd.DataFrame: DataFrame met de kolommen 'date', 'compound', en de titel van het nieuwsartikel.
#     """
#     # Definieer de Finviz URL voor de gegeven ticker
#     finviz_url = f'https://finviz.com/quote.ashx?t={ticker}&p=d'
    
#     # Maak een lege lijst voor de opgehaalde gegevens
#     parsed_data = []
#     previous_date = None  # Houd de vorige datum bij

#     # Verzend de verzoeken en verwerk de HTML-inhoud
#     req = Request(url=finviz_url, headers={'user-agent': 'my-app'})
#     response = urlopen(req)
#     html = BeautifulSoup(response, features='html.parser')
    
#     # Zoek de tabel met nieuwsitems
#     news_table = html.find(id='news-table')
    
#     # Itereer over de nieuwsitems in de tabel
#     for row in news_table.findAll('tr'):
#         title = row.a.text  # Haal de titel van het nieuwsartikel
#         timestamp = row.td.text.strip()  # Haal de timestamp uit de td en strip eventuele spaties
        
#         # Split de timestamp in datum en tijd
#         date_data = timestamp.split(' ')

#         # Controleer of we een datum hebben
#         if len(date_data) == 1:  # Geen datum, alleen tijd
#             if previous_date is not None:  # Vul de vorige datum in
#                 date = previous_date
#                 time = date_data[0]  # De tijd
#             else:
#                 date = 'Unknown'  # Als er geen datum of vorige datum is
#                 time = date_data[0]  # De tijd
#         else:  # Als we een datum en tijd hebben
#             date = date_data[0]  # De datum
#             time = date_data[1]  # De tijd
#             previous_date = date  # Sla de huidige datum op voor het volgende artikel

#         # Voeg de gegevens toe aan de lijst
#         parsed_data.append([ticker, date, time, title])

#     # Zet de parsed data om in een DataFrame
#     df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

#     # Sentiment analyzer initialiseren
#     vader = SentimentIntensityAnalyzer()

#     # Functie om de sentimenten te berekenen
#     f = lambda title: vader.polarity_scores(title)['compound']
#     df['compound'] = df['title'].apply(f)

#     # Datumverwerking
#     def parse_date(date):
#         if date.lower() == 'today':
#             return pd.to_datetime('today').date()  # Zet 'Today' om naar de huidige datum
#         try:
#             return pd.to_datetime(date).date()  # Converteer normale datums
#         except Exception as e:
#             return None  # Als het niet lukt, return None

#     # Pas de datumverwerking toe op de 'date' kolom
#     df['date'] = df['date'].apply(parse_date)

#     return df

# # Voorbeeld van het aanroepen van de functie voor de 'SPY' ticker
# df_spy = fetch_and_analyze_sentiment("SPY")
# print(df_spy)

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

def fetch_and_analyze_sentiment(ticker: str, data: pd.DataFrame):
    """
    Haalt nieuwsartikelen op van Finviz voor een gegeven ticker, voert sentimentanalyse uit,
    en voegt de gemiddelde compound per dag toe aan de bestaande DataFrame.
    
    Args:
    ticker (str): De ticker voor de aandelen (bijv. 'SPY').
    data (pd.DataFrame): De bestaande DataFrame waaraan de 'compound' kolom toegevoegd wordt.
    
    Returns:
    pd.DataFrame: De originele DataFrame met de toegevoegde 'compound' kolom.
    """
    # Definieer de Finviz URL voor de gegeven ticker
    finviz_url = f'https://finviz.com/quote.ashx?t={ticker}&p=d'
    
    # Maak een lege lijst voor de opgehaalde gegevens
    parsed_data = []
    previous_date = None  # Houd de vorige datum bij

    # Verzend de verzoeken en verwerk de HTML-inhoud
    req = Request(url=finviz_url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    html = BeautifulSoup(response, features='html.parser')
    
    # Zoek de tabel met nieuwsitems
    news_table = html.find(id='news-table')
    
    # Itereer over de nieuwsitems in de tabel
    for row in news_table.findAll('tr'):
        title = row.a.text  # Haal de titel van het nieuwsartikel
        timestamp = row.td.text.strip()  # Haal de timestamp uit de td en strip eventuele spaties
        
        # Split de timestamp in datum en tijd
        date_data = timestamp.split(' ')

        # Controleer of we een datum hebben
        if len(date_data) == 1:  # Geen datum, alleen tijd
            if previous_date is not None:  # Vul de vorige datum in
                date = previous_date
                time = date_data[0]  # De tijd
            else:
                date = 'Unknown'  # Als er geen datum of vorige datum is
                time = date_data[0]  # De tijd
        else:  # Als we een datum en tijd hebben
            date = date_data[0]  # De datum
            time = date_data[1]  # De tijd
            previous_date = date  # Sla de huidige datum op voor het volgende artikel

        # Voeg de gegevens toe aan de lijst
        parsed_data.append([ticker, date, time, title])

    # Zet de parsed data om in een DataFrame
    df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

    # Sentiment analyzer initialiseren
    vader = SentimentIntensityAnalyzer()

    # Functie om de sentimenten te berekenen
    f = lambda title: vader.polarity_scores(title)['compound']
    df['compound'] = df['title'].apply(f)

    # Datumverwerking
    def parse_date(date):
        if date.lower() == 'today':
            return pd.to_datetime('today').strftime('%Y-%m-%d')  # Zet 'Today' om naar de huidige datum in YYYY-MM-DD
        try:
            return pd.to_datetime(date).strftime('%Y-%m-%d')  # Converteer normale datums naar YYYY-MM-DD formaat
        except Exception as e:
            return None  # Als het niet lukt, return None

    # Pas de datumverwerking toe op de 'date' kolom
    df['date'] = df['date'].apply(parse_date)

    # Groepeer de data op 'date' en bereken het gemiddelde van 'compound' per dag
    df_daily_avg = df.groupby('date')['compound'].mean().reset_index()

    # Zet de 'date' kolom als de index van de DataFrame en hernoem 'date' naar 'Date'
    df_daily_avg.set_index('date', inplace=True)
    df_daily_avg.index.name = 'Date'

    # Voeg de 'compound' kolom van df_daily_avg toe aan de originele data DataFrame
    # Omdat 'Date' nu de index is in beide DataFrames, kunnen we eenvoudig joinen
    data = data.join(df_daily_avg[['compound']], how='left')

    return data, df_daily_avg


