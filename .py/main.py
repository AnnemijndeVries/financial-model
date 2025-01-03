import technische_analyse
import fundamentele_analyse
import fin_model
import train_lstm_model
import yfinance as yf
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.losses import binary_crossentropy
from imblearn.over_sampling import SMOTE
from logging_config import setup_logger

# Logger initialiseren
logger = setup_logger()



pd.options.display.float_format = '{:.6f}'.format

# Parameters
lagg = 30
start = "2011-01-01"
seq_length = 90 
drempelwaarde = 0.5
ticker = "SPY"

if __name__ == "__main__":

    # Historische data op voor de S&P 500 tot de huidige dag
    data = yf.download(ticker, start=start, progress=False)  

    # Controleer of er een MultiIndex is en maak deze plat
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Verwijder uren uit de datums en formatteer als YYYY-MM-DD
    data.index = data.index.strftime('%Y-%m-%d')
    data.index = pd.to_datetime(data.index)

    # Technische analyse
    data_technische_analyse = technische_analyse.add_indicators(data, lagg)
    data_technische_analyse_2 = technische_analyse.add_change_features(data_technische_analyse)
    data_technische_analyse_3 = technische_analyse.add_economic_indicators_from_fred(data_technische_analyse_2)

    # Fundamentele analyse
    data_fundamentele = fundamentele_analyse.add_economic_indicators(data_technische_analyse_3)

    # Sentiment analyse
    df_sentiment = data_fundamentele.copy()

    # Linear model
    df_linearmodel = fin_model.linear_model_predictions(df_sentiment, start=start)

    # Financiele modellen
    df_fin_model_1 = fin_model.fin_model_1(df_linearmodel)
    df_fin_model_2 = fin_model.fin_model_2(df_fin_model_1)
    df_fin_model_3 = fin_model.fin_model_3(df_fin_model_2)

    # Drop lege rijen
    df_fin_model_3.dropna(inplace=True)

    # Stel de target kolom in
    target_column = 'Target'

    # Selecteer de features dynamisch uit de dataset (alle kolommen behalve de 'Target' en eventueel andere kolommen die je niet wilt gebruiken)
    feature_columns = [col for col in df_fin_model_3.columns if col != target_column and col != 'Date']

    # Zorg ervoor dat er geen duplicaten zijn, mocht je per ongeluk dezelfde feature meerdere keren toevoegen
    feature_columns = list(set(feature_columns))

    # Selecteer de features (X) en target (y)
    X = df_fin_model_3[feature_columns]
    y = df_fin_model_3[target_column]

    logger.info("Selected features: %s", X.columns.tolist())
    # print("Selected features:", X.columns)
    # print("Target:", y.name)

    # # Controleer of alle gewenste kolommen correct zijn toegevoegd
    # print("Kolommen in X:", X.columns)
    # print("Voorbeeld van X:", X.head())
    # print("Voorbeeld van y:", y.head())

    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(X, y)

    # Train model
    model, history, scaler, pca = train_lstm_model.train_lstm_pipeline_exact(
    X, 
    y, 
    seq_length=30, 
    test_size=0.2, 
    pca_components=10, 
    lasso_alpha=0.0001, 
    learning_rate=0.0001, 
    batch_size=32, 
    epochs=50
)



