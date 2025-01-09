import analyses.technische_analyse as technische_analyse
import analyses.fundamentele_analyse as fundamentele_analyse
import models.train_lstm_model as train_lstm_model
import analyses.fin_model as fin_model
import yfinance as yf
import pandas as pd
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from config.logging_config import setup_logger

# import logging
# import sys
# import os

# os.environ['PYTHONIOENCODING'] = 'utf-8'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Verberg INFO en WARNING van TensorFlow

# # Logger instellen
# logger = setup_logger()

# # Omleiding van stdout en stderr naar logger
# class StreamToLogger:
#     def __init__(self, logger, level):
#         self.logger = logger
#         self.level = level

#     def write(self, message):
#         if message.strip():  # Negeer lege regels
#             self.logger.log(self.level, message.strip())

#     def flush(self):  # Voor compatibiliteit
#         pass

# # Omleiden van stdout en stderr
# sys.stdout = StreamToLogger(logger, logging.INFO)
# sys.stderr = StreamToLogger(logger, logging.ERROR)

# # TensorFlow logs integreren in de logger
# tf_logger = tf.get_logger()
# tf_logger.setLevel(logging.INFO)

# # Verwijder bestaande handlers (om dubbele logs te voorkomen)
# if tf_logger.hasHandlers():
#     tf_logger.handlers.clear()

# # Voeg logger-handlers toe aan TensorFlow
# for handler in logger.handlers:
#     tf_logger.addHandler(handler)

import logging
import sys
import os
import tensorflow as tf

# Forceer UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Minimaliseer TensorFlow-uitvoer (alleen ERROR en CRITICAL)

# Logger instellen
def setup_logger():
    logger = logging.getLogger("ModelTrainingLogger")
    logger.setLevel(logging.INFO)

    # Controleer of er al handlers zijn (voorkom dubbele handlers)
    if not logger.handlers:
        # StreamHandler instellen voor console-uitvoer
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
        )
        logger.addHandler(stream_handler)

        # FileHandler instellen voor logbestand
        file_handler = logging.FileHandler("model_training.log", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()

# Omleiding van stdout en stderr naar logger
class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():  # Negeer lege regels
            # Voorkom lus door te controleren of de boodschap niet al van de logger komt
            if not message.startswith("["):  # Vermijd berichten die al gelogd zijn
                self.logger.log(self.level, message.strip())

    def flush(self):
        pass


# Omleiden van stdout en stderr
sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

# TensorFlow logs integreren in de logger
tf_logger = tf.get_logger()
tf_logger.setLevel(logging.ERROR)  # Alleen ERROR-niveau berichten van TensorFlow

# Verwijder bestaande handlers (om dubbele logs te voorkomen)
if tf_logger.hasHandlers():
    tf_logger.handlers.clear()

# Voeg logger-handlers toe aan TensorFlow logger
for handler in logger.handlers:
    tf_logger.addHandler(handler)


# Instellen van pandas-weergave
pd.options.display.float_format = '{:.6f}'.format

# Parameters
lagg = 30
start = "2011-01-01"
seq_length = 90
drempelwaarde = 0.5
ticker = "SPY"

# if __name__ == "__main__":

#     # Historische data op voor de S&P 500 tot de huidige dag
#     data = yf.download(ticker, start=start, progress=False)  

#     # Controleer of er een MultiIndex is en maak deze plat
#     if isinstance(data.columns, pd.MultiIndex):
#         data.columns = data.columns.get_level_values(0)

#     # Verwijder uren uit de datums en formatteer als YYYY-MM-DD
#     data.index = data.index.strftime('%Y-%m-%d')
#     data.index = pd.to_datetime(data.index)

#     # Technische analyse
#     data_technische_analyse = technische_analyse.add_indicators(data, lagg)
#     data_technische_analyse_2 = technische_analyse.add_change_features(data_technische_analyse)
#     data_technische_analyse_3 = technische_analyse.add_economic_indicators_from_fred(data_technische_analyse_2)

#     # Fundamentele analyse
#     data_fundamentele = fundamentele_analyse.add_economic_indicators(data_technische_analyse_3)

#     # Sentiment analyse
#     df_sentiment = data_fundamentele.copy()

#     # Linear model
#     df_linearmodel = fin_model.linear_model_predictions(df_sentiment, start=start)

#     # Financiele modellen
#     df_fin_model_1 = fin_model.fin_model_1(df_linearmodel)
#     df_fin_model_2 = fin_model.fin_model_2(df_fin_model_1)
#     df_fin_model_3 = fin_model.fin_model_3(df_fin_model_2)

#     # Drop lege rijen
#     df_fin_model_3.dropna(inplace=True)

#     # Stel de target kolom in
#     target_column = 'Target'

#     # Selecteer de features dynamisch uit de dataset (alle kolommen behalve de 'Target' en eventueel andere kolommen die je niet wilt gebruiken)
#     feature_columns = [col for col in df_fin_model_3.columns if col != target_column and col != 'Date']

#     # Zorg ervoor dat er geen duplicaten zijn, mocht je per ongeluk dezelfde feature meerdere keren toevoegen
#     feature_columns = list(set(feature_columns))

#     # Selecteer de features (X) en target (y)
#     X = df_fin_model_3[feature_columns]
#     y = df_fin_model_3[target_column]

#     logger.info("Selected features: %s", X.columns.tolist())
#     # print("Selected features:", X.columns)
#     # print("Target:", y.name)

#     # # Controleer of alle gewenste kolommen correct zijn toegevoegd
#     # print("Kolommen in X:", X.columns)
#     # print("Voorbeeld van X:", X.head())
#     # print("Voorbeeld van y:", y.head())

#     sm = SMOTE(random_state=42)
#     X, y = sm.fit_resample(X, y)

#     # Train model
#     model, history, scaler, pca = train_lstm_model.train_lstm_pipeline_exact(
#     X, 
#     y, 
#     seq_length=30, 
#     test_size=0.2, 
#     pca_components=10, 
#     lasso_alpha=0.0001, 
#     learning_rate=0.0001, 
#     batch_size=32, 
#     epochs=50
# )


# ... Imports blijven hetzelfde ...

# if __name__ == "__main__":
#     logger.info("Script gestart")

#     # Historische data ophalen
#     logger.info("Start met het downloaden van data voor ticker: %s", ticker)
#     try:
#         data = yf.download(ticker, start=start, progress=False)  
#         logger.info("Data succesvol gedownload, aantal rijen: %d, kolommen: %d", len(data), len(data.columns))
#     except Exception as e:
#         logger.error("Fout bij het downloaden van data: %s", e)
#         raise

#     # Controleer MultiIndex
#     if isinstance(data.columns, pd.MultiIndex):
#         logger.debug("Data bevat een MultiIndex, wordt omgezet naar platte index")
#         data.columns = data.columns.get_level_values(0)

#     # Datumformat aanpassen
#     logger.debug("Formatteren van datums")
#     data.index = data.index.strftime('%Y-%m-%d')
#     data.index = pd.to_datetime(data.index)

#     # Technische analyse
#     logger.info("Start technische analyse")
#     try:
#         data_technische_analyse = technische_analyse.add_indicators(data, lagg)
#         logger.info("Indicatoren toegevoegd: %s", data_technische_analyse.columns[-5:].tolist())  # Laatste 5 indicatoren
#         data_technische_analyse_2 = technische_analyse.add_change_features(data_technische_analyse)
#         data_technische_analyse_3 = technische_analyse.add_economic_indicators_from_fred(data_technische_analyse_2)
#         logger.info("Technische analyse succesvol uitgevoerd")
#     except Exception as e:
#         logger.error("Fout bij technische analyse: %s", e)
#         raise

#     # Fundamentele analyse
#     logger.info("Start fundamentele analyse")
#     try:
#         data_fundamentele = fundamentele_analyse.add_economic_indicators(data_technische_analyse_3)
#         logger.info("Fundamentele indicatoren toegevoegd: %s", data_fundamentele.columns[-5:].tolist())  # Laatste 5 indicatoren
#     except Exception as e:
#         logger.error("Fout bij fundamentele analyse: %s", e)
#         raise

#     # Sentiment analyse
#     logger.info("Kopiëren data voor sentimentanalyse")
#     df_sentiment = data_fundamentele.copy()

#     # Financiële modellen
#     logger.info("Start financiële modellen")
#     try:
#         df_linearmodel = fin_model.linear_model_predictions(df_sentiment, start=start)
#         logger.info("Lineair model succesvol getraind")
#         df_fin_model_1 = fin_model.fin_model_1(df_linearmodel)
#         df_fin_model_2 = fin_model.fin_model_2(df_fin_model_1)
#         df_fin_model_3 = fin_model.fin_model_3(df_fin_model_2)
#         logger.info("Financiële modellen succesvol uitgevoerd")
#     except Exception as e:
#         logger.error("Fout bij financiële modellen: %s", e)
#         raise

#     # Controleer lege rijen
#     initial_rows = len(df_fin_model_3)
#     df_fin_model_3.dropna(inplace=True)
#     dropped_rows = initial_rows - len(df_fin_model_3)
#     logger.info("Aantal rijen verwijderd vanwege NaN-waarden: %d", dropped_rows)

#     # Selecteer features en target
#     target_column = 'Target'
#     feature_columns = [col for col in df_fin_model_3.columns if col != target_column and col != 'Date']
#     feature_columns = list(set(feature_columns))

#     logger.info("Aantal geselecteerde features: %d", len(feature_columns))
#     logger.info("Geselecteerde features: %s", feature_columns[:10])  # Laat de eerste 10 zien

#     X = df_fin_model_3[feature_columns]
#     y = df_fin_model_3[target_column]

#     # SMOTE toepassen
#     logger.info("Toepassen van SMOTE om data te balanceren")
#     sm = SMOTE(random_state=42)
#     X, y = sm.fit_resample(X, y)
#     logger.info("SMOTE succesvol toegepast, nieuwe datasetgrootte: X=%d, y=%d", len(X), len(y))

#     # Train model
#     logger.info("Start training van het LSTM-model")
#     try:
#         model, history, scaler, pca = train_lstm_model.train_lstm_pipeline_exact(
#             X, 
#             y, 
#             seq_length=30, 
#             test_size=0.2, 
#             pca_components=10, 
#             lasso_alpha=0.0001, 
#             learning_rate=0.0001, 
#             batch_size=32, 
#             epochs=50
#         )
#         logger.info("Modeltraining succesvol afgerond")
        
#         # Log PCA-resultaten
#         logger.info("Vorm van X_pca: %s", X.shape)
#         logger.info("Verklaarde variatie door componenten: %s", pca.explained_variance_ratio_.tolist())
        
#         logger.info(
#             "Eindtrainingsverlies: %.4f, Eindvalideringsverlies: %.4f, "
#             "Eindtrainingsaccuratie: %.4f, Eindvalideringsaccuratie: %.4f",
#             history.history['loss'][-1],
#             history.history['val_loss'][-1],
#             history.history['accuracy'][-1],
#             history.history['val_accuracy'][-1]
#         )

#     except Exception as e:
#         logger.error("Fout tijdens modeltraining: %s", e)
#         raise

#     logger.info("Script succesvol voltooid")

if __name__ == "__main__":
    logger.info("Script gestart")

    # Historische data ophalen
    try:
        data = yf.download(ticker, start=start, progress=False)  
        logger.info("Data gedownload: %d rijen, %d kolommen", len(data), len(data.columns))
    except Exception as e:
        logger.error("Fout bij het downloaden van data: %s", e)
        raise

    # Controleer MultiIndex en pas datumformat aan
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        logger.debug("MultiIndex omgezet naar platte index")
    data.index = pd.to_datetime(data.index.strftime('%Y-%m-%d'))

    # Technische analyse
    try:
        data_technische_analyse = technische_analyse.add_indicators(data, lagg)
        data_technische_analyse_2 = technische_analyse.add_change_features(data_technische_analyse)
        data_technische_analyse_3 = technische_analyse.add_economic_indicators_from_fred(data_technische_analyse_2)
        logger.info("Technische analyse uitgevoerd, indicatoren toegevoegd: %s", data_technische_analyse_3.columns[-5:].tolist())
    except Exception as e:
        logger.error("Fout bij technische analyse: %s", e)
        raise

    # Fundamentele analyse
    try:
        data_fundamentele = fundamentele_analyse.add_economic_indicators(data_technische_analyse_3)
        logger.info("Fundamentele indicatoren toegevoegd: %s", data_fundamentele.columns[-5:].tolist())
    except Exception as e:
        logger.error("Fout bij fundamentele analyse: %s", e)
        raise

    # Sentiment analyse
    df_sentiment = data_fundamentele.copy()

    # Financiële modellen
    try:
        df_linearmodel = fin_model.linear_model_predictions(df_sentiment, start=start)
        df_fin_model_1 = fin_model.fin_model_1(df_linearmodel)
        df_fin_model_2 = fin_model.fin_model_2(df_fin_model_1)
        df_fin_model_3 = fin_model.fin_model_3(df_fin_model_2)
        logger.info("Financiële modellen uitgevoerd, aantal rijen: %d, aantal kolommen: %d", len(df_fin_model_3), len(df_fin_model_3.columns))
    except Exception as e:
        logger.error("Fout bij financiële modellen: %s", e)
        raise

    # Verwijder lege rijen en log resultaat
    initial_rows = len(df_fin_model_3)
    df_fin_model_3.dropna(inplace=True)
    dropped_rows = initial_rows - len(df_fin_model_3)
    logger.info("NaN-waarden verwijderd, %d rijen overgebleven, %d rijen verwijderd", len(df_fin_model_3), dropped_rows)

    # Selecteer features en target
    target_column = 'Target'
    feature_columns = [col for col in df_fin_model_3.columns if col != target_column and col != 'Date']
    feature_columns = list(set(feature_columns))
    logger.info("Features geselecteerd (%d totaal), voorbeelden: %s", len(feature_columns), feature_columns[:5])

    X = df_fin_model_3[feature_columns]
    y = df_fin_model_3[target_column]

    # SMOTE toepassen
    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(X, y)
    logger.info("SMOTE toegepast, nieuwe datasetgrootte: X=%d, y=%d", len(X), len(y))

    # Train model
    try:
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
        logger.info("Model getraind, eindtrainingsverlies: %.4f, eindvalideringsverlies: %.4f", history.history['loss'][-1], history.history['val_loss'][-1])
        logger.info("Eindtrainingsaccuratie: %.4f, eindvalideringsaccuratie: %.4f", history.history['accuracy'][-1], history.history['val_accuracy'][-1])
        logger.info("PCA-resultaten: verklaarde variatie door componenten: %s", pca.explained_variance_ratio_.tolist())
    except Exception as e:
        logger.error("Fout tijdens modeltraining: %s", e)
        raise

    logger.info("Script voltooid")
