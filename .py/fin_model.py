import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import yfinance as yf

def linear_model_predictions(df, start):
    """
    Voegt voorspellingen toe aan een DataFrame gebaseerd op een lineair regressiemodel.

    Args:
    df (pd.DataFrame): Een bestaand DataFrame met een 'Close'-kolom.
    start (str): De startdatum voor eventuele aanpassing van historische data (YYYY-MM-DD).

    Returns:
    pd.DataFrame: Het originele DataFrame aangevuld met een kolom 'Linear_model' met voorspellingen.
    """
    # Controleer of er een MultiIndex is en maak deze plat
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Verwijder uren uit de datums en formatteer als YYYY-MM-DD
    df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
    df.index = pd.to_datetime(df.index)


    # Maak een kopie van de originele DataFrame
    df_copy = df.copy()

    # Voeg een 'Numbers'-kolom toe voor de lineaire regressie (intern gebruik)
    df_copy['Numbers'] = list(range(len(df_copy)))

    # Definieer de input (X) en output (y) voor de lineaire regressie
    X = np.array(df_copy[['Numbers']])
    y = df_copy['Close'].values

    # Train het lineaire regressiemodel
    lin_model = LinearRegression().fit(X, y)

    # Voeg voorspellingen toe aan het originele DataFrame
    df['Linear_model'] = lin_model.coef_[0] * (df_copy['Numbers'] + 1) + lin_model.intercept_

    return df

import pandas_ta as ta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV

def fin_model_1(df, param_grid=None):
    """
    Voegt kolommen toe voor signalen en probabiliteiten aan de originele dataset.

    Args:
    df (pd.DataFrame): De originele dataset.
    param_grid (dict): Parameter grid voor modeloptimalisatie.

    Returns:
    pd.DataFrame: De originele dataset met 'Signal_fin_model_1' en 'Probability_fin_model_1'.
    """
    # Reset index en verwerk data
    if df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Vereiste kolommen
    df2 = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']].copy()
    df2 = df2[df2.High != df2.Low]
    df2.reset_index(inplace=True, drop=True)

    # Bereken technische indicatoren met pandas_ta
    df2.ta.bbands(append=True, length=30, std=2)
    df2.ta.rsi(append=True, length=14)
    df2['atr'] = df2.ta.atr(length=14)

    # Hernoem kolommen
    df2.rename(columns={
        'BBL_30_2.0': 'bbl', 'BBM_30_2.0': 'bbm', 'BBU_30_2.0': 'bbh', 'RSI_14': 'rsi'
    }, inplace=True)
    df2['bb_width'] = (df2['bbh'] - df2['bbl']) / df2['bbm']
    df2['Signal_fin_model_1'] = 0

    # Bereken signalen
    for i in range(1, len(df2)):
        if (df2['Close'].iloc[i - 1] < df2['bbl'].iloc[i - 1] and
            df2['rsi'].iloc[i - 1] < 30 and
            df2['Close'].iloc[i] > df2['High'].iloc[i - 1] and
            df2['bb_width'].iloc[i] > 0.0015):
            df2.at[i, 'Signal_fin_model_1'] = 1

    # Bereid de data voor modeltraining
    df2 = df2.dropna()
    feature_columns = ['Close', 'High', 'Low', 'Open', 'Volume', 'bb_width', 'rsi', 'atr']
    X = df2[feature_columns]
    y = df2['Signal_fin_model_1']

    # Schalen en balanceren
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    sm = SMOTE(random_state=42)
    X_balanced, y_balanced = sm.fit_resample(X_scaled, y)

    # Modeloptimalisatie
    model = LogisticRegression()
    if param_grid is None:
        param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs'], 'max_iter': [100, 200]}

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc')
    grid_search.fit(X_balanced, y_balanced)

    # Calibreer het model en bereken probabiliteiten
    calibrated_model = CalibratedClassifierCV(grid_search.best_estimator_, method='sigmoid', cv=3)
    calibrated_model.fit(X_balanced, y_balanced)

    probabilities = calibrated_model.predict_proba(X_scaled)[:, 1]

    # Voeg de berekende kolommen toe aan df2
    df2['Probability_fin_model_1'] = probabilities

    # Combineer de resultaten terug met de originele dataset
    df = df.merge(
        df2[['Date', 'Signal_fin_model_1', 'Probability_fin_model_1']],
        on='Date',
        how='left'
    )
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index('Date', inplace=True)

    return df


# Importeer benodigde bibliotheken
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV



def fin_model_2(df, param_grid=None):
    """
    Voegt kolommen toe voor signalen en probabiliteiten aan de originele dataset.

    Args:
    df (pd.DataFrame): De originele dataset.
    param_grid (dict): Parameter grid voor modeloptimalisatie.

    Returns:
    pd.DataFrame: De originele dataset met 'Signal_fin_model_2' en 'Probability_fin_model_2'.
    """
    # Reset index en verwerk data
    if df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Vereiste kolommen
    df2 = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']].copy()
    df2 = df2[df2.High != df2.Low]
    df2.reset_index(inplace=True, drop=True)

    # Bereken technische indicatoren
    df2['sma_50'] = df2['Close'].rolling(window=50).mean()
    df2['sma_200'] = df2['Close'].rolling(window=200).mean()
    df2['momentum'] = df2['Close'] - df2['Close'].shift(10)
    df2['volatility'] = df2['Close'].rolling(window=20).std()
    df2['roc'] = ((df2['Close'] - df2['Close'].shift(12)) / df2['Close'].shift(12)) * 100

    # Signal-logica
    df2['Signal_fin_model_2'] = 0
    for i in range(1, len(df2)):
        if (df2['Close'].iloc[i] > df2['sma_50'].iloc[i] and
            df2['sma_50'].iloc[i] > df2['sma_200'].iloc[i] and
            df2['momentum'].iloc[i] > 0 and
            df2['volatility'].iloc[i] > df2['volatility'].mean()):
            df2.at[i, 'Signal_fin_model_2'] = 1

    # Bereid de data voor modeltraining
    df2 = df2.dropna()
    feature_columns = ['Close', 'High', 'Low', 'Open', 'Volume', 'momentum', 'volatility', 'roc']
    X = df2[feature_columns]
    y = df2['Signal_fin_model_2']

    # Schalen en balanceren
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sm = SMOTE(random_state=42)
    X_balanced, y_balanced = sm.fit_resample(X_scaled, y)

    # Modeloptimalisatie
    model = RandomForestClassifier(random_state=42)
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc')
    grid_search.fit(X_balanced, y_balanced)

    # Calibreer het model en bereken probabiliteiten
    calibrated_model = CalibratedClassifierCV(grid_search.best_estimator_, method='sigmoid', cv=3)
    calibrated_model.fit(X_balanced, y_balanced)

    probabilities = calibrated_model.predict_proba(X_scaled)[:, 1]

    # Voeg de berekende kolommen toe aan df2
    df2['Probability_fin_model_2'] = probabilities

    # Combineer de resultaten terug met de originele dataset
    df = df.merge(
        df2[['Date', 'Signal_fin_model_2', 'Probability_fin_model_2']],
        on='Date',
        how='left'
    )
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index('Date', inplace=True)

    return df

# import pandas as pd
# import numpy as np
# import ta
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.preprocessing import MinMaxScaler
# from imblearn.over_sampling import SMOTE


# def fin_model_3(df, param_grid=None):
#     """
#     Voegt candlestick-gebaseerde signalen toe aan de originele dataset en voorspelt marktrichting.

#     Args:
#     df (pd.DataFrame): De originele dataset.
#     param_grid (dict): Parameter grid voor modeloptimalisatie.

#     Returns:
#     pd.DataFrame: De originele dataset met 'Signal_fin_model_3' en 'Probability_fin_model_3'.
#     """
#     # Reset index en verwerk data
#     if df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
#         df = df.reset_index()

#     # Vereiste kolommen
#     df2 = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']].copy()
#     df2 = df2[df2.High != df2.Low]
#     df2.reset_index(inplace=True, drop=True)

#     # Bereken candle-attributen
#     df2['body_size'] = abs(df2['Close'] - df2['Open'])  # Grootte van de candle body
#     df2['upper_wick'] = df2['High'] - np.maximum(df2['Close'], df2['Open'])
#     df2['lower_wick'] = np.minimum(df2['Close'], df2['Open']) - df2['Low']
#     df2['candle_ratio'] = df2['body_size'] / (df2['High'] - df2['Low'])  # Relatieve grootte van de body

#     # Voeg een ATR-indicator toe
#     df2['atr'] = ta.volatility.average_true_range(high=df2['High'], low=df2['Low'], close=df2['Close'], window=14)

#     # Candlestick signalen
#     df2['Signal_fin_model_3'] = 0

#     for i in range(1, len(df2)):
#         # Bullish engulfing pattern
#         if (df2['Close'].iloc[i] > df2['Open'].iloc[i] and
#             df2['Close'].iloc[i - 1] < df2['Open'].iloc[i - 1] and
#             df2['Close'].iloc[i] > df2['Open'].iloc[i - 1] and
#             df2['Open'].iloc[i] < df2['Close'].iloc[i - 1]):
#             df2.at[i, 'Signal_fin_model_3'] = 1  # Koopsignaal

#         # Bearish engulfing pattern
#         elif (df2['Close'].iloc[i] < df2['Open'].iloc[i] and
#               df2['Close'].iloc[i - 1] > df2['Open'].iloc[i - 1] and
#               df2['Close'].iloc[i] < df2['Open'].iloc[i - 1] and
#               df2['Open'].iloc[i] > df2['Close'].iloc[i - 1]):
#             df2.at[i, 'Signal_fin_model_3'] = -1  # Verkoopsignaal

#         # Hammer pattern
#         elif (df2['body_size'].iloc[i] < df2['lower_wick'].iloc[i] and
#               df2['upper_wick'].iloc[i] < 0.3 * df2['lower_wick'].iloc[i]):
#             df2.at[i, 'Signal_fin_model_3'] = 1  # Koopsignaal

#         # Shooting star pattern
#         elif (df2['body_size'].iloc[i] < df2['upper_wick'].iloc[i] and
#               df2['lower_wick'].iloc[i] < 0.3 * df2['upper_wick'].iloc[i]):
#             df2.at[i, 'Signal_fin_model_3'] = -1  # Verkoopsignaal

#     # Bereid de data voor machine learning
#     df2 = df2.dropna()
#     feature_columns = ['Close', 'High', 'Low', 'Open', 'Volume', 'body_size', 'upper_wick', 'lower_wick', 'atr']
#     X = df2[feature_columns]
#     y = df2['Signal_fin_model_3']

#     # Schalen en balanceren
#     scaler = MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)

#     sm = SMOTE(random_state=42)
#     X_balanced, y_balanced = sm.fit_resample(X_scaled, y)

#     # Modeloptimalisatie
#     model = RandomForestClassifier(random_state=42)
#     if param_grid is None:
#         param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]}

#     grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc')
#     grid_search.fit(X_balanced, y_balanced)

#     # Calibreer het model en bereken probabiliteiten
#     calibrated_model = CalibratedClassifierCV(grid_search.best_estimator_, method='sigmoid', cv=3)
#     calibrated_model.fit(X_balanced, y_balanced)

#     probabilities = calibrated_model.predict_proba(X_scaled)[:, 1]

#     # Voeg de berekende kolommen toe aan df2
#     df2['Probability_fin_model_3'] = probabilities

#     # Combineer de resultaten terug met de originele dataset
#     df = df.merge(
#         df2[['Date', 'Signal_fin_model_3', 'Probability_fin_model_3']],
#         on='Date',
#         how='left'
#     )

#     df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#     df.set_index('Date', inplace=True)

#     return df

import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer, roc_auc_score

def fin_model_3(df, param_grid=None):
    """
    Voegt candlestick-gebaseerde signalen toe aan de originele dataset en voorspelt marktrichting.

    Args:
    df (pd.DataFrame): De originele dataset.
    param_grid (dict): Parameter grid voor modeloptimalisatie.

    Returns:
    pd.DataFrame: De originele dataset met 'Signal_fin_model_3' en 'Probability_fin_model_3'.
    """
    # Reset index en controleer de data
    if df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Vereiste kolommen
    required_columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"De dataset moet de volgende kolommen bevatten: {required_columns}")

    df2 = df[required_columns].copy()
    df2 = df2[df2.High != df2.Low].reset_index(drop=True)

    # Bereken candle-attributen
    df2['body_size'] = abs(df2['Close'] - df2['Open'])
    df2['upper_wick'] = df2['High'] - np.maximum(df2['Close'], df2['Open'])
    df2['lower_wick'] = np.minimum(df2['Close'], df2['Open']) - df2['Low']
    df2['candle_ratio'] = df2['body_size'] / (df2['High'] - df2['Low'])
    df2['atr'] = ta.volatility.average_true_range(
        high=df2['High'], low=df2['Low'], close=df2['Close'], window=14
    )

    # Voeg candlestick signalen toe
    df2['Signal_fin_model_3'] = 0
    for i in range(1, len(df2)):
        # Bullish engulfing
        if (df2['Close'].iloc[i] > df2['Open'].iloc[i] and
            df2['Close'].iloc[i - 1] < df2['Open'].iloc[i - 1] and
            df2['Close'].iloc[i] > df2['Open'].iloc[i - 1] and
            df2['Open'].iloc[i] < df2['Close'].iloc[i - 1]):
            df2.at[i, 'Signal_fin_model_3'] = 1
        # Bearish engulfing
        elif (df2['Close'].iloc[i] < df2['Open'].iloc[i] and
              df2['Close'].iloc[i - 1] > df2['Open'].iloc[i - 1] and
              df2['Close'].iloc[i] < df2['Open'].iloc[i - 1] and
              df2['Open'].iloc[i] > df2['Close'].iloc[i - 1]):
            df2.at[i, 'Signal_fin_model_3'] = -1
        # Hammer
        elif (df2['body_size'].iloc[i] < df2['lower_wick'].iloc[i] and
              df2['upper_wick'].iloc[i] < 0.3 * df2['lower_wick'].iloc[i]):
            df2.at[i, 'Signal_fin_model_3'] = 1
        # Shooting star
        elif (df2['body_size'].iloc[i] < df2['upper_wick'].iloc[i] and
              df2['lower_wick'].iloc[i] < 0.3 * df2['upper_wick'].iloc[i]):
            df2.at[i, 'Signal_fin_model_3'] = -1

    # Machine learning features
    df2 = df2.dropna()
    feature_columns = ['Close', 'High', 'Low', 'Open', 'Volume', 'body_size', 'upper_wick', 'lower_wick', 'atr']
    X = df2[feature_columns]
    y = df2['Signal_fin_model_3']

    # Schalen en balanceren
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    sm = SMOTE(random_state=42)
    X_balanced, y_balanced = sm.fit_resample(X_scaled, y)

    # Modeloptimalisatie
    model = RandomForestClassifier(random_state=42)
    if param_grid is None:
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]}

    # Gebruik een aangepaste ROC AUC scorer
    def custom_roc_auc(y_true, y_pred_proba):
        return roc_auc_score(y_true, y_pred_proba, multi_class='ovr')

    roc_auc_scorer = make_scorer(custom_roc_auc, needs_proba=True)

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring=roc_auc_scorer)
    grid_search.fit(X_balanced, y_balanced)

    # Calibreer het model
    calibrated_model = CalibratedClassifierCV(grid_search.best_estimator_, method='sigmoid', cv=3)
    calibrated_model.fit(X_balanced, y_balanced)

    # Voorspel probabiliteiten
    probabilities = calibrated_model.predict_proba(X_scaled)

    # Voeg probabiliteiten toe
    df2['Probability_fin_model_3'] = probabilities[:, 1]

    # Combineer met de originele dataset
    df = df.merge(
        df2[['Date', 'Signal_fin_model_3', 'Probability_fin_model_3']],
        on='Date',
        how='left'
    )
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index('Date', inplace=True)

    return df
