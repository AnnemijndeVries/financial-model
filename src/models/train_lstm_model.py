import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.losses import binary_crossentropy
from config.logging_config import setup_logger
import tensorflow as tf


# Logger initialiseren
logger = setup_logger()


def train_lstm_pipeline_exact(X, y, seq_length=30, test_size=0.2, pca_components=10, lasso_alpha=0.0001, 
                              learning_rate=0.0001, batch_size=32, epochs=50, correlation_threshold=0.9):
    logger.info("Start training pipeline")

    # Zet een vaste seed
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Stap 1: Schaal de data naar de range [0, 1] voor betere prestaties met LSTM
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)  # X moet een numpy array of dataframe zijn

    # Stap 2: Correlatiematrix-analyse (optioneel, voor redundante features)
    feature_df = pd.DataFrame(X_scaled, columns=[f"Feature_{i}" for i in range(X_scaled.shape[1])])
    correlation_matrix = feature_df.corr()
    correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)

    # Verwijder sterk gecorreleerde features
    feature_df_reduced = feature_df.drop(columns=correlated_features)
    X_reduced = feature_df_reduced.values
    print(f"Sterk gecorreleerde features verwijderd: {correlated_features}")

    # Voeg een target-kolom toe op basis van een drempelwaarde
    threshold = 0.05
    X['Significant_Change'] = (X['Change'].abs() > threshold).astype(int)

    # Stap 3: PCA toepassen (optioneel, voor dimensiereductie)
    pca = PCA(n_components=pca_components)  # Kies het aantal gewenste componenten
    X_pca = pca.fit_transform(X_reduced)
    print(f"Vorm van X_pca: {X_pca.shape}")
    print(f"Verklaarde variatie door componenten: {pca.explained_variance_ratio_}")

    # Stap 4: Pas Lasso toe voor feature selectie
    lasso = Lasso(alpha=lasso_alpha)  # Kies een lage alpha-waarde
    lasso.fit(X_pca, y)  # Pas Lasso aan op de PCA-componenten en de targets

    # Haal de geselecteerde features op
    selected_features = np.where(lasso.coef_ != 0)[0]  # Indices van niet-nul coëfficiënten
    print(f"Geselecteerde features (indexen): {selected_features}")

    # Filter X_pca om alleen de geselecteerde features te behouden
    X_selected = X_pca[:, selected_features] if selected_features.size > 0 else X_pca

    # Stap 5: Maak sequenties van tijdstappen voor het LSTM-model
    def create_sequences(X, y, seq_length=30):
        X_seqs, y_seqs = [], []
        for i in range(len(X) - seq_length):
            X_seqs.append(X[i:i + seq_length])
            y_seqs.append(y.iloc[i + seq_length])  # Gebruik iloc voor numerieke positie
        return np.array(X_seqs), np.array(y_seqs)

    # Gebruik 30 dagen als sequentielengte (je kunt dit aanpassen)
    X_seqs, y_seqs = create_sequences(X_selected, y, seq_length)

    # Stap 6: Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X_seqs, y_seqs, test_size=test_size, random_state=42)

    # Stap 7: Bouw het model
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),  
        LSTM(50, return_sequences=True),
        Dropout(0.1),                                      # Dropout om overfitting te voorkomen
        LSTM(50, return_sequences=False),
        Dropout(0.1),
        Dense(1, activation='sigmoid')                     # Outputlaag voor classificatie (voor binair)
    ])

    # Compileer het model met een lagere learning rate
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=binary_crossentropy, metrics=['accuracy'])

    # ModelCheckpoint callback
    checkpoint = ModelCheckpoint(
        filepath='best_model.keras',   # Gebruik de .keras extensie
        monitor='val_accuracy',        # Monitor de validatie-accuracy
        save_best_only=True,           # Alleen opslaan als dit het beste model tot nu toe is
        mode='max',                    # Kies het model met de hoogste validatie-accuracy
        verbose=1                      # Print een melding als het model wordt opgeslagen
    )

    # Train het model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[checkpoint]
    )
    
    logger.info("Pipeline voltooid")
    return model, history, scaler, pca