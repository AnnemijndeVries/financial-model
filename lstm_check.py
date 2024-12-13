print("Unieke waarden in y_train:", np.unique(y_train))
print("Unieke waarden in y_test:", np.unique(y_test))

# Aantal voorbeelden per klasse
print("Klassenverdeling in y_train:", np.bincount(y_train.astype(int)))

print("Min en Max waarden in X_train:", np.min(X_train), np.max(X_train))

