import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------- CONFIG -----------------------
CSV_PATH = os.path.join(os.path.dirname(__file__), "iris.csv")  
TEST_SIZE = 0.2
RANDOM_STATE = 42

def main():
    df = pd.read_csv(CSV_PATH)
    print("Primeras filas del dataset:")
    print(df.head(), "\n")

    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
    y = df["species"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2, weights="uniform")
    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy (k=5, euclidiana): {acc:.4f}")
    print("\nReporte de clasificación (baseline):")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusión (baseline):\n", cm, "\n")

    param_grid = {
        "n_neighbors": list(range(1, 16)),
        "metric": ["minkowski", "manhattan"],
        "p": [1, 2]  
    }
    grid = GridSearchCV(
        KNeighborsClassifier(weights="uniform"),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)

    print("Mejores hiperparámetros:", grid.best_params_)
    print(f"Mejor accuracy (CV): {grid.best_score_:.4f}")

    best_knn = grid.best_estimator_
    y_pred_best = best_knn.predict(X_test_scaled)
    acc_best = accuracy_score(y_test, y_pred_best)
    print(f"Accuracy en test (mejor modelo): {acc_best:.4f}\n")

    print("Reporte de clasificación (mejor modelo):")
    print(classification_report(y_test, y_pred_best))

    cm_best = confusion_matrix(y_test, y_pred_best)
    print("Matriz de confusión (mejor modelo):\n", cm_best, "\n")

    X_scaled = scaler.fit_transform(X)  
    cv_scores = cross_val_score(best_knn, X_scaled, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"Cross-Validation Accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    ks = list(range(1, 16))
    accs = []
    for k in ks:
        model = KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=2, weights="uniform")
        model.fit(X_train_scaled, y_train)
        y_val = model.predict(X_test_scaled)
        accs.append(accuracy_score(y_test, y_val))

    plt.figure()
    plt.plot(ks, accs, marker="o")
    plt.title("Accuracy vs k (distancia euclidiana)")
    plt.xlabel("k (n_neighbors)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_vs_k.png", dpi=150)
    plt.show()

    plt.figure()
    plt.imshow(cm_best, interpolation="nearest")
    plt.title("Matriz de confusión (mejor modelo)")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.colorbar()
    import numpy as np
    tick_marks = np.arange(len(np.unique(y)))
    plt.xticks(tick_marks, np.unique(y), rotation=45)
    plt.yticks(tick_marks, np.unique(y))
    for i in range(cm_best.shape[0]):
        for j in range(cm_best.shape[1]):
            plt.text(j, i, cm_best[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()

    print("Se guardaron las gráficas: accuracy_vs_k.png y confusion_matrix.png")

if __name__ == "__main__":
    main()
