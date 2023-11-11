
# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import shap  # SHAP es una biblioteca para explicar las predicciones

# Cargar los datos
data = pd.read_csv('C:/python/Cleaned-Data.csv')  # Asegurar de que la ruta al archivo es correcta

# Preparar los datos
# Seleccionamos las características y la variable objetivo
features = data.drop(columns=['Severity_Mild', 'Severity_Moderate', 'Severity_None', 'Severity_Severe', 'Country'])
target = data['Severity_Severe']

# Dividir los datos en conjunto de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Entrenar el modelo de árbol de decisión con balanceo de clases y una profundidad limitada
decision_tree_balanced = DecisionTreeClassifier(
    class_weight='balanced',
    max_depth=5,  # La profundidad se ha limitado para facilitar la visualización
    min_samples_split=50,
    min_samples_leaf=25,
    random_state=42
)
decision_tree_balanced.fit(X_train, y_train)

# Predecir las etiquetas en el conjunto de prueba
y_pred_balanced = decision_tree_balanced.predict(X_test)

# Visualizar el árbol de decisión
plt.figure(figsize=(20,10))
plot_tree(decision_tree_balanced, filled=True, feature_names=features.columns.tolist(), class_names=['No Severe', 'Severe'], rounded=True, proportion=False, precision=2)
plt.show()

# Obtener e imprimir la importancia de las características
importances_balanced = decision_tree_balanced.feature_importances_
feature_importances_balanced = pd.DataFrame(importances_balanced, index=features.columns, columns=["Importance"]).sort_values("Importance", ascending=False)
print(feature_importances_balanced)

# Generar el informe de clasificación con zero_division configurado en 0
report_balanced = classification_report(y_test, y_pred_balanced, target_names=['No Severe', 'Severe'], zero_division=0)
print(report_balanced)

# Inicializar el explainer de SHAP
shap_explainer = shap.TreeExplainer(decision_tree_balanced)

# Calcular los valores SHAP para el conjunto de entrenamiento
shap_values = shap_explainer.shap_values(X_train)

# Visualizar la importancia de las características con SHAP
shap.summary_plot(shap_values, X_train, plot_type="bar")
