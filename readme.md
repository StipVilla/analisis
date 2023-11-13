# Análisis de Datos con Árbol de Decisión Balanceado y SHAP

Este repositorio contiene un análisis de datos utilizando un modelo de árbol de decisión balanceado y la biblioteca SHAP para explicar las predicciones. El conjunto de datos utilizado es 'Cleaned-Data.csv'.

## Contenido del Repositorio

- **Jupyter Notebook: [analysis.ipynb](analysis.ipynb)**
  - Importa las bibliotecas necesarias.
  - Carga y prepara los datos.
  - Divide los datos en conjuntos de entrenamiento y prueba.
  - Entrena un modelo de árbol de decisión balanceado.
  - Visualiza el árbol de decisión.
  - Calcula e imprime la importancia de las características.
  - Genera un informe de clasificación con zero_division configurado en 0.
  - Inicializa el explainer de SHAP.
  - Calcula los valores SHAP para el conjunto de entrenamiento.
  - Visualiza la importancia de las características con SHAP.

## Uso

1. Abre el archivo [analysis.ipynb](analysis.ipynb) en un entorno Jupyter Notebook.
2. Asegúrate de tener las bibliotecas necesarias instaladas (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `shap`).
3. Ejecuta cada celda del notebook en orden.

## Configuración del Entorno

- Se asume que tienes un entorno de Python 3.x configurado.
- Puedes instalar las bibliotecas necesarias ejecutando `pip install pandas numpy scikit-learn matplotlib shap`.

