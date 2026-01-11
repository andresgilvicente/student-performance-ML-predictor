"""

Este script se empleará para almacenar todas las funciones utilizadas en el proyecto final de Aprendizaje Automático. Estas funciones nos 
permitirán hacer codificación de valores categóricos, imputar valores nulos, mostrar gráficos o visualizaciones útiles para el análisis, 
incluso dividir el dataset en train y validation. También se almacenarán las clases de algunos modelos de aprendizaje automático, así como 
funciones dedicadas a la validación cruzada y búsqueda de parámetros óptimos en los modelos.

Autor: Andrés Gil Vicente
Asignatura: Aprendizaje Automático
Curso: 2º A iMAT


""" 

######################################################################################################################################################

# LIBRERÍAS NECESARIAS

import pandas as pd
import numpy as np
import random
import os
from scipy.stats import mode
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, 
                             classification_report, mean_absolute_error,mean_squared_error, r2_score)
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeRegressor, export_graphviz, DecisionTreeClassifier
from io import StringIO
from IPython.display import Image  
import pydotplus
import seaborn as sns
import math

######################################################################################################################################################

# PREPROCESADO DE LOS DATOS

def dummy_encoding(X, categorical_columns):
    """

    Realiza dummy encoding sobre las columnas categóricas del DataFrame X.

    Args:
        X (pd.DataFrame): DataFrame de entrada.
        categorical_columns (list): Lista con los nombres de columnas categóricas a codificar.

    Returns:
        X_transformed (pd.DataFrame): con las columnas categóricas codificadas como dummies (una menos por columna).

    """
    X_transformed = X.copy()

    for col in categorical_columns:

        # Obtenemos los valores únicos de la columna, ordenados para consistencia
        unique_values = sorted(X_transformed[col].dropna().unique())

        # Dummy encoding: eliminamos la primera categoría (para evitar colinealidad)
        categories_to_encode = unique_values[1:]

        # Creamos una nueva columna por cada categoría distinta (sin contar la que hemos quitado)
        for category in categories_to_encode:
            new_col_name = f"{col}_{category}"
            X_transformed[new_col_name] = (X_transformed[col] == category).astype(int)  # Esto genera 1s o 0s en función de lo que corresponda

        # Eliminamos la columna original
        X_transformed.drop(columns=[col], inplace=True)

    return X_transformed

def gestionar_valores_nulos(X):
    """

    Gestiona los valores nulos en un DataFrame realizando imputación según el tipo de variable.

    Para cada columna del DataFrame:
    - Si es categórica (tipo "object"), imputa los valores nulos con la moda.
    - Si es numérica, imputa los valores nulos con la mediana.

    Args:
        X (pd.DataFrame): DataFrame de entrada con posibles valores nulos.

    Returns:
        pd.DataFrame: DataFrame con los valores nulos imputados.
    
    """
    # Hacemos una copia del dataframe
    X = X.copy()

    # Iteramos por todas las columnas del Dataframe
    for col in X.columns:

        # Comprobamos si se trata de una columna categórica o numérica para hacer la imputación
        if X[col].dtype == "object":
            X.loc[X[col].isna(), col] = X[col].mode()[0]

        else:
            X.loc[X[col].isna(), col] = int(X[col].median())  # Podríamos hacerlo con la media también

    # Devolvemos el dataframe ya limpio
    return X

def ajustar_outliers_faltas(X, th=150):
    """

    Ajusta los valores atípicos (outliers) en la columna "faltas" de un DataFrame,
    aplicando un límite superior (clip) para reducir su impacto.

    Args:
        X (pd.DataFrame): DataFrame de entrada que contiene la columna "faltas" con valores numéricos.
        th (int): Threashold para hacer el clip superior de la columna de "faltas".

    Returns:
        pd.DataFrame: Una copia del DataFrame original con la columna "faltas" ajustada,
        donde los valores mayores a 150 han sido limitados a ese valor.
    
    """
    # Hacemos una copia del dataframe
    df = X.copy()

    # Aplicamos el upper clip con el threshold establecido previamente
    df.faltas = df["faltas"].clip(upper=th)

    # Devolvemos el dataframe
    return df

def ajustar_valores_decimales_faltas(X):
    """

    Ajusta la columna "faltas" del DataFrame eliminando valores decimales.

    Esta función convierte todos los valores de la columna "faltas" a enteros.
    Está pensada para corregir casos en los que, por errores en el procesamiento o imputación,
    aparecen valores decimales en una variable que debería ser entera (como un recuento de faltas).

    Args:
        X (pd.DataFrame): DataFrame de entrada que contiene la columna "faltas".

    Returns:
        pd.DataFrame: DataFrame con la columna "faltas" convertida a valores enteros.
    
    
    """
    # Hacemos una copia del dataframe
    df = X.copy()

    # Convertimos todos los valores de faltas a enteros, para que no quede ningún decimal (valor imposible)
    df["faltas"] = df["faltas"].apply(lambda x: int(x))

    # Devolvemos el dataframe ya ajustado
    return df

def dividir_train_validation(df, proporcion_train=0.75, random_state=48):
    """

    Divide un DataFrame en conjuntos de entrenamiento y validación con mezcla aleatoria de las filas.

    Args:
        df (pd.DataFrame): El DataFrame original con los datos.
        variable_objetivo (str): El nombre de la columna que se desea predecir.
        proporcion_train (float, optional): Proporción de los datos para entrenamiento. El resto se usará para validación. Por defecto es 0.75 (875% entrenamiento).
        random_state (int, optional): Semilla para hacer reproducible la mezcla aleatoria. Por defecto es 42.

    Returns:
        df_train (pd.DataFrame): Dataframe del conjunto de entrenamiento.
        df_val (pd.DataFrame): Dataframe del conjunto de validación.

    """

    # Mezclamos aleatoriamente el DataFrame
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Calculamos el índice de corte
    corte = int(len(df_shuffled) * proporcion_train)

    # Dividimos el dataframe en subconjuntos
    df_train = df_shuffled.iloc[:corte]
    df_val = df_shuffled.iloc[corte:]

    # Devolvemos todos los subconjuntos obtenidos
    return df_train, df_val

def estandarizar_dataset(X, cols_estandarizar, media_train, std_train):
    """
    Normaliza columnas específicas de un DataFrame usando normalización Z-score:
    (valor - media) / desviación estándar.

    Args:
        X (pandas.DataFrame): El conjunto de datos original.
        cols_normalizar (lista de str) Nombres de las columnas a normalizar.

    Returns:
        (DataFrame): con las columnas especificadas normalizadas.
    
    """
    # Hacemos una copia del dataframe
    df = X.copy()

    # Hacemos la normalización de forma vectorial, es decir columna a columna
    df[cols_estandarizar] = (df[cols_estandarizar] - media_train) / std_train

    # Devolvemos el dataframe ya normalizado
    return df

def guardar_df_to_csv(X, carpeta: str, nombre: str):
    """

    Guarda un DataFrame en un archivo CSV en la carpeta especificada con el nombre dado.

    Args:
        X (pd.DataFrame): DataFrame a guardar.
        carpeta (str): Ruta de la carpeta donde se guardará el archivo.
        nombre (str): Nombre del archivo CSV (sin extensión).

    Returns:
        None

    """

    # Verificamos que la carpeta existe, y si no, la creamos
    if not os.path.exists(carpeta):
        os.makedirs(carpeta, exist_ok=True)

    # Construimos la ruta completa del archivo
    ruta_completa = os.path.join(carpeta, f"{nombre}.csv")

    # Guardar el DataFrame en formato CSV
    X.to_csv(ruta_completa, index=False)
    
def eliminar_anomalias_razon(X):
    """

    Corrige una anomalía de transcripción en la columna "razon" del DataFrame.

    Esta función reemplaza el valor incorrecto "otras" por el valor correcto "otros"
    en la columna "razon", asumiendo que se trata de un error de transcripción categórica.

    Args:
        X (pd.DataFrame): DataFrame que contiene la columna "razon".

    Returns:
        pd.DataFrame: DataFrame con la columna "razon" corregida.

    """
    # Hacemos una copia del dataframe
    df = X.copy()

    # Imputamos el valor que debería ser el correcto (eliminando el error de transcripción)
    df.loc[df["razon"] == "otras", "razon"] = "otros"

    # Devolvemos el dataframe ya corregido
    return df

def plot_residuals(data, output_column, prediction_column):  
    """

    Genera gráficos de los residuos de un modelo respecto a todas las variables del DataFrame.

    Utiliza diagramas de caja para variables categóricas y diagramas de dispersión para 
    variables continuas. También incluye un histograma y un gráfico Q-Q de los residuos, 
    así como gráficos de residuos vs. valores reales y residuos vs. predicciones.

    Args:
        data (pd.DataFrame): DataFrame que contiene los datos.
        output_column (str): Nombre de la columna de salida real (target).
        prediction_column (str): Nombre de la columna con las predicciones del modelo.

    Returns:
        None

    """
    # Calculamos los residuos: diferencia entre valores reales y predichos
    residuals = data[output_column]-data[prediction_column] 
    
    num_features = len(data.columns) - 2  # Exclude output and prediction columns

    # Determinamos número de filas y columnas para los subplots
    num_rows = int(np.ceil(np.sqrt(num_features + 4)))  # Add 4 for histogram, Q-Q plot, true output vs residuals, and predictions vs residuals
    num_cols = int(np.ceil((num_features + 4) / num_rows))

    # Gráfico 1: Histograma de los residuos
    plt.figure(figsize=(5 * num_cols, 4 * num_rows))
    plt.subplot(num_rows, num_cols, 1)
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')

    # Gráfico 2: QQ-plot de los residuos
    plt.subplot(num_rows, num_cols, 2)
    stats.probplot(residuals, dist="norm", plot=plt)       
    plt.title('Q-Q Plot of Residuals')

    # Gráfico 3: Residuos vs valores reales
    plt.subplot(num_rows, num_cols, 3)
    plt.scatter(data[output_column], residuals, alpha=0.5)     
    plt.title('Residuals vs True Output')
    plt.xlabel('True Output')
    plt.ylabel('Residuals')

    # Gráfico 4: Residuos vs predicciones
    plt.subplot(num_rows, num_cols, 4)
    plt.scatter(data[prediction_column], residuals, alpha=0.5)   
    plt.title('Residuals vs Predictions')
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')

    # Gráficos 5 en adelante: Residuos vs cada variable de entrada
    for i, col in enumerate(data.columns):
        if col not in [output_column, prediction_column]:
            plt.subplot(num_rows, num_cols, i + 5)
            plt.scatter(data[col], residuals, alpha=0.5)   
            plt.title(f'Residuals vs {col}')
            plt.xlabel(col)
            plt.ylabel('Residuals')

    plt.tight_layout()
    plt.show()

def plot_real_vs_pred(y_true, y_pred, title='Valores reales vs predichos'):
    """
    
    Dibuja un gráfico de dispersión comparando los valores reales frente a los valores predichos.

    Este gráfico es útil para evaluar visualmente el rendimiento de un modelo de regresión.
    La línea discontinua roja representa la línea ideal (donde y_pred = y_true).

    Args:
        y_true (array-like): Valores reales de la variable objetivo.
        y_pred (array-like): Valores predichos por el modelo.
        title (str, opcional): Título del gráfico. Por defecto, 'Valores reales vs predichos'.

    Returns:
        None
    
    """
    # Creamos la figura y el gráfico de dispersión
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)

    # Dibujamos la línea ideal (cuando la predicción es perfecta)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Línea ideal')

    # Etiquetas y formato
    plt.xlabel('Valor real')
    plt.ylabel('Valor predicho')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def sacar_escalado_train(X, columnas_estadarizar):
    """

    Calcula y guarda las estadísticas de escalado (media y desviación estándar) para un conjunto de columnas.

    Esta función recorre las columnas especificadas, calcula la media y desviación estándar
    de cada una, y guarda estos valores en un archivo CSV dentro de la carpeta "processed_data".
    Sirve como paso previo para aplicar el mismo escalado posteriormente sobre otros conjuntos (por ejemplo, validación o test).

    Args:
        X (pd.DataFrame): DataFrame de entrada con las variables originales.
        columnas_estadarizar (list of str): Lista con los nombres de las columnas a estandarizar.

    Returns:
        dict: Diccionario con la media y desviación estándar por columna, con el siguiente formato:
              {
                  'col1': {'mean': ..., 'std': ...},
                  'col2': {'mean': ..., 'std': ...},
                  ...
              }
    
    """
    stats = {}
    df = X.copy()

    # Calculamos media y desviación típica para cada columna
    for col in columnas_estadarizar:
        media = df[col].mean()
        std = df[col].std()
        stats[col] = {'mean': media, 'std': std}

    # Guardamos los resultados como CSV
    os.makedirs("processed_data", exist_ok=True)
    pd.DataFrame(stats).T.to_csv("processed_data/stats_estadarizar_train.csv")

    # Devolvemos el diccionario
    return stats

######################################################################################################################################################

# KNN REGRESIÓN

def minkowski_distance(a, b, p=2):
    """

    Calcula la distancia de Minkowski entre dos arrays.

    La distancia de Minkowski es una medida de distancia generalizada que incluye
    casos conocidos como la distancia euclídea (p=2) o la distancia Manhattan (p=1).

    Args:
        a (np.ndarray): Primer array.
        b (np.ndarray): Segundo array.
        p (int, opcional): Grado de la distancia de Minkowski. Por defecto es 2 (distancia euclídea).

    Returns:
        float: Distancia de Minkowski entre los arrays "a" y "b".

    """

    # Devolvemos la distancia de minkowski en formato float
    return float((np.sum([(abs(a-b))**p]))**(1/p))

class knn_reg:
    def __init__(self):
        self.k = None
        self.p = None
        self.x_train = None
        self.y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int = 5, p: int = 2):
        """

        Ajusta (entrena) el modelo utilizando los datos de entrenamiento y sus etiquetas correspondientes.

        Esta función guarda internamente los datos de entrenamiento y valida los parámetros:
        - "X_train" y "y_train" deben tener el mismo número de observaciones (filas).
        - "k" debe ser un entero positivo que indica el número de vecinos a considerar.
        - "p" debe ser un entero positivo que indica el grado de la distancia de Minkowski a utilizar.

        Args:
            X_train (np.ndarray): Matriz de características de entrenamiento.
            y_train (np.ndarray): Vector de etiquetas o valores objetivo correspondientes.
            k (int, opcional): Número de vecinos a usar para predicción. Por defecto 5.
            p (int, opcional): Grado de la distancia de Minkowski. Por defecto 2 (distancia euclídea).

        Raises:
            ValueError: Si "k" o "p" no son enteros positivos.
            ValueError: Si "X_train" y "y_train" no tienen el mismo número de filas.

        Returns:
            None

        """
        # Comprobamos que los tipos de datos coinciden con las especificaciones necesarias
        if isinstance(k,int) and k>0 and isinstance(p,int) and p>0:
            self.k = k
            self.p = p
        else:
            # Raiseamos un error en caso de que no se cumpla el formato necesario
            raise ValueError("k and p must be positive integers.")

        if len(X_train) == len(y_train):
            self.x_train = X_train
            self.y_train = y_train
        else:
            # Raiseamos un error en caso de que no se cumpla el formato necesario
            raise ValueError("Length of X_train and y_train must be equal.")

    def predict(self, X:np.ndarray) -> np.ndarray:
        """

        Predice los valores objetivo para nuevas muestras utilizando el algoritmo KNN.

        Para cada muestra en "X", se calcula la distancia respecto a los datos de entrenamiento,
        se seleccionan los "k" vecinos más cercanos y se aplica una función estadística
        (como la media o moda) sobre sus etiquetas para generar la predicción.

        Args:
            X (np.ndarray): Muestras nuevas para las que se desea predecir la etiqueta o valor objetivo.

        Returns:
            np.ndarray: Vector de predicciones para cada muestra en "X".

        """
        # Lista de predicciones
        predictions = []

        # Por cada dato del dataframe
        for point in X:

            # Calculamos las distancias para saber cuáles son los vecinos cercanos a nuestro punto
            distances = self.compute_distances(point)

            # Obtenemos sus índices
            neighbors_indexes = self.get_k_nearest_neighbors(distances)

            # Sacamos los valores de T3 que tienen estos vecinos (labels)
            labels = self.y_train[neighbors_indexes]

            # Computamos la predicción de T3 para el dato nuevo con la media de los valores de T3 de sus vecinos cercanos
            prediccion = self.estadistica_entre_vecinos(knn_labels=labels)

            # Guardamos la predicción de T3
            predictions.append(prediccion)

        # Devolvemos el array de predicciones de T3 
        return np.array(predictions)

    def compute_distances(self, point: np.ndarray) -> np.ndarray:
        """
        Calcula la distancia desde un punto dado a todos los puntos del conjunto de entrenamiento.

        Utiliza la distancia de Minkowski (según el parámetro "p" definido en el modelo)
        para comparar la muestra "point" con cada observación en "self.x_train".

        Args:
            point (np.ndarray): Muestra individual para la que se desea calcular la distancia a cada punto del entrenamiento.

        Returns:
            np.ndarray: Array con las distancias desde "point" a cada punto del conjunto de entrenamiento.

        """
        # Computamos la distancia entre cada uno de los puntos de x_train (metido en un np.array) con respecto al np.array de point
        return np.array([minkowski_distance(point,x) for x in self.x_train])
    
    def get_k_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        """
        Obtiene los índices de los k vecinos más cercanos a un punto, dados sus valores de distancia.

        Ordena las distancias de menor a mayor y selecciona los índices de los "k" valores más pequeños.

        Args:
            distances (np.ndarray): Array de distancias entre un punto y cada muestra del conjunto de entrenamiento.

        Returns:
            np.ndarray: Índices de las "k" muestras más cercanas en el conjunto de entrenamiento.

        Nota:
            Utiliza la función "np.argsort()" para obtener los índices ordenados por distancia.

        """
        # Utilizamos el método np.argsort, que devuelve los índices de la matriz (por filas), que la ordenaría en orden ascendente
        indices = np.argsort(distances)

        # Al poner [:self.k] con numpy significa que se incluye el índice self.k
        return indices[:self.k]
    
    def estadistica_entre_vecinos(self, knn_labels:np.ndarray) -> float:
        """
        Obtenemos el estadístico (media, mediana, etc) entre los vecinos más cercanos 
        a nuestro datos de interés.

        Args:
            knn_labels (np.ndarray): Son las etiquetas (T3) de los vecinos más cercanos
        Returns:
            prediccion (float): resultado del cálculo que se ha realizado entre todos los vecinos cercanos

        """
        # Calculamos la media, aunque también podríamos usar la mediana si quisiéramos
        prediccion = np.mean(knn_labels)

        # Devolvemos la predicción
        return prediccion

    def __str__(self):
        """

        String representation of the kNN model.

        """
        return f"kNN model (k={self.k}, p={self.p})"

######################################################################################################################################################

# CROSS VALIDATION

def cross_validation(model, X, y, nFolds, k):
    """
    Realiza validación cruzada para evaluar el rendimiento de un modelo de aprendizaje automático.

    Esta función implementa manualmente una validación cruzada de tipo n-fold. Si "nFolds" es -1,
    se aplica validación Leave-One-Out (LOO), donde cada muestra se utiliza como conjunto de validación
    una vez, y el resto como entrenamiento.

    Args:
        model: Estimador tipo scikit-learn.
            Modelo que implementa los métodos .fit() y .predict(). Debe aceptar el argumento "k" en .fit().
        X (np.ndarray): Matriz de características de entrada, de forma (n_muestras, n_features).
        y (np.ndarray): Vector de etiquetas/valores objetivo, de forma (n_muestras,).
        nFolds (int): Número de particiones para la validación cruzada. Si es -1, se usa Leave-One-Out.
        k (int): Número de vecinos a usar durante el ajuste del modelo (solo si aplica).

    Returns:
        Tuple[float, float]: Media y desviación estándar del R² en las distintas particiones.
    """
    # Si se elige Leave-One-Out, cada muestra será un fold
    if nFolds == -1:
        nFolds = X.shape[0]

    # Calculamos el tamaño de cada fold
    fold_size = X.shape[0] // nFolds  # número de muestras entre número de folds

    # Lista para almacenar los scores de R² de cada partición
    r2_scores = []

    for i in range(nFolds):
        # Índices para el conjunto de validación
        valid_indices = np.array([j for j in range(fold_size*i, fold_size*(i+1))])

        # Índices para el conjunto de entrenamiento
        train_indices = np.array([j for j in range(X.shape[0]) if j not in valid_indices])

        # División de los datos
        X_train, X_valid = X[train_indices], X[valid_indices]
        y_train, y_valid = y[train_indices], y[valid_indices]

        # Entrenamiento del modelo
        model.fit(X_train, y_train, k=k)

        # Predicción sobre el conjunto de validación
        predicciones = model.predict(X_valid)

        # Cálculo del score (R²) y almacenamiento
        r2 = r2_score(y_true=y_valid, y_pred=predicciones)

        # Añadimos el R2 de esta iteración a la lista
        r2_scores.append(r2)

    # Calculamos la media y la desviación típica
    media_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)

    # Devolvemos le media y la std de R2
    return media_r2, std_r2

    """

    Realiza validación cruzada sobre un modelo de aprendizaje supervisado, como Random Forest.

    Esta función implementa manualmente la validación cruzada n-fold. Si "nFolds" es igual a -1,
    se realiza Leave-One-Out (LOO), utilizando una muestra como validación y el resto como entrenamiento
    en cada iteración.

    Args:
        model: Estimador tipo scikit-learn.
            El modelo a evaluar. Debe implementar los métodos ".fit()" y ".predict()".
        X (np.ndarray): Matriz de características, de tamaño (n_muestras, n_variables).
        y (np.ndarray): Vector de valores objetivo, de tamaño (n_muestras,).
        nFolds (int): Número de particiones a realizar. Si es -1, se aplica Leave-One-Out.

    Returns:
        Tuple[float, float]: Media y desviación estándar del R² en las distintas particiones.

    """
    # Si se ha indicado Leave-One-Out
    if nFolds == -1:
        nFolds = X.shape[0]

    # Tamaño de cada fold
    fold_size = X.shape[0] // nFolds  # número de muestras entre número de folds

    # Lista para almacenar los scores de R² en cada partición
    r2_scores = []

    for i in range(nFolds):
        # Índices para el conjunto de validación
        valid_indices = np.array([j for j in range(fold_size*i, fold_size*(i+1))])

        # Índices para el conjunto de entrenamiento (todo lo que no sea validación)
        train_indices = np.array([j for j in range(X.shape[0]) if j not in valid_indices])

        # División de datos en entrenamiento y validación
        X_train, X_valid = X[train_indices], X[valid_indices]
        y_train, y_valid = y[train_indices], y[valid_indices]

        # Entrenamiento del modelo
        model.fit(X_train, y_train)

        # Predicción sobre el conjunto de validación
        predicciones = model.predict(X_valid)

        # Evaluación de R²
        r2 = r2_score(y_true=y_valid, y_pred=predicciones)

        # Añadimos la accuracy de esta iteración a la lista
        r2_scores.append(r2)

    # Calculamos la media y la desviación típica
    media_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)

    # Devolvemos la media y la std de R2
    return media_r2, std_r2

######################################################################################################################################################

# REGRESIÓN LINEAL

class LinearRegressor:
    """

    Modelo de Regresión Lineal Extendido con soporte para variables categóricas y ajuste mediante descenso de gradiente. 
    
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """

        Ajusta el modelo utilizando mínimos cuadrados o descenso de gradiente.

        Esta función permite entrenar el modelo de regresión lineal usando dos métodos:
        - "least_squares": resuelve la ecuación normal para encontrar los coeficientes óptimos.
        - "gradient_descent": ajusta los coeficientes iterativamente minimizando el error con descenso de gradiente.

        Args:
            X (np.ndarray): Matriz de variables independientes (2D). Si es 1D, se transforma automáticamente.
            y (np.ndarray): Vector de la variable dependiente (1D).
            method (str): Método de entrenamiento. Debe ser "least_squares" o "gradient_descent".
            learning_rate (float): Tasa de aprendizaje para el descenso de gradiente.
            iterations (int): Número de iteraciones para el descenso de gradiente.

        Returns:
            None: Modifica internamente los coeficientes ("self.coefficients") y el intercepto ("self.intercept").

        Raises:
            ValueError: Si se indica un método no soportado.

        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Ajusta el modelo utilizando regresión lineal múltiple mediante la ecuación matricial.

        Este método aplica la solución cerrada de mínimos cuadrados:
        w = (Xᵀ·X)⁻¹·Xᵀ·y, utilizando la pseudo-inversa de NumPy ("np.linalg.pinv") para garantizar estabilidad
        numérica en caso de que la matriz no sea invertible.

        Se espera que "X" ya contenga una columna de unos al inicio (bias) para modelar el intercepto.

        Args:
            X (np.ndarray): Matriz de variables independientes (2D), incluyendo la columna de bias.
            y (np.ndarray): Vector de la variable dependiente (1D).

        Returns:
            None: Modifica los atributos del modelo "self.coefficients" y "self.intercept" in-place.
        """
        # X = np.c_[X,np.ones(X.shape[0])]  # añadimos una columna de 1s

        w = np.linalg.pinv(X.T @ X) @ (X.T @ y)  # la @ se usa para hacer el producto, X.T representa la traspuesta

        self.intercept = w[0]  # extraemos el valor del término independiente (el primero del vector w)
        self.coefficients = w[1:]  # extraemos los coeficientes de la regresión (son todos los términos de w menos el primero)

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Ajusta el modelo utilizando el método de descenso de gradiente.

        Este método entrena el modelo de regresión lineal de forma iterativa ajustando los coeficientes
        y el término independiente mediante la minimización de la función de pérdida (error cuadrático medio).

        Args:
            X (np.ndarray): Matriz de características (2D), incluyendo una columna inicial de unos para el intercepto (bias).
            y (np.ndarray): Vector de valores objetivo (1D).
            learning_rate (float): Tasa de aprendizaje que controla la magnitud de los pasos en cada iteración.
            iterations (int): Número de iteraciones a ejecutar para actualizar los parámetros del modelo.

        Returns:
            None: Modifica internamente "self.coefficients" y "self.intercept".
        """

        # Inicializamos parámetros con valores pequeños aleatorios
        m = len(y)
        self.coefficients = (
            np.random.rand(X.shape[1] - 1) * 0.01
        )  
        self.intercept = np.random.rand() * 0.01

        # Para graficar en el notebook:
        self.loss_vals = []  
        self.params = []  

        # Implementamos gradient descent 
        for epoch in range(iterations):
            predictions = self.predict(X=X[:, 1:])  # X es una matriz con la primera columna de todo 1s, por lo que pasamos al preidict la segunda columna solo
            error = predictions - y   # en la diapo pone y-predictions

            # Calcular loss y almacenarla
            mse = np.mean(error**2)
            self.loss_vals.append(mse)

            # Valores actuales de intercept y coeficients
            self.params.append((self.intercept, *self.coefficients))

            # Cálculo del gradiente
            gradient = (learning_rate/m) * X.T.dot(error)

            """
            Lo planteamos como un producto de vector fila, error, (1xN) por matriz, X, (Nx2)
            De esta manera obtenemos un vector de 1x2, donde cada columna es lo que en la diapositiva
            sale como thetha sub j, es decir que para j=0 tenemos la parte del gradiente del intercept
            y para j = 1 tenemos la parte del gradiente de los coeficientes.

            Esto es equivalente a lo que nos dice la fórmula que para actualizar el thetha j, hay que hacer
            el sumatorio desde i=1 hasta i=N, de el error[i] multiplicado por el x[i] de la columna j.
            Este sumatorio es equivalente a hacer el producto vectorial de la fila error por la columna j de la matriz X, 
            porque se multiplican elemento a elemento y luego se suma todo. Así tenemos una forma más compacta.
            
            """
            
            # Actualización de los parámetros
            self.intercept -= gradient[0]
            self.coefficients -= gradient[1:]
            

            # Cálculo y printeo de la loss
            if epoch % 10000 == 0:
                mse = np.power(evaluate_regression(y,predictions)["RMSE"],2)
                # print(f"predictions: {predictions}")
                # print(f"error: {error}")
                # print(f"Epoch {epoch}: MSE = {mse}")

    def predict(self, X):
        """
        Predice los valores de la variable dependiente usando el modelo previamente ajustado.

        Esta función aplica la fórmula del modelo lineal: Y = X·w + b, donde:
        - "X" es el array de entrada (puede ser unidimensional o bidimensional),
        - "w" son los coeficientes aprendidos ("self.coefficients"),
        - "b" es el término independiente ("self.intercept").

        Args:
            X (np.ndarray): Datos de entrada (variables independientes). Puede ser un array 1D o 2D.

        Returns:
            np.ndarray: Valores predichos de la variable dependiente.

        Raises:
            ValueError: Si el modelo no ha sido ajustado previamente (coeficientes o intercepto no definidos).

        """

        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            predictions = X*self.coefficients + self.intercept  # Y = X*w + b  (modo unidimensional)
        else:
            predictions = X.dot(self.coefficients) + self.intercept  # Y = X*w + b  (modo multidimensional)
        return predictions

def evaluate_regression(y_true, y_pred):
    """

    Evalúa el rendimiento de un modelo de regresión calculando R², RMSE, MSE y MAE.

    Esta función compara los valores predichos por un modelo con los valores reales,
    devolviendo métricas clásicas de regresión: coeficiente de determinación (R²),
    error cuadrático medio (MSE), su raíz cuadrada (RMSE), y el error absoluto medio (MAE).

    Args:
        y_true (np.ndarray): Valores reales de la variable objetivo.
        y_pred (np.ndarray): Valores predichos por el modelo de regresión.

    Returns:
        dict: Diccionario con los valores de R², MSE, RMSE y MAE.
              Las claves del diccionario son: "R2", "MSE", "RMSE", "MAE".

    """

    rss = np.sum((y_true-y_pred)**2)
    tss = np.sum((y_true-np.mean(y_true))**2)

    # R^2 Score
    r_squared = 1 - (rss/tss)

    # Root Mean Squared Error
    N = len(y_pred)
    rmse = np.sqrt( 1/N * np.sum(np.power(y_true-y_pred, 2)))

    # MSE
    mse = rmse**2

    # Mean Absolute Error
    mae = (1/N)*np.sum(abs(y_true-y_pred))

    print("Evaluación del modelo de regresión:")
    print(f"MAE  (Error Absoluto Medio): {mae:.3f}")
    print(f"MSE  (Error Cuadrático Medio): {mse:.3f}")
    print(f"RMSE (Raíz del MSE): {rmse:.3f}")
    print(f"R²   (Coef. de determinación): {r_squared:.3f}")

    return {"R2": r_squared, "MSE": mse, "RMSE": rmse, "MAE": mae}

######################################################################################################################################################

# ÁRBOLES, BOOSTING, RANDOM FOREST

def plot_rmse_vs_depth(x_train, y_train, x_val, y_val, max_depth_range=range(1, 20), figsize=(10, 6)):
    """

    Grafica el error RMSE en entrenamiento y validación en función de la profundidad máxima de un árbol de decisión.

    Esta función entrena varios modelos "DecisionTreeRegressor" variando la profundidad máxima ("max_depth") 
    y calcula el error cuadrático medio (RMSE) en el conjunto de entrenamiento y en el de validación. 
    Luego, genera una gráfica que permite visualizar el sobreajuste o infraajuste del modelo.

    Args:
        x_train (np.ndarray or pd.DataFrame): Conjunto de características para entrenamiento.
        y_train (np.ndarray or pd.Series): Valores objetivo del conjunto de entrenamiento.
        x_val (np.ndarray or pd.DataFrame): Conjunto de características para validación.
        y_val (np.ndarray or pd.Series): Valores objetivo del conjunto de validación.
        max_depth_range (range, opcional): Rango de profundidades máximas a evaluar. Por defecto, de 1 a 19.
        figsize (tuple, opcional): Tamaño de la figura del gráfico. Por defecto, (10, 6).

    Returns:
        None. Muestra un gráfico en pantalla con el RMSE en función de la profundidad.
    
    """
    
    train_rmse = []
    test_rmse = []
    tree_size = list(max_depth_range)

    # Entrenamos un árbol para cada profundidad y guardamos los RMSE
    for i in tree_size:
        model = DecisionTreeRegressor(max_depth=i, random_state=1)
        model.fit(x_train, y_train)

        predictions_train = model.predict(x_train)
        predictions_test = model.predict(x_val)

        rmse_train = np.sqrt(mean_squared_error(y_true=y_train, y_pred=predictions_train))
        rmse_test = np.sqrt(mean_squared_error(y_true=y_val, y_pred=predictions_test))

        train_rmse.append(rmse_train)
        test_rmse.append(rmse_test)

    # Gráfico de los errores
    plt.figure(figsize=figsize)
    plt.plot(tree_size, train_rmse, 'r*--', label='Train RMSE')
    plt.plot(tree_size, test_rmse, 'b.-', label='Validation RMSE')
    plt.xlabel('Max Depth del Árbol')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Error RMSE vs Profundidad del Árbol de Decisión')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def cross_validation_random_forest(X, max_feats_values, n_trees_values, max_samples_values, nFolds=5):
    """
    Realiza validación cruzada para un modelo Random Forest personalizado.

    Esta función implementa manualmente una versión simplificada de Random Forest,
    evaluando combinaciones de hiperparámetros mediante validación cruzada.
    Se prueban distintas configuraciones de:
    - Número de árboles ("n_trees")
    - Proporción de características ("max_feats")
    - Proporción de muestras por árbol ("max_samples")

    Args:
        X (pd.DataFrame): DataFrame que contiene las variables predictoras y la variable objetivo ('T3').
        max_feats_values (list[float]): Lista de proporciones de variables a usar por árbol (entre 0 y 1).
        n_trees_values (list[int]): Lista con los números de árboles a probar.
        max_samples_values (list[float]): Lista de proporciones de muestras por árbol (entre 0 y 1).
        nFolds (int, opcional): Número de particiones para validación cruzada. Por defecto, 5.

    Returns:
        dict: Diccionario con claves (n_trees, max_feats, max_samples) y valores (media_r2, std_r2),
              donde "media_r2" es el R² medio obtenido y "std_r2" su desviación estándar.
    
    
    """
    df_train = X.copy()
    resultados = {}
    target_col = "T3"

    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]

    for n_trees in n_trees_values:
        for max_feats in max_feats_values:
            for max_samples in max_samples_values:

                r2_scores = []
                fold_size = len(df_train) // nFolds

                for i in range(nFolds):

                    # Índices de validación y entrenamiento
                    valid_start = i * fold_size
                    valid_end = (i + 1) * fold_size if i < nFolds - 1 else len(df_train)

                    valid_indices = list(range(valid_start, valid_end))
                    train_indices = list(set(range(len(df_train))) - set(valid_indices))

                    X_train, X_val = X.iloc[train_indices], X.iloc[valid_indices]
                    Y_train, Y_val = y.iloc[train_indices], y.iloc[valid_indices]

                    predictions = []

                    for _ in range(n_trees):

                        # Bootstrap
                        sample_indices = np.random.choice(X_train.index, size=int(len(X_train) * max_samples), replace=True)
                        X_bootstrap = X_train.loc[sample_indices]
                        y_bootstrap = Y_train.loc[sample_indices]

                        # Subconjunto de features
                        n_feats = int(X_train.shape[1] * max_feats)
                        feature_subset = np.random.choice(X_train.columns, size=n_feats, replace=False)
                        X_bootstrap_subset = X_bootstrap[feature_subset]
                        X_val_subset = X_val[feature_subset]

                        # Entrenamiento y predicción
                        tree = DecisionTreeRegressor()
                        tree.fit(X_bootstrap_subset, y_bootstrap)
                        y_pred = tree.predict(X_val_subset)
                        predictions.append(y_pred)

                    # Voto por promedio
                    y_pred_final = np.mean(predictions, axis=0)
                    r2 = r2_score(Y_val, y_pred_final)
                    r2_scores.append(r2)

                media_r2 = np.mean(r2_scores)
                std_r2 = np.std(r2_scores)
                resultados[(n_trees, max_feats, max_samples)] = (media_r2, std_r2)

    return resultados

def cross_validation_boosting(X, n_estimators_values,  max_depth_values, subsample_values, nFolds = 5):
    """
    Realiza validación cruzada para un modelo de Gradient Boosting (regresión).

    Esta función evalúa distintas combinaciones de hiperparámetros para un modelo 
    "GradientBoostingRegressor" de "scikit-learn", utilizando validación cruzada n-fold. 
    Devuelve el rendimiento medio y la desviación estándar del R² para cada configuración.

    Args:
        X (pd.DataFrame): DataFrame que incluye tanto las variables predictoras como la variable objetivo ("T3").
        n_estimators_values (list[int]): Lista con los números de árboles (estimadores) a evaluar.
        max_depth_values (list[int]): Lista con profundidades máximas del árbol base (weak learner).
        subsample_values (list[float]): Lista con proporciones del conjunto de datos a muestrear para cada estimador (entre 0 y 1).
        nFolds (int, opcional): Número de particiones (folds) para la validación cruzada. Por defecto es 5.

    Returns:
        dict: Diccionario con claves (n_estimators, max_depth, subsample) y valores (media_r2, std_r2), 
              donde "media_r2" es el R² medio y "std_r2" su desviación estándar.
    
    """
    df_train = X.copy()
    resultados = {}
    target_col = "T3"

    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]

    for n_estimators in n_estimators_values:
        for max_depth in max_depth_values:
            for subsample in subsample_values:

                r2_scores = []
                fold_size = len(df_train) // nFolds

                for i in range(nFolds):

                    # Índices de validación y entrenamiento
                    valid_start = i * fold_size
                    valid_end = (i + 1) * fold_size if i < nFolds - 1 else len(df_train)

                    valid_indices = list(range(valid_start, valid_end))
                    train_indices = list(set(range(len(df_train))) - set(valid_indices))

                    X_train, X_val = X.iloc[train_indices], X.iloc[valid_indices]
                    y_train, y_val = y.iloc[train_indices], y.iloc[valid_indices]

                    # Entrenamiento del modelo (una sola vez)
                    gbr = GradientBoostingRegressor(
                        n_estimators=n_estimators,
                        learning_rate=0.1,
                        max_depth=max_depth,
                        subsample=subsample,
                        random_state=48
                    )
                    gbr.fit(X_train, y_train)

                    # Predicción
                    y_pred = gbr.predict(X_val)
                    r2 = r2_score(y_val, y_pred)
                    r2_scores.append(r2)

                # Guardamos los resultados de esta combinación de hiperparámetros
                media_r2 = np.mean(r2_scores)
                std_r2 = np.std(r2_scores)
                resultados[(n_estimators, max_depth, subsample)] = (media_r2, std_r2)

    return resultados

def plot_resultados_modelo(resultados, fila, columna, valor='mean_r2', fmt='.3f', cmap='YlGnBu', nombres_params=None):
    """
    Visualiza un heatmap de los resultados del modelo en función de los hiperparámetros.

    Args:
        resultados (dict): Claves: tuplas de hiperparámetros, Valores: (media, std)
        fila (str): Hiperparámetro que se usará como fila del heatmap.
        columna (list[str]): Lista de hiperparámetros que se combinarán como columna.
        valor (str): Métrica a visualizar.
        fmt (str): Formato de los números en el heatmap.
        cmap (str): Mapa de color para el heatmap.
        nombres_params (list[str], opcional): Nombres reales de los hiperparámetros, en el mismo orden que las tuplas.

    Returns:
        None
    
    """

    # Detectamos el número de hiperparámetros
    n_param = len(next(iter(resultados)))
    if nombres_params is None:
        nombres_params = [f"param_{i}" for i in range(n_param)]

    # Construimos el DataFrame
    df_resultados = pd.DataFrame([
        {**{nombres_params[i]: k[i] for i in range(n_param)},
         'mean_r2': v[0], 'std_r2': v[1]}
        for k, v in resultados.items()
    ])

    # Comprobamos que las columnas existen
    for col in [fila] + columna:
        if col not in df_resultados.columns:
            raise KeyError(f"La columna '{col}' no existe. Asegúrate de que los nombres en 'nombres_params' coincidan.")

    # Creamos una columna combinada
    if len(columna) > 1:
        df_resultados["config"] = df_resultados.apply(
            lambda row: "-".join([f"{row[c]:.2f}" if isinstance(row[c], float) else str(row[c]) for c in columna]), axis=1
        )
    else:
        df_resultados["config"] = df_resultados[columna[0]]

    # Creamos la tabla para el heatmap
    heatmap_data = df_resultados.pivot(index=fila, columns="config", values=valor)

    # Ploteamos
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=fmt, cmap=cmap, linewidths=0.5)
    plt.title(f"{valor} por configuración de hiperparámetros")
    plt.xlabel(" - ".join(columna))
    plt.ylabel(fila)
    plt.tight_layout()
    plt.show()

def plot_metric_evolution(resultados, eje_x, hue=None, style=None, metric='mean_r2', std_metric='std_r2', nombres_params=None):
    """

    Grafica la evolución de una métrica en función de un hiperparámetro, para cualquier modelo.

    Args:
        resultados (dict): Claves = tuplas de hiperparámetros, Valores = (métrica_media, std)
        eje_x (str): Nombre del hiperparámetro para el eje X.
        hue (str, opcional): Hiperparámetro para diferenciar colores.
        style (str, opcional): Hiperparámetro para diferenciar estilos de línea.
        metric (str): Métrica a visualizar (por defecto 'mean_r2').
        std_metric (str): Métrica de desviación estándar (por defecto 'std_r2').
        nombres_params (list[str], opcional): Nombres reales de los hiperparámetros, en el orden de las tuplas.
    
    Returns:
        None

    """

    # Detectamos el número de hiperparámetros
    n_param = len(next(iter(resultados)))
    if nombres_params is None:
        nombres_params = [f"param_{i}" for i in range(n_param)]

    # Construimos el DataFrame desde el diccionario
    df = pd.DataFrame([
        {**{nombres_params[i]: k[i] for i in range(n_param)},
         metric: v[0], std_metric: v[1]}
        for k, v in resultados.items()
    ])

    # Comprobamos las columnas
    for col in [eje_x, hue, style]:
        if col and col not in df.columns:
            raise KeyError(f"La columna '{col}' no existe. Verifica los nombres de los parámetros.")

    # Preparamos paleta si hay hue
    if hue and df[hue].nunique() <= 10:
        palette = sns.color_palette("colorblind", df[hue].nunique())
        palette_dict = dict(zip(sorted(df[hue].unique()), palette))
    else:
        palette_dict = None

    # Ordenamos por eje_x para una gráfica limpia
    df_sorted = df.sort_values(by=[eje_x] + ([hue] if hue else []) + ([style] if style else []))

    # Ploteamos
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    sns.lineplot( data=df_sorted, x=eje_x, y=metric, hue=hue, style=style, marker='o', errorbar=('sd'), palette=palette_dict)

    plt.title(f'{metric} vs {eje_x}')
    plt.xlabel(eje_x)
    plt.ylabel(metric)
    plt.grid(True)

    if hue or style:
        leyenda = " / ".join(filter(None, [hue, style]))
        plt.legend(title=leyenda)

    plt.tight_layout()
    plt.show()

######################################################################################################################################################

# REGRESIÓN LOGÍSTICA

class LogisticRegressor:
    def __init__(self):
        """
        Inicializa una instancia del modelo de Regresión Logística.

        Atributos:
            weights (np.ndarray): Vector de pesos del modelo (coeficientes). Se inicializa como None 
                                y se establecerá durante el entrenamiento.
            bias (float): Término independiente (bias) del modelo. Se inicializa como None y se ajustará 
                        en la fase de entrenamiento.

        """
        self.weights = None
        self.bias = None

    def fit(self, X, y, learning_rate=0.01, num_iterations=1000, penalty=None, l1_ratio=0.5, C=1.0, verbose=False, print_every=100):
        """

        Ajusta el modelo de regresión logística a los datos usando descenso de gradiente.

        Este método inicializa los pesos y el sesgo, y luego actualiza iterativamente estos parámetros
        siguiendo el gradiente negativo de la función de pérdida (log-likelihood).

        Soporta distintos tipos de regularización:
        - Sin regularización: el modelo se entrena únicamente minimizando la pérdida.
        - L1 (Lasso): penaliza los valores absolutos de los coeficientes para fomentar la esparsidad.
        - L2 (Ridge): penaliza los cuadrados de los coeficientes, reduciendo su magnitud.
        - ElasticNet: combina L1 y L2, ponderados por "l1_ratio".

        Args:
            X (np.ndarray): Matriz de características (muestras x variables), de tamaño (m, n).
            y (np.ndarray): Vector de etiquetas verdaderas (binarias), de tamaño (m,).
            learning_rate (float): Tasa de aprendizaje utilizada para actualizar los parámetros.
            num_iterations (int): Número de iteraciones a ejecutar del algoritmo de optimización.
            penalty (str): Tipo de regularización a aplicar. Puede ser 'lasso', 'ridge', 'elasticnet' o None.
            l1_ratio (float): Parámetro de mezcla para ElasticNet. 0 equivale a solo L2, 1 equivale a solo L1.
            C (float): Inverso de la fuerza de regularización. Cuanto menor sea, más fuerte es la penalización.
            verbose (bool): Si es True, muestra la pérdida cada cierto número de iteraciones.
            print_every (int): Frecuencia con la que se imprime la pérdida si "verbose=True".

        Actualiza:
            self.weights (np.ndarray): Pesos ajustados del modelo.
            self.bias (float): Término independiente ajustado tras el entrenamiento.

        """
        # Obtener m (número de datos) y n (número de features)
        m = X.shape[0]
        n = X.shape[1]

        # Inicilizar todos los parámetros a cero
        self.weights = np.zeros((n))
        self.bias = 0

        # Descenso de gradiente
        for i in range(num_iterations): 

            # Predecimos la probabilidad
            y_hat = self.predict_proba(X)

            # Calculamos la pérdida
            loss = self.log_likelihood(y, y_hat)

            # Mostramos la pérdida
            if i % print_every == 0 and verbose:
                print(f"Iteration {i}: Loss {loss}")

            # Calculamos los gradientes
            dw = (1 / m) * X.T.dot(y_hat - y)
            db = (1 / m) * np.sum(y_hat - y)

            # Añadimos la parte de actualización correspondiente a la regularización
            if penalty == "lasso":
                dw = self.lasso_regularization(dw, m, C)
            elif penalty == "ridge":
                dw = self.ridge_regularization(dw, m, C)
            elif penalty == "elasticnet":
                dw = self.elasticnet_regularization(dw, m, C, l1_ratio)

            # Actualizamos los parámetros
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict_proba(self, X):
        """

        Calcula las probabilidades predichas de pertenecer a la clase positiva (1) para cada muestra.

        Este método utiliza la función sigmoide para transformar los valores lineales ("z = X·w + b") 
        en probabilidades entre 0 y 1. Es el paso de "activación" en la regresión logística.

        Args:
            X (np.ndarray): Matriz de características de entrada de forma (m, n), donde m es el número de muestras 
                            y n el número de características.

        Returns:
            np.ndarray: Array de tamaño (m, 1) con las probabilidades de pertenencia a la clase positiva para cada muestra.

        """

        # Calculamos el valor de z
        z = X.dot(self.weights) + self.bias

        # Aplicamos la transformación de la sigmoide
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """

        Predice las etiquetas de clase (0 o 1) para cada muestra en "X" a partir de sus probabilidades predichas.

        Este método utiliza un umbral ("threshold") para convertir las probabilidades devueltas por "predict_proba"
        en etiquetas binarias. Si la probabilidad es mayor que el umbral, se predice clase 1; de lo contrario, clase 0.

        Args:
            X (np.ndarray): Matriz de características de entrada, de forma (m, n), donde m es el número de muestras
                            y n el número de variables independientes.
            threshold (float): Umbral para decidir la clase positiva. Por defecto es 0.5.

        Returns:
            np.ndarray: Array unidimensional de tamaño (m,) con las predicciones de clase (0 o 1) para cada muestra.

        """
    
        # Predecimos la clase usando un threshold de separación, que se basa en la probabilidad de ser de la clase positiva
        probabilities = self.predict_proba(X)
        classification_result = np.array([1 if prob > threshold else 0 for prob in probabilities])

        return classification_result

    def lasso_regularization(self, dw, m, C):
        """

        Aplica regularización L1 (Lasso) al gradiente durante la actualización de pesos
        en el descenso de gradiente.

        La regularización L1 penaliza la magnitud absoluta de los coeficientes, lo que promueve
        la esparsidad en el modelo (algunos pesos pueden quedar en cero), funcionando de forma implícita 
        como una técnica de selección de características.

        El término de regularización se añade directamente al gradiente de la función de pérdida,
        y es proporcional al signo de cada peso, escalado por el inverso del número de muestras ("m")
        y la intensidad de regularización ("C").

        Args:
            dw (np.ndarray): Gradiente de la función de pérdida respecto a los pesos (sin regularización).
            m (int): Número de muestras del conjunto de datos.
            C (float): Inverso de la fuerza de regularización. Valores más pequeños implican mayor penalización.

        Returns:
            np.ndarray: Gradiente ajustado tras aplicar la regularización L1.

        """

        # Añadimos la contribución de Lasso a la función objetivo
        lasso_gradient = (C/m)*np.sign(self.weights)
        return dw + lasso_gradient

    def ridge_regularization(self, dw, m, C):
        """
        Aplica regularización L2 (Ridge) al gradiente durante el paso de actualización de pesos
        en el descenso de gradiente.

        La regularización L2 penaliza la magnitud de los coeficientes elevando al cuadrado sus valores,
        lo que ayuda a evitar el sobreajuste reduciendo el impacto de pesos excesivamente grandes.
        Promueve soluciones más estables y distribuidas.

        El término de regularización se añade al gradiente original como una proporción directa
        de los pesos actuales, escalada por la fuerza de regularización "C" e inversamente proporcional
        al número de muestras "m".

        Args:
            dw (np.ndarray): Gradiente de la función de pérdida respecto a los pesos, antes de aplicar regularización.
            m (int): Número de muestras del conjunto de entrenamiento.
            C (float): Inverso de la fuerza de regularización. Valores más pequeños implican mayor penalización.

        Returns:
            np.ndarray: Gradiente ajustado tras aplicar regularización L2.
        """

        # Añadimos la contribución de Ridge a la función objetivo
        ridge_gradient = (C/m)*self.weights

        return dw + ridge_gradient

    def elasticnet_regularization(self, dw, m, C, l1_ratio):
        """

        Aplica regularización Elastic Net al gradiente durante la actualización de pesos en el descenso de gradiente.

        Elastic Net combina regularización L1 (Lasso) y L2 (Ridge), lo que proporciona un equilibrio entre:
        - la inducción de esparsidad (L1), y
        - la reducción de magnitudes de los pesos (L2).

        Esta técnica ayuda a prevenir el sobreajuste y mejora la generalización del modelo, 
        especialmente en contextos con muchas variables correlacionadas o irrelevantes.

        La contribución regularizadora se calcula como:
            λ * [ l1_ratio * ∇L1 + (1 - l1_ratio) * ∇L2 ]
        donde:
            - ∇L1 es el gradiente de la penalización L1 (signo de los pesos),
            - ∇L2 es el gradiente de la penalización L2 (los propios pesos).

        Args:
            dw (np.ndarray): Gradiente actual de la función de pérdida respecto a los pesos (sin regularización).
            m (int): Número de muestras del conjunto de entrenamiento.
            C (float): Inverso de la fuerza de regularización. Cuanto menor, mayor es la penalización.
            l1_ratio (float): Proporción de mezcla entre L1 y L2. 0 equivale a Ridge (solo L2), 1 a Lasso (solo L1).

        Returns:
            np.ndarray: Gradiente ajustado tras aplicar la regularización Elastic Net.

        """
        # Combinamos la contribución de Lasso y de Ridge
        lasso_gradient = (C/m)*np.sign(self.weights).T
        ridge_gradient = (C/m)*self.weights.T

        elasticnet_gradient = l1_ratio*lasso_gradient + (1-l1_ratio)*ridge_gradient

        return dw + elasticnet_gradient

    @staticmethod
    def log_likelihood(y, y_hat):
        """

        Calcula la función de pérdida Log-Likelihood para regresión logística, equivalente
        a la entropía cruzada entre las etiquetas reales y las probabilidades predichas.

        Esta función mide qué tan bien el modelo predice las clases reales. Se basa en la siguiente fórmula:

            L(y, y_hat) = -(1/m) * Σ [ y * log(y_hat) + (1 - y) * log(1 - y_hat) ]

        donde:
            - m es el número total de observaciones,
            - y es el vector de etiquetas reales (0 o 1),
            - y_hat son las probabilidades predichas de pertenecer a la clase positiva (1),
            - log es el logaritmo natural.

        Args:
            y (np.ndarray): Etiquetas reales. Vector unidimensional con valores binarios (0 o 1).
            y_hat (np.ndarray): Probabilidades predichas de la clase positiva. Vector de valores entre 0 y 1.

        Returns:
            float: Valor escalar de la pérdida (log-likelihood).

        """

        # TODO: Implement the loss function (log-likelihood)
        m = y.shape[0]  # Number of examples
        loss = -(1/m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss

    @staticmethod
    def sigmoid(z):
        """
        Calcula la función sigmoide de "z", que puede ser un escalar o un array de NumPy.

        La función sigmoide se utiliza como función de activación en regresión logística. 
        Convierte cualquier valor real en un rango entre 0 y 1, lo que permite interpretarlo 
        como una probabilidad. Su fórmula es: 
            sigmoid(z) = 1 / (1 + exp(-z))

        Args:
            z (float o np.ndarray): Valor o array de entrada sobre el que aplicar la función sigmoide.

        Returns:
            float o np.ndarray: Resultado de aplicar la función sigmoide a "z", con los mismos tipos/formas que la entrada.
        """

        # Convertimos los logits en probabilidades
        sigmoid_value = 1/(1+np.exp(-z))

        # Devolvemos el valor calculado
        return sigmoid_value
    
######################################################################################################################################################





###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################

###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################

###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################

###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################

###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################

###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################
###################################




