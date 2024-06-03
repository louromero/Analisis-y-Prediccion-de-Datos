# ANÁLISIS Y PREDICCIÓN DE DATOS DE EMPLEADOS

## Objetivo
El objetivo de este proyecto es demostrar la comprensión y aplicación práctica de conceptos de análisis de datos a través de un proyecto específico que involucra el manejo, análisis y modelado de un dataset de empleados.

## Descripción del Proyecto
El proyecto consiste en desarrollar un análisis y modelado sobre un dataset de empleados de una empresa. Este dataset incluye información sobre educación, año de incorporación, ciudad de trabajo, categoría salarial, edad, género, si han sido asignados temporalmente a la banca (EverBenched), experiencia en el dominio actual y si el empleado tomó tiempo libre (LeaveOrNot). El dataset ha sido modificado para incluir datos faltantes, añadiendo realismo al desafío analítico.

## Requisitos
Las siguientes bibliotecas de Python son necesarias para ejecutar el proyecto:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Puedes instalarlas utilizando pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Instrucciones para Ejecutar
1. Clona este repositorio de la siguiente manera:

```bash
git clone https://github.com/louromero/Analisis-y-Prediccion-de-Datos.git
```

2. Asegúrate de que las dependencias necesarias están instaladas.
3. Ejecuta el script de Python `solucion.py` desde la terminal.

```bash
python solucion.py
```

## Descripción del Código
El proyecto se divide en varias etapas:

### Preprocesamiento de Datos
- **Cargar el dataset** desde un archivo CSV.
- **Verificar valores faltantes** y realizar imputaciones:
  - Convertir `LeaveOrNot` de valores binarios a etiquetas categóricas.
  - Eliminar filas con valores faltantes en `ExperienceInCurrentDomain` y `JoiningYear`.
  - Imputar datos faltantes en `Age` con la media.
  - Imputar datos faltantes en `PaymentTier` con la moda.
  - Eliminar registros con valores atípicos basándose en el análisis de IQR.

### Análisis Exploratorio de Datos (EDA)
- **Distribución de géneros** con un gráfico de torta.
- **Distribución de niveles de estudio** con un histograma y un gráfico de torta.
- **Propensión de los jóvenes a tomar licencias** con un histograma.
- **Distribución de clases** para verificar si el dataset está balanceado.

### Modelado de Datos
- **Preparar los datos para el modelado**:
  - Separar la columna objetivo.
  - Convertir variables categóricas a variables dummy.
  - Realizar una partición estratificada del dataset.
- **Entrenar dos modelos RandomForest**:
  - Uno sin cambios.
  - Otro usando `class_weight="balanced"`.
- **Calcular métricas de desempeño**:
  - Accuracy en los conjuntos de entrenamiento y prueba.
  - Matriz de confusión.
  - F1 Score.

### Comparación y Optimización
- Comparar el rendimiento de los modelos a nivel de Accuracy y F1 score.
- Explicar las diferencias utilizando las matrices de confusión.

## Resumen del Análisis
El análisis incluye una revisión del dataset, imputación de valores faltantes y manejo de datos atípicos. Los modelos RandomForest se entrenan y se comparan en términos de precisión y F1 score, con ajustes adicionales para optimizar el rendimiento.
