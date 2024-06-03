import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

# ---------------------------- PROCESAMIENTO DE DATOS --------------------

# Cargar el dataset
df = pd.read_csv('EmployeesData.csv')

# Verificar valores faltantes
print("\n -------------------------- Valores faltantes: -----------------------------")
 # Verificar valores faltantes
valores_faltantes = df.isnull().sum().sum()
if valores_faltantes > 0:  
    print(f"Valores faltantes total: {valores_faltantes}")
    print(df.isnull().sum())
        # Eliminar filas con valores faltantes (si es necesario)
        # df = df.dropna()
else:
    print("No se encontraron valores faltantes")

# Convertir 'LeaveOrNot' de binario a categórico
print("\n---------------------------------------------------------")
df['LeaveOrNot'] = df['LeaveOrNot'].map({1: 'Leave', 0: 'Not Leave'})
print(df['LeaveOrNot'])

# Eliminar filas con valores faltantes en 'ExperienceInCurrentDomain' y 'JoiningYear'
df = df.dropna(subset=['ExperienceInCurrentDomain', 'JoiningYear'])

# Imputar datos faltantes en 'Age' con la media
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Imputar datos faltantes en 'PaymentTier' con la moda
df['PaymentTier'] = df['PaymentTier'].fillna(df['PaymentTier'].mode()[0])

# Eliminar valores atípicos basados en IQR
for column in ['Age', 'ExperienceInCurrentDomain']:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# ---------------------- ANALISIS EXPLORATORIO DE DATOS (EDA) ----------------------------

# Distribución de sexos
df['Gender'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Distribución de Sexos')
plt.ylabel('')
plt.show()

# Distribución de niveles de estudio
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
df['Education'].value_counts().plot(kind='bar', ax=ax[0])

# Rotacion para que se puedan ver las etiquetas
for tick in ax[0].get_xticklabels():
    tick.set_rotation(0)
df['Education'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[1])
ax[0].set_title('Histograma de Educación')
ax[1].set_title('Gráfico de Torta de Educación')
plt.show()

# Propensión de jóvenes a tomar licencias
sns.histplot(data=df, x='Age', hue='LeaveOrNot', multiple='stack')
plt.title('Propensión de Jóvenes a Tomar Licencias')
plt.show()

# Distribución de clases
df['LeaveOrNot'].value_counts().plot(kind='bar')
# Rotar las etiquetas del eje x
plt.xticks(rotation=0)
plt.title('Distribución de Clases')
plt.show()

# Balance
balance = df['LeaveOrNot'].value_counts(normalize=True)
print("\n --------------------------- Balance: --------------------------------------")
print(balance)

# ----------------------------- MODELADO DE DATOS------------------------------
# Preparar los datos
X = df.drop('LeaveOrNot', axis=1)
y = df['LeaveOrNot']

# Convertir variables categóricas a dummies
X = pd.get_dummies(X, drop_first=True)

# Partición estratificada del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Entrenar RandomForest sin cambios
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Entrenar RandomForest con class_weight="balanced"
rf_balanced = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_balanced.fit(X_train, y_train)

# Evaluar modelos
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)
y_pred_balanced_train = rf_balanced.predict(X_train)
y_pred_balanced_test = rf_balanced.predict(X_test)

print("\n ---------------------- Random Forest sin cambios --------------------------")
print("Accuracy en entrenamiento:", accuracy_score(y_train, y_pred_train))
print("Accuracy en test:", accuracy_score(y_test, y_pred_test))
print("F1 Score:", f1_score(y_test, y_pred_test, pos_label='Leave'))

print("\n -------------- Random Forest con class_weight='balanced' ------------------")
print("Accuracy en entrenamiento:", accuracy_score(y_train, y_pred_balanced_train))
print("Accuracy en test:", accuracy_score(y_test, y_pred_balanced_test))
print("F1 Score:", f1_score(y_test, y_pred_balanced_test, pos_label='Leave'))

# Matrices de confusión
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
plt.title('RandomForest sin cambios')
plt.show()

ConfusionMatrixDisplay.from_estimator(rf_balanced, X_test, y_test)
plt.title('Matriz de Confusión - RandomForest con class_weight="balanced"')
plt.show()

