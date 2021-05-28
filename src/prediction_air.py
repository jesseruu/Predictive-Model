import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

# Importamos el dataset limpio
ica_quality = pd.read_csv('train/minambiente_train_data.csv')

# Definimos las variables de X, y dependiendo de la particula 
X = ica_quality['FECHA'].values
op = input("Seleccione una opcion: ").upper()
y = ica_quality[op].values

X = np.array(X).reshape(-1,1)

# Dividimos el dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Se entrena el modelo
clf = RandomForestRegressor(n_estimators=200)
clf.fit(X_train, y_train)

# Una vez entrenado, se establece el modelo predictivo
y_pred = clf.predict(X_test)

# Rendimiento del modelo
print('Error Medio Absoluto:', metrics.mean_absolute_error(y_test, y_pred))
print('Error medio cuadratico:', metrics.mean_squared_error(y_test, y_pred))
print('Raiz error medio cuadrático:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Precision
print('Precision:', clf.score(X_train, y_train))