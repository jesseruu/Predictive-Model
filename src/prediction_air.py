import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

ica_quality = pd.read_csv('train/minambiente_train_data.csv')

X = ica_quality['FECHA'].values
op = input("Seleccione una opcion: ").upper()
y = ica_quality[op].values

X = np.array(X).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestRegressor(n_estimators=200)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Rendimiento del modelo
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('Precision:', clf.score(X_train, y_train))

'''plt.scatter(X_test['FECHA:MONTH'], y_test, color = "blue")
plt.scatter(X_test['FECHA:MONTH'], y_pred, color = "green")
plt.title("Modelo de prediccion")
plt.xlabel("Fechas")
plt.ylabel(op)
plt.xticks(rotation = 45)
plt.show()'''