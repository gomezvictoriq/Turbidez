#Importo librerías

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#Cargo los datos de interés y divido en entrenamiento y validación 

X, y = load_diabetes(return_X_y=True)
X = X[:, [2]]  # Use only one feature
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=20, shuffle=False
)

#Creo un modelo de regresión y ajusto utilizando los datos de entrenamiento.

regressor = LinearRegression().fit(X_train, y_train)

#Evalúo el modelo generado a partir de las métricas de desempeño.

y_pred = regressor.predict(X_test)
p_rmse = mean_squared_error(y_test, y_pred)
p_r2 = r2_score(y_test, y_pred)


print("RMSE", p_rmse)
print("R2", p_r2)

#Visualizo los resultados comparando el conjunto de entrenamiento y validación.


fig, ax = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)

ax[0].plot(
    X_train,
    regressor.predict(X_train),
    linewidth=3,
    color="#17A77E",
    label="Modelo",
)
ax[0].scatter(X_train, y_train, label="Entrenamiento", color = "#9D50A6", alpha = .6)
ax[0].set(xlabel="Característica", ylabel="Objetivo", title="Conjunto de entrenamiento")
ax[0].legend()

ax[1].plot(X_test, y_pred, linewidth=3, color="#17A77E", label="Modelo")
ax[1].scatter(X_test, y_test, label="Validación", color = "#9D50A6", alpha = .6)
ax[1].set(xlabel="Característica", ylabel="Objetivo", title="Conjunto de validación")
ax[1].legend()

fig.suptitle("Regresión lineal")

plt.show()



