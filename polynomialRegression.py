import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
df = pd.read_excel('dataset.xlsx')
y = df.Cases.values.reshape(-1,1)
X = df.Date.values.reshape(-1,1)
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)
tahminal = int(input("Enter the day:"))
tahmin  =int(pol_reg.predict(poly_reg.fit_transform([[tahminal]])))
plt.scatter(X, y, color='red')
plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Predection: {0}'.format(tahmin))
plt.xlabel('Date')
plt.ylabel('Cases')
plt.show()