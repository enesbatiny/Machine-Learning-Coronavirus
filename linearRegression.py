import pandas as pd
import datetime
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df = pd.read_excel('dataset.xlsx')
cases = df.Cases.values.reshape(-1,1)
date = df.Date.values.reshape(-1,1)
regression = LinearRegression()
regression.fit(date,cases)
result = int(input("Enter the day:"))
print()
plt.scatter(df.Date,df.Cases)
x = np.arange(min(df.Date),max(df.Date)).reshape(-1,1)
plt.plot(x,regression.predict(x),color="red")
plt.xlabel("Date")
plt.ylabel("Cases")
plt.title("Predection: {0} Success Rate: {1}".format(int(regression.predict([[result]])),r2_score(cases,regression.predict(date))))
plt.show() 