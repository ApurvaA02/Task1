import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

dataFrame = pd.read_csv(r"C:\Users\APURVA\PycharmProjects\MLInternsip\train.csv")
dataFrame.head()
print(dataFrame['SalePrice'].describe())

x = dataFrame[['GrLivArea','BsmtFullBath','BsmtHalfBath','BedroomAbvGr','FullBath','HalfBath']]
y = dataFrame['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('coefficient of determination: %.2f' % r2_score(y_test, y_pred))
