import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
model = RandomForestRegressor(
    n_jobs=-1,
    random_state=42,
    n_estimators=2500,
    max_depth=16,
    min_samples_split=8,
)
data = pd.read_csv('data.csv',index_col=False)
le = LabelEncoder()
x,y = data, data['selling_price']
x = x.drop(columns=['selling_price','id'])
x['name'] = le.fit_transform(x['name'])
x['fuel'] = le.fit_transform(x['fuel'])
x['seller_type'] = le.fit_transform(x['seller_type'])
x['transmission'] = le.fit_transform(x['transmission'])
x['owner'] = le.fit_transform(x['owner'])
x['Mileage Unit'] = le.fit_transform(x['Mileage Unit'])
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
pred = model.predict(x_test)
r2 = r2_score(y_test,pred)
mae = mean_absolute_error(y_test,pred)
mape = mean_absolute_percentage_error(y_test,pred)
mse = mean_squared_error(y_test,pred)
print(f'R2: {r2}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}')
print(f'MSE: {mse}')
joblib.dump({'model':model,'enc':le},'predictor.pkl',compress=3)