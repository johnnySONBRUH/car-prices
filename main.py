import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('data.csv', index_col=False)

# Separate features and target
x = data.drop(columns=['selling_price', 'id'])
y = data['selling_price']

# Use separate LabelEncoders for each categorical column
encoders = {}
categorical_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'Mileage Unit']

for col in categorical_cols:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])
    encoders[col] = le

# Train model
model = RandomForestRegressor(
    n_jobs=-1,
    random_state=42,
    n_estimators=100, # Keeping it reasonable for training
)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(x_train, y_train)

# Evaluate
pred = model.predict(x_test)
r2 = r2_score(y_test, pred)
print(f'R2: {r2}')

# Save model and ALL encoders
joblib.dump({'model': model, 'encoders': encoders}, 'predictor.pkl', compress=3)
