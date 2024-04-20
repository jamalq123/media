import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt



# Load the data from Excel file
data = pd.read_excel('media_adspends.xlsx')
# https://www.youtube.com/watch?v=rsyrZnZ8J2o OneHotEncoder

# Preprocessing
X = data[['Rating', 'Channels']]
y = data['Ad_rev']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), ['Channels'])
    ],
    remainder='passthrough'
)

# Define the model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
# Calculate R^2 score
r2 = r2_score(y_test, y_pred)

# Streamlit app
st.title('Ad Revenue Prediction')

# Display accuracy (R^2 score)
st.write('R^2 Score:', r2)

# Round y_pred and y_test to one decimal point
y_pred_rounded = y_pred.round(1)
y_test_rounded = y_test.round(1)

# Display y_pred and y_test in DataFrames
st.subheader('Predicted vs Actual Ad Revenue:')
df_results = pd.DataFrame({'Actual': y_test_rounded, 'Predicted': y_pred_rounded})
st.write(df_results)

# Plot predicted vs actual ad revenue
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Ad Revenue')
plt.title('Predicted vs Actual Ad Revenue')
plt.legend()
#st.pyplot(plt)

# Input for rating and channel
rating = st.number_input('Rating', min_value=0.00, max_value=10.00, step=0.1)
channel = st.selectbox('Channels', data['Channels'].unique())

# Make prediction
input_data = pd.DataFrame([[rating, channel]], columns=['Rating', 'Channels'])
prediction = model.predict(input_data)
# Display prediction
st.write('Predicted Ad Revenue:', prediction[0])