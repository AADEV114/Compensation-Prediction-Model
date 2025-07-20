import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('train_set.csv')
test_df = pd.read_csv('test_set.csv')

print(train_df.info())
print(train_df.describe())
print(train_df.isnull().sum())

for col in train_df.columns:
    if train_df[col].dtype == 'object':
        train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(train_df[col].mode()[0])
    else:
        train_df[col] = train_df[col].fillna(train_df[col].median())
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(train_df[col].median())

from sklearn.preprocessing import OrdinalEncoder

cat_cols = train_df.select_dtypes(include='object').columns.tolist()
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_df[cat_cols] = encoder.fit_transform(train_df[cat_cols])
test_df[cat_cols] = encoder.transform(test_df[cat_cols])

plt.figure(figsize=(12,8))
sns.heatmap(train_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
plt.savefig("Feature Correlation Heatmap.png")

target_col = 'Total_Compensation'
X = train_df.drop(columns=[target_col])
y = train_df[target_col]

from sklearn.preprocessing import StandardScaler

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
num_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, r2_score

y_pred = model.predict(X_val)
print("MAE:", mean_absolute_error(y_val, y_pred))
print("R2 Score:", r2_score(y_val, y_pred))

test_df_copy = test_df.copy()
test_df_copy[target_col] = model.predict(test_df)

test_df_copy.to_csv('compensation_predictions.csv', index=False)
print("Saved compensation_predictions.csv with all columns and prediction.")
