# src/prepare_data.py
import pandas as pd
import argparse


def load_world_bank(path):
df = pd.read_csv(path)
# Expect columns: country, year, co2_per_capita, gdp_per_capita, population, energy_use, urban_pct
return df


def engineer_features(df):
df = df.sort_values(['country','year'])
# lag features: co2 lag 1,2,3
for l in [1,2,3]:
df[f'co2_lag_{l}'] = df.groupby('country')['co2_per_capita'].shift(l)
# rolling mean last 3 years
df['co2_roll3'] = df.groupby('country')['co2_per_capita'].rolling(3).mean().reset_index(0,drop=True)
# percent change gdp
df['gdp_pct_change'] = df.groupby('country')['gdp_per_capita'].pct_change()
df = df.dropna()
return df

<def main(input, out):
df = load_world_bank(input)
df_proc = engineer_features(df)
df_proc.to_csv(out, index=False)
print(f'Processed saved to {out}, rows: {len(df_proc)}')


if __name__ == '__main__':
p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--out', required=True)
args = p.parse_args()
main(args.input, args.out)


# src/features.py
import numpy as np


def select_features(df):
feature_cols = [c for c in df.columns if c.startswith('co2_lag_')] + ['co2_roll3','gdp_per_capita','population','energy_use','urban_pct','gdp_pct_change']
X = df[feature_cols]
y = df['co2_per_capita']
return X, y

# src/train_model.py
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
from features import select_features



def train(df_path, out_path):
df = pd.read_csv(df_path)
X, y = select_features(df)
groups = df['country']
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups))
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]


# Baseline
ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred_base = ridge.predict(X_val)


# Main model
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)


metrics = {
'ridge_mae': mean_absolute_error(y_val, y_pred_base),
'rf_mae': mean_absolute_error(y_val, y_pred),
'rf_rmse': mean_squared_error(y_val, y_pred, squared=False)
}
print('Metrics:', metrics)


# Feature importances
importances = rf.feature_importances_
feat_names = X.columns.tolist()
plt.figure(figsize=(8,5))
plt.barh(feat_names, importances)
plt.title('Feature importances')
plt.tight_layout()
plt.savefig('screenshots/feat_importance.png')


joblib.dump(rf, out_path)
print(f'Model saved to {out_path}')


if __name__ == '__main__':
p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--model_out', required=True)
args = p.parse_args()
train(args.input, args.model_out)


# src/predict.py
import pandas as pd
import joblib
from features import select_features


def predict(model_path, data_csv):
model = joblib.load(model_path)
df = pd.read_csv(data_csv)
X, _ = select_features(df)
preds = model.predict(X)
df['pred_co2'] = preds
df.to_csv('predictions.csv', index=False)
print('Predictions saved to predictions.csv')


if __name__ == '__main__':
import sys
predict(sys.argv[1], sys.argv[2])


# src/app_streamlit.py
import streamlit as st
import pandas as pd
import joblib


st.title('CO₂ Forecaster — SDG13 Demo')
model = joblib.load('models/rf_model.pkl')


uploaded = st.file_uploader('Upload processed CSV (from prepare_data)', type='csv')
if uploaded:
df = pd.read_csv(uploaded)
X = df[[c for c in df.columns if c.startswith('co2_lag_')] + ['co2_roll3','gdp_per_capita','population','energy_use','urban_pct','gdp_pct_change']]
preds = model.predict(X)
df['pred_co2'] = preds
st.dataframe(df[['country','year','co2_per_capita','pred_co2']].head(20))
st.line_chart(df.set_index('year')[['co2_per_capita','pred_co2']].dropna())


