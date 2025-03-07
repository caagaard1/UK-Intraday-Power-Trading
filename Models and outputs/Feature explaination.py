import pandas as pd
import glob
import os

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np
import joblib
from matplotlib import dates as mdates


# Load the XGBoost model
model_path = r"C:\Users\chrsr\PycharmProjects\pythonProject\Models and outputs\xgb_model_intra_13_00.joblib"
loaded_model = joblib.load(model_path)

# Get feature importances
feature_importances = loaded_model.feature_importances_
feature_names = loaded_model.get_booster().feature_names

# Create a list of tuples (feature name, importance)
feature_importance_pairs = list(zip(feature_names, feature_importances))

# Sort the list by importance in descending order
sorted_features = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)

# Get the top 5 features
top_5_features = sorted_features[:10]
print(top_5_features)
for feature, importance in top_5_features:
    print(f"\nFeature: {feature}")
    print(f"Importance: {importance:.4f}")

    print("\nInterpretation of the feature's contribution:")

    # Get the feature values for the current feature
    feature_values = loaded_model.get_booster().trees_to_dataframe()
    feature_values = feature_values[feature_values['Feature'] == feature]

    # Analyze split conditions
    split_conditions = feature_values['Split'].dropna().tolist()
    if split_conditions:
        min_split = min(split_conditions)
        max_split = max(split_conditions)
        print(f"- The model uses '{feature}' to make decisions in the range of {min_split:.2f} to {max_split:.2f}.")
        print(f"- Higher values generally indicate {'increased' if feature_values['Gain'].mean() > 0 else 'decreased'} predictions.")

    # Analyze feature interactions
    interacting_features = feature_values['Feature'].value_counts().index.tolist()[1:]
    if interacting_features:
        print(f"- '{feature}' often interacts with: {', '.join(interacting_features[:3])}.")

    print(f"- This feature contributes {importance*100:.2f}% of the total importance in the model's decisions.")

    if importance > 0.1:
        print("- It has a very strong influence on the model's predictions.")
    elif 0.05 < importance <= 0.1:
        print("- It has a moderate influence on the model's predictions.")
    else:
        print("- It has a relatively minor influence on the model's predictions.")

print("\nNote: This interpretation is based on the overall model structure and may not reflect the exact impact on individual predictions.")
