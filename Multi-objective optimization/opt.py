import joblib
import glob
import os
import numpy as np
import pandas as pd

try:
    # Load three XGBoost models
    model_CA = joblib.load(r'XGBoost_CA.pkl')
    model_OIR = joblib.load(r'XGBoost_OIR.pkl')
    model_Mw = joblib.load(r'XGBoost_Mw.pkl')
    print("The model has finished loading")
except Exception as e:
    print(f"Model loading failed: {e}")
    raise

# Define feature columns (based on the previous model training features)
feature_cols = ['R1', 'R2', 'R3', 'Metal_NPA', 'd_cp-M', 'd_B-M', 'E_els', 'E_rep', 'LUMO', '1-octene', 'Al/M', 'T']

# Make predictions for each CSV file and add a prediction column
input_dir = 'ansa.csv/non-bridged.csv'
all_files = glob.glob(os.path.join(input_dir, '*.csv'))

print(f"find {len(all_files)} CSV files in {input_dir} for prediction.")

for idx, file in enumerate(all_files):
    try:
        df = pd.read_csv(file)
        print(f"Processing file: {os.path.basename(file)}, Rows: {len(df)}")

        # Check whether all feature columns are included
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            print(f"File {file} is missing feature columns: {missing_cols}, skipping")
            continue

        # Feature extraction
        X = df[feature_cols]

        # Make predictions
        df['CA_pred'] = model_CA.predict(X)
        df['OIR_pred'] = model_OIR.predict(X)
        df['Mw_pred'] = model_Mw.predict(X)

        # Save the updated file (optional, if you want to keep the prediction results)
        # df.to_csv(file, index=False)

        print(f"已完成预测: {os.path.basename(file)}")

    except Exception as e:
        print(f"Processing file {file} failed: {e}")
        continue

print("All files have been predicted.")


# Iterate through each CSV file, progressively accumulate the Pareto front points, 
# repeatedly accumulate them and then determine the next Pareto front, 
# and finally save all the accumulated Pareto front points.

pareto_points = None

def filter_df(df):
    return df[
        (df['CA_pred'] >= 6) & (df['CA_pred'] <= 9.5) &
        (df['incorp_pred'] >= 5) & (df['incorp_pred'] <= 45) &
        (df['Mw_pred'] >= 1) & (df['Mw_pred'] <= 3.5) &
        (df['T'] >= 20) & (df['T'] <= 160)
    ].copy()


def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1) | np.all(costs[is_efficient] == c, axis=1)
            is_efficient[i] = True
    return is_efficient

for idx, file in enumerate(all_files):
    df = pd.read_csv(file)
    # Check whether all feature columns are included
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"文件 {file} 缺少特征列: {missing_cols}，跳过")
        continue

    X = df[feature_cols]
    df['CA_pred'] = model_CA.predict(X)
    df['incorp_pred'] = model_OIR.predict(X)
    df['Mw_pred'] = model_Mw.predict(X)
    df_final = filter_df(df)
    if len(df_final) == 0:
        continue
    if pareto_points is None:
        pareto_points = df_final
    else:
        combined = pd.concat([pareto_points, df_final], ignore_index=True)
        costs = combined[['CA_pred', 'incorp_pred', 'Mw_pred']].values
        pareto_mask = is_pareto_efficient(costs)
        pareto_points = combined[pareto_mask]
    print(f'{idx+1} out of {len(all_files)} files processed; current number of Pareto points: {len(pareto_points)}')

if pareto_points is not None and len(pareto_points) > 0:
    pareto_points.to_csv('ansa/nonbridge_all_pareto_points_incremental.csv', index=False)  
    print(f'All incremental Pareto front points have been saved to nsa/nonbridge_all_pareto_points_incremental.csv. Number of points: {len(pareto_points)}')
else:
    print('No data matching the criteria was found.')


# Create a three-dimensional plot of all points on the Pareto front (including the Pareto points and their projections)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8), dpi=150)
ax = fig.add_subplot(111, projection='3d')
CA = pareto_points['CA_pred'].values
incorp = pareto_points['incorp_pred'].values
Mw = pareto_points['Mw_pred'].values
# Plot the Pareto front points
ax.scatter(CA, incorp, Mw, c="#1e09d2", marker='^', s=80, label='Pareto Front', edgecolor='k', linewidth=0.8)
# Three-sided mapping points with transparency
ax.scatter(CA, incorp, np.full_like(Mw, Mw.min()), c="#f11417", marker='o', s=60, alpha=0.4, label='CA-OIR plane')
ax.scatter(CA, np.full_like(incorp, incorp.min()), Mw, c="#34cd2f", marker='^', s=60, alpha=0.4, label='CA-Mw plane')
ax.scatter(np.full_like(CA, CA.min()), incorp, Mw, c="#fb7e09", marker='v', s=60, alpha=0.4, label='OIR-Mw plane')
ax.set_xlabel('CA_pred', fontsize=14, labelpad=12)
ax.set_ylabel('OIR_pred', fontsize=14, labelpad=12)
ax.set_zlabel('Mw_pred', fontsize=14, labelpad=12)
ax.set_title('Pareto Front with 3-Plane Projections', fontsize=16, pad=20)
ax.view_init(elev=22, azim=35)
ax.w_xaxis.line.set_zorder(100)
ax.w_yaxis.line.set_zorder(100)
ax.w_zaxis.line.set_zorder(100)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('Mw$_{pred}$', fontsize=14, labelpad=18, rotation=90)
ax.legend(loc='upper left', fontsize=9, frameon=True, facecolor='white')
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('nsa/nonbridge_all_pareto_points_incremental.csv')

X = df['CA_pred']
Y = df['OIR_pred']
Z = df['Mw_pred']

fig = plt.figure(figsize=(12, 8), dpi=150)
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_trisurf(X, Y, Z, cmap='plasma', alpha=0.8, edgecolor='none')


fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, pad=0.1)

ax.scatter(X, Y, Z, c='red', marker='^', s=50, label='Pareto Points')

ax.set_xlabel('CA_pred', fontsize=14, labelpad=12)
ax.set_ylabel('OIR_pred', fontsize=14, labelpad=12)
ax.set_zlabel('Mw_pred', fontsize=14, labelpad=12)
ax.set_title('3D Pareto Surface (Color Adjustable)', fontsize=16, pad=20)
ax.view_init(elev=22, azim=35)
ax.legend(loc='upper left', fontsize=12, frameon=True, facecolor='white')
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show()