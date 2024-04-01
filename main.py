import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from fastai.tabular.all import *
from fastai.vision.all import *

energy_efficiency = pd.read_csv('https://archive.ics.uci.edu/static/public/242/data.csv')

print(energy_efficiency.head())

# Data Preparation
splits = RandomSplitter(valid_pct=0.2)(range_of(energy_efficiency))
to = TabularPandas(
    df=energy_efficiency,
    procs=[Normalize, Categorify],
    cat_names=['X6', 'X8'],
    cont_names=['X1', 'X2', 'X3', 'X4', 'X5', 'X7'],
    y_names=['Y1', 'Y2'],
    splits=splits
)

# variable_names_mapping = {
#     'X1': 'Relative Compactness',
#     'X2': 'Surface Area',
#     'X3': 'Wall Area',
#     'X4': 'Roof Area',
#     'X5': 'Overall Height',
#     'X6': 'Orientation',
#     'X7': 'Glazing Area',
#     'X8': 'Glazing Area Distribution',
# }

# energy_efficiency.rename(columns=variable_names_mapping, inplace=True)
# energy_efficiency.rename(columns={'Y1': 'Heating Load', 'Y2': 'Cooling Load'}, inplace=True)

# Initialize the TabularDataLoaders
dls = to.dataloaders(bs=64)

# Define your tabular learner
learn = tabular_learner(dls, metrics=[rmse])

# Fine-tune the model
learn.fit_one_cycle(38, lr_max=5e-3)

learn.summary()

valid_dl = dls.valid
valid_preds = learn.get_preds(dl=valid_dl)
rmse_valid = rmse(valid_preds[0], valid_preds[1])
print("RMSE on validation set:", rmse_valid)

# Get predictions for validation set
preds, _ = learn.get_preds(dl=valid_dl)

# Extract Y1 and Y2 predictions
y1_pred, y2_pred = preds[:, 0].numpy(), preds[:, 1].numpy()

# Extract Y1 and Y2 actual values
y1_actual, y2_actual = valid_dl.items.iloc[:, -2].values, valid_dl.items.iloc[:, -1].values

# Plotting Y1
plt.figure(figsize=(8, 6))
plt.scatter(y1_actual, y1_pred, color='blue', label='Y1 Predictions')
plt.plot(
    [y1_actual.min(), y1_actual.max()],
    [y1_actual.min(), y1_actual.max()],
    color='red',
    linestyle='--',
    label='Ideal Line')
plt.title('Y1 Predictions vs Actual')
plt.xlabel('Actual Y1')
plt.ylabel('Predicted Y1')
plt.legend()
plt.grid(True)
plt.show()

# Plotting Y2
plt.figure(figsize=(8, 6))
plt.scatter(y2_actual, y2_pred, color='green', label='Y2 Predictions')
plt.plot(
    [y2_actual.min(), y2_actual.max()],
    [y2_actual.min(), y2_actual.max()],
    color='red',
    linestyle='--',
    label='Ideal Line'
)
plt.title('Y2 Predictions vs Actual')
plt.xlabel('Actual Y2')
plt.ylabel('Predicted Y2')
plt.legend()
plt.grid(True)
plt.show()
