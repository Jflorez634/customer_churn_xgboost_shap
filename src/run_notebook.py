import nbformat
from nbclient import NotebookClient
import sys

nb_path = "semana_02_churn_ml_shap/notebooks/01_churn_prediction_y_shap.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': 'semana_02_churn_ml_shap/notebooks/'}})
print("Executing notebook...")
client.execute()

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook executed and saved.")
