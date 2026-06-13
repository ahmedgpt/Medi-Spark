"""
Generate disease_symptoms.json from the training data.
Maps each disease to its list of associated symptoms.
"""
import os, json, warnings
warnings.filterwarnings("ignore")

TRAIN_PATH = r"C:\Users\This pc\Desktop\med_spark material\datase\Training.csv"
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "disease_symptoms.json")

import pandas as pd
df = pd.read_csv(TRAIN_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
bad = [c for c in df.columns if "unnamed" in c]
if bad:
    df.drop(columns=bad, inplace=True)

target_col = "prognosis" if "prognosis" in df.columns else df.columns[-1]
symptom_cols = [c for c in df.columns if c != target_col]

disease_map = {}
for disease in sorted(df[target_col].unique()):
    rows = df[df[target_col] == disease]
    active = [col for col in symptom_cols if rows[col].sum() > 0]
    disease_map[disease] = active

with open(OUTPUT, "w") as f:
    json.dump(disease_map, f, indent=2)

print(f"Saved {len(disease_map)} diseases to {OUTPUT}")
for d, syms in disease_map.items():
    print(f"  {d}: {len(syms)} symptoms")
