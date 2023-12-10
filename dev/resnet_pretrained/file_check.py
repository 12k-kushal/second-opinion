
import os
import pandas as pd

df = pd.read_csv("./../../data/labels.csv")

for file in df['subject_id']:
    file_full_name = file+"_1-1.png"
    if os.path.exists("./../../data/processed/"+file_full_name):
        print(file_full_name)