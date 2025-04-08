from sklearn import feature_selection
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix
import scikitplot as skplt

data = pd.read_csv('~.csv')
data['Infection site'] = data['Infection site'].map({
    'lung': 12, 'upper airway': 14, 'other': 13, 'L+U': 4, 'L+O': 3,
    'bronchus': 11, 'L+B': 1, 'L+U+O': 5, 'L+B+O': 2, 'Three': 7, 'TO': 6,
    'U+O': 10, 'U+B': 8, 'B+O': 0, 'U+B+O': 9
})

data_Catogry = data[['Gender(M/F)', 'pathogens']]
data_Number = data[[
    'Age', 'ALT', 'AST', 'GOP/ALT', 'TBIL', 'DBIL', 'TP', 'ALB', 'GLB', 'A/G', 
    'ALP', 'GGT', 'CK', 'LDH', 'K', 'Na', 'Cl', 'P', 'UREA', "Cr'", 'UA', 'CO2', 
    'TG', 'WBC', 'GR%', 'M%', 'EOS%', 'B%', 'N%', 'L%', 'M', 'E', 'B', 'RBC',
    'Hb', 'MCV', 'MCHC', 'RDW', 'PLT', 'MPV', 'PDW'
]]

encoder = OrdinalEncoder()
encoder.fit(data_Catogry)
data_CatogryEnc = pd.DataFrame(encoder.transform(data_Catogry))
data_CatogryEnc.columns = data_Catogry.columns

data_enc = pd.concat([data_CatogryEnc, data[['Infection site']], data_Number], axis=1)
data_feature = data_enc[[
    'Gender(M/F)', 'Infection site', 'Age', 'ALT', 'AST', 'GOP/ALT',
    'TBIL', 'DBIL', 'TP', 'ALB', 'GLB', 'A/G', 'ALP', 'GGT', 'CK',
    'LDH', 'K', 'Na', 'Cl', 'P', 'UREA', "Cr'", 'UA', 'CO2', 'TG',
    'WBC', 'GR%', 'M%', 'EOS%', 'B%', 'N%', 'L%', 'M', 'E', 'B', 'RBC',
    'Hb', 'MCV', 'MCHC', 'RDW', 'PLT', 'MPV', 'PDW'
]]
data_targetClass = data_enc['pathogens']

models_thresholds = {
    'Ab': 0.88, 'Aspergillus': 0.94, 'BP': 0.865, 'Candida': 0.94, 
    'Chlmydia': 0.8, 'ecoli': 0.95, 'HAdV': 0.94, 'HBoV': 0.977, 
    'HCoV': 0.85, 'HI': 0.9, 'HMPV': 0.985, 'HPIV': 0.89, 
    'HRV': 0.79, 'IFV': 0.991, 'KLP': 0.96, 'MC': 0.99, 
    'MP': 0.4, 'otherfungi': 0.97, 'PA': 0.83, 'RSV': 0.76, 
    'SA': 0.845, 'SP': 0.88
}

def calculate_confusion_matrix(model_name, threshold):
    model = joblib.load(f'model/{model_name}.pkl')
    y_proba = model.predict_proba(data_feature)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    conf_matrix = confusion_matrix(data_targetClass, y_pred)
    return conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

results = {}
for model_name, threshold in models_thresholds.items():
    file_name = 'otherfungi' if model_name == 'of' else model_name
    results[model_name] = calculate_confusion_matrix(file_name, threshold)

values = [results[model_name][0, 0] for model_name in models_thresholds.keys()]
result_df = pd.DataFrame({'Value': values}, index=list(models_thresholds.keys()))

plt.figure(figsize=(20, 1))
ax = sns.heatmap(result_df.T, annot=False, cmap='Blues', cbar=True, fmt=".2f",
                cbar_kws={'label': 'Probability'})

ax.set_xticklabels(result_df.index, rotation=45, ha='center')
plt.tight_layout()
plt.show()