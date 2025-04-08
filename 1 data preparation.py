from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer,KNNImputer
data=pd.read_csv('~.csv')
data['Infection site'].unique()
data_Catogry=data[['Gender(M/F)','Infection site']]
data_Number=data[[ 'Age', 'ALT', 'AST', 'GOP/ALT',
       'TBIL', 'DBIL', 'TP', 'ALB', 'GLB', 'A/G', 'ALP', 'GGT', 'CK',
       'LDH', 'K', 'Na', 'Cl', 'P', 'UREA', "Cr'", 'UA', 'CO2', 'TG',
       'WBC', 'GR%', 'M%', 'EOS%', 'B%', 'N%', 'L%', 'M', 'E', 'B', 'RBC',
       'Hb', 'MCV', 'MCHC', 'RDW', 'PLT', 'MPV', 'PDW']]
encoder=OrdinalEncoder()
encoder.fit(data_Catogry)
data_CatogryEnc=pd.DataFrame(encoder.transform(data_Catogry))
data_CatogryEnc.columns=data_Catogry.columns
data_enc=pd.concat([data_CatogryEnc,data_Number],axis=1)
data_enc.dropna(thresh=data_enc.shape[1]*0.65,inplace=True)
data_enc.shape
#imp=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imp=KNNImputer(n_neighbors=300)
data_encImpute=pd.DataFrame(imp.fit_transform(data_enc))
data_encImpute.columns=data_enc.columns
data_encImpute.to_csv('~.csv')