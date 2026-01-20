from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.model_selection import train_test_split
import joblib
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
data_targetClass=data['Infection site']
data_feature=data_enc
X_train, X_test, y_train, y_test=train_test_split(data_feature,data_targetClass,stratify=data_targetClass,test_size=0.3,random_state=42)
#imp=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imp=KNNImputer(n_neighbors=300)
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)
data_encImpute=pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)], axis=0)
data_encImpute.columns=data_enc.columns
data_encImpute.to_csv('~.csv')
joblib.dump(imp, 'knn_imputer.joblib')