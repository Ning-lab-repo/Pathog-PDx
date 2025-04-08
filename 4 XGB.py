from sklearn import feature_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
data=pd.read_csv('~.csv')
data_featureCate=data[['Gender(M/F)']]
data_featureNum=data[[  'Age', 'ALT', 'AST', 'GOP/ALT',
       'TBIL', 'DBIL', 'TP', 'ALB', 'GLB', 'A/G', 'ALP', 'GGT', 'CK',
       'LDH', 'K', 'Na', 'Cl', 'P', 'UREA', "Cr'", 'UA', 'CO2', 'TG',
       'WBC', 'GR%', 'M%', 'EOS%', 'B%', 'N%', 'L%', 'M', 'E', 'B', 'RBC',
       'Hb', 'MCV', 'MCHC', 'RDW', 'PLT', 'MPV', 'PDW']]

scaler=MinMaxScaler()
data_featureNum=scaler.fit_transform(data_featureNum)
data_featureCate=np.array(data_featureCate)
data_feature=np.hstack((data_featureCate,data_featureNum))
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTENC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
data_targetClass=data['Infection site']
X_train, class_X_test, y_train, class_y_test=train_test_split(data_feature,data_targetClass,stratify=data_targetClass,test_size=0.3,random_state=42)

categorical_features = [data.columns.get_loc('Gender(M/F)'), data_feature.columns.get_loc('Infection site')]

smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42)

class_X_train, class_y_train = smote_nc.fit_resample(X_train, y_train)
print(sum(y_train==0))
print(sum(y_train==1))
print(sum(class_y_train==0))
print(sum(class_y_train==1))

from scipy.stats import reciprocal
from scipy.stats import randint
from scipy.stats.distributions import expon,uniform,norm,poisson,bernoulli,expon,lognorm
from sklearn.model_selection import  GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier
clf=XGBClassifier()
param_distribute={'learning_rate':np.arange(0.01,0.5,0.1), 'n_estimators':[50,1000,1500,2000,3000],'loss':['log_loss', 'exponential'],'max_depth':np.arange(1, 20, 4),'min_samples_leaf':range(1, 8)
             ,'min_samples_split':np.arange(1, 20, 4),'max_features':['auto', 'sqrt', 'log2']}
cv=KFold(n_splits=3,shuffle=True,random_state=0)
Grid_clf=RandomizedSearchCV(clf,param_distributions=param_distribute,scoring='roc_auc_ovo',cv=cv,n_jobs=24)
#Grid_clf=GridSearchCV(clf,param_grid=param_distribute,scoring='roc_auc_ovo',cv=cv,n_jobs=24)
Grid_clf.fit(class_X_train,class_y_train)
best_param=Grid_clf.best_params_
Grid_clf.score(class_X_test,class_y_test) 
clf.set_params(**best_param)
clf.fit(class_X_train,class_y_train)
clf.score(class_X_test,class_y_test)

import os
from sklearn.utils import resample
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, recall_score,
                             confusion_matrix, precision_score, f1_score, matthews_corrcoef)
import numpy as np 

n_bootstraps = 100
auc_scores, accuracy_scores, sensitivity_scores, specificity_scores = [], [], [], []
ppv_scores, npv_scores, f1_scores, mcc_scores, youden_index_scores = [], [], [], [], []
    
for _ in range(n_bootstraps):
    y_score = clf.predict_proba(class_X_test)[:, 1]
    y_test_resampled, y_score_resampled = resample(class_y_test, y_score,n_samples=round(len(class_y_test)*99/100),random_state=np.random.randint(1, 100))
    auc_resampled = roc_auc_score(y_test_resampled, y_score_resampled)
    auc_scores.append(auc_resampled)

    y_pred_resampled = np.where(y_score_resampled > 0.5, 1, 0)
    accuracy_scores.append(accuracy_score(y_test_resampled, y_pred_resampled))
    sensitivity_scores.append(recall_score(y_test_resampled, y_pred_resampled))
    tn_resampled, fp_resampled, fn_resampled, tp_resampled = confusion_matrix(y_test_resampled, y_pred_resampled).ravel()
    specificity_scores.append(tn_resampled / (tn_resampled + fp_resampled))
    ppv_scores.append(precision_score(y_test_resampled, y_pred_resampled))
    npv_scores.append(tn_resampled / (tn_resampled + fn_resampled))
    f1_scores.append(f1_score(y_test_resampled, y_pred_resampled))
    youden_index_scores.append(sensitivity_scores[-1] + specificity_scores[-1] - 1)
    
def compute_ci(scores):
    sorted_scores = np.array(scores)
    sorted_scores.sort()
    lower = sorted_scores[int(0.025 * len(sorted_scores))]
    upper = sorted_scores[int(0.975 * len(sorted_scores))]
    return lower, upper
    
auc_ci = compute_ci(auc_scores)
accuracy_ci = compute_ci(accuracy_scores)
sensitivity_ci = compute_ci(sensitivity_scores)
specificity_ci = compute_ci(specificity_scores)
ppv_ci = compute_ci(ppv_scores)
npv_ci = compute_ci(npv_scores)
f1_ci = compute_ci(f1_scores)
mcc_ci = compute_ci(mcc_scores)
youden_index_ci = compute_ci(youden_index_scores)

print(auc_ci)
print(accuracy_ci)
print(sensitivity_ci)
print(specificity_ci)
print(ppv_ci)
print(npv_ci)
print(f1_ci)
print(youden_index_ci)

import joblib
joblib.dump(clf,'xgb.pkl',compress=3)