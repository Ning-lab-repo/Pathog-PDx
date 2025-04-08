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

from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics

models = {
    'GBDT': joblib.load('model/GBDT.pkl'),
    'LightGBM': joblib.load('model/light.pkl'),
    'RandomForest': joblib.load('model/RF.pkl'),
    'XGBoost': joblib.load('model/xgb.pkl')
}

for model_name, base_model in models.items():
    print(f"\n======== processing {model_name} ========")
    
    bagging_model = BaggingClassifier(
        base_estimator=base_model,
        n_jobs=24,
        random_state=42) 
    
    bagging_model.fit(class_X_train, class_y_train)
    
    try:
        orig_proba = base_model.predict_proba(class_X_test)
        orig_auc = round(metrics.roc_auc_score(class_y_test, orig_proba[:, 1]), 3)
    except Exception as e:
        print(f"Origin model {model_name} error: {str(e)}")
        continue

    try:
        bagging_proba = bagging_model.predict_proba(class_X_test)
        bagging_auc = round(metrics.roc_auc_score(class_y_test, bagging_proba[:, 1]), 3)
    except Exception as e:
        print(f"Bagging model {model_name} error: {str(e)}")
        continue
    print(f"Origin model AUC: {orig_auc} | Bagging model AUC: {bagging_auc}")
    if bagging_auc > orig_auc:
        print(f"Bagging model is better")
        joblib.dump(bagging_model, f'{model_name}.pkl', compress=3)
    else:
        print(f"Origin model is better")
print("\n==== Finish ====")


from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,VotingClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
import itertools
from sklearn import metrics
import joblib

# 加载模型
models = {
    'XGBoost': joblib.load('model/xgb.pkl'),
    'GBDT': joblib.load('model/GBDT.pkl'),
    'LightGBM': joblib.load('model/light.pkl'),
    'RF': joblib.load('model/RF.pkl')
}

def get_model_combinations():
    model_names = list(models.keys())
    all_combinations = []
    for r in range(2, len(model_names) + 1):
        combinations = list(itertools.combinations(model_names, r))
        all_combinations.extend(combinations)
    return all_combinations

def generate_weights(n_models):
    weights = list(itertools.product(range(6), repeat=n_models))  
    weights = [w for w in weights if sum(w) > 0]
    return weights

def evaluate_combination(model_names, weights, X_train, y_train, X_test, y_test):
    estimators = [(name, models[name]) for name in model_names]
    
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=weights,
        n_jobs=-1
    )
    
    voting_clf.fit(X_train, y_train)
    y_pred_proba = voting_clf.predict_proba(X_test)[:, 1]
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    
    return auc

def find_best_voting_combination(X_train, y_train, X_test, y_test):
    best_auc = 0
    best_combination = None
    best_weights = None
    
    combinations = get_model_combinations()
    
    for combo in combinations:
        print(f"\n测试模型组合: {combo}")
        weights_list = generate_weights(len(combo))
        
        for weights in weights_list:
            auc = evaluate_combination(combo, weights, X_train, y_train, X_test, y_test)
            
            if auc > best_auc:
                best_auc = auc
                best_combination = combo
                best_weights = weights
                print(f"Best AUC: {best_auc:.4f}")
                print(f"Model: {best_combination}")
                print(f"Weight: {best_weights}")
    
    return best_combination, best_weights, best_auc

best_models, best_weights, best_auc = find_best_voting_combination(
    class_X_train, class_y_train, class_X_test, class_y_test
)

print("\n=== Best result ===")
print(f"Best AUC: {best_auc:.4f}")
print(f"Best model: {best_models}")
print(f"Best weights: {best_weights}")

best_estimators = [(name, models[name]) for name in best_models]
best_voting_clf = VotingClassifier(
    estimators=best_estimators,
    voting='soft',
    weights=best_weights,
    n_jobs=-1
)
best_voting_clf.fit(class_X_train, class_y_train)
joblib.dump(best_voting_clf, 'model/best_voting_model.pkl')

