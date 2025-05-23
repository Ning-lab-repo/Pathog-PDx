# PathogPDx: A Multicenter Study on Pediatric Early  Diagnosis of 22 Respiratory Tract Pathogens in Single and Mixed-Infections Using Machine Learning Approaches

Upper respiratory tract infection, bronchitis, and pneumonia are the three most common respiratory diseases in children, which are often caused by pathogens infection in clinical practice. They have similar symptoms, but treatment plans vary with different respiratory tract infections (RTIs) pathogens. This study aimed to establish and validate an explainable prediction model based on the machine learning (ML) approach for pan-pathogens in children, enabling more precise investigation and individualized clinical management.
This retrospective multicenter study was conducted on ill children for the derivation and validation of the prediction model. The study included 133,162 children admitted to The First Affiliated Hospital of Xiamen University (XMFH) from January 2015 to September 2023, Xiamen’s Children Hospital (XMCH) from January 2021 to December 2023, Shenzhen Second People’s Hospital (SSPH) from January 2018 to December 2023, PIC, and MIMIC-III databases. The derivation cohort, consisting of XMFH, PIC, and MIMIC-III databases, was separated for training and internal validation, and 2 external cohorts from XMCH and SSPH was employed for external validation. With 40 medical characteristics easily obtained or evaluated from the first examination after admission, ML algorithms were used to construct prediction models. Several evaluation indexes, including the area under thereceiver-operating-characteristic curve (AUC), were used to compare the predictive performance. The SHapley Additive exPlanation method was used to rank the feature importance and explain the final model.

Code:

Data preparation:

1 data preparation.py

Base models:

2 RandomForest.py

3 GBDT.py

4 XGB.py

5 LightGBM.py

Pathog-PDx model:

6 PathogPDx.py

Result:

demo.py
