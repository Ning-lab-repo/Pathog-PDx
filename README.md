# Development and validation of a machine learning-based diagnostic system for 22 pediatric respiratory pathogens: a large-scale multicenter study

## Introduction

Pediatric respiratory tract infections continue to pose a significant global health challenge, with current diagnostic approaches often limited by delays in pathogen identification that frequently lead to empirical antibiotic use. While artificial intelligence has demonstrated potential in infection diagnosis, existing models have largely focused on single pathogens or have not been specifically optimized for pediatric populations. To achieve species-level identification of respiratory pathogens and prognosticates critical outcomes, we developed Pathog-PDx, a multi-task system that classifies 22 infectious pathogens and predicts ICU admission risk, integrating predicted sites of infection as an auxiliary factor in pathogen identification, applicable to both monoinfections and mixed infection

Unlike conventional approaches, Pathog-PDx processes accessible clinical variables (laboratory results, vital signs, demographics) to identify pathogens and clinically relevant co-infection patterns—such as Mycoplasma pneumoniae with respiratory syncytial virus. The model was  trained on data from 134,500 children across diverse age groups and disease severity levels. Through explainable AI (SHAP analysis), the model reveals interpretable biomarkers, enhancing clinical trust and decision support.

In evaluations across multiple cohorts (including prospective assessment of 1,338 patients), Pathog-PDx showed promising results with AUCs of 0.86-0.90 for distinguishing bacterial, viral and fungal infections, and 0.88 for identifying mixed infections. The study was approved by institutional review boards at all participating centers. The system has been implemented as a practical web-based decision support tool (https://pathogpdx.zzu.edu.cn).

<img src="FigTable/Figure1.png" width="1000"/>

## Prerequisites:

The tool was developed using the following dependencies:

1. Python (3.11.7 or greater)
2. NumPy (1.26.4 or greater).
3. pandas (2.1.4 or greater).
4. matplotlib (3.8.0 or greater).
5. shap (0.47.2 or greater).
6. scikit-learn (1.4.2 or greater).
7. scipy (1.11.4 or greater).

## Code:

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

## Demo:

To make our Pathog-PDx more accessible and user-friendly, we have hosted it on website. This interactive demo allows users to experience the power of our model in real-time, providing an intuitive interface for uploading diagnostic information and receiving diagnostic predictions. Check out our demo https://pathogpdx.zzu.edu.cn/predict to see our model in action and explore its potential.
