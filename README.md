# UMA_IE500_Data_Mining

# Predictive Modeling Pipeline for Structured Datasets

This project presents a complete data science pipeline for building and evaluating machine learning models using structured datasets. The pipeline includes data cleaning, exploratory data analysis and visualization, feature engineering, model training (using multiple algorithms), and comprehensive model selection and evaluation.

## 📁 Project Structure
(Important notebooks to see and run to get the dataset, or the model states #if not having the zip file containing everything)

```
[0] Data_Cleaning_Test
|   └── cleaning_trial.ipynb
[1] Data Exploration and Visualization
│   └── explore_and_vis_time.ipynb
[2] Feature Engineering_Test
│   └── Feature_selection_trial.ipynb
[3] [3] Full preparation_engineering and baseline model evaluation
│   ├── distribution check.ipynb
│   ├── full dataset preparation and baseline model evaluation.ipynb
│   └── shuffle closed model experiment notebook.ipynb
[4] Model Training
│   ├── XGBoost.ipynb
│   ├── decision_tree.ipynb
│   ├── knn.ipynb
|   ├── logiic_regression.ipynb
│   └── random_forest.ipynb
[5] Model Selection
│   ├── Model_eval_and_compare.ipynb
│   ├── check_for_ensemble_model.ipynb
│   └── Chosen_model_RF_vis_and_interpretation

Additional Files:
- .gitignore
- README.md
- _getting_the_data.py
- data_scaling.py
- feature_engineering_function.py
```

## 🧠 Model Overview

We explore and compare the performance of multiple machine learning models:

- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost
- logistic regression

Evaluation metrics include accuracy, precision, recall, F1-score, and ROC-AUC where applicable (e.g. not for KNN).
Furthermore, for final evaluation of candidate models - RF and XGB - further qualitative evaluation was conduncted. For details see the notebook: "Model_eval_and_compare"

## 🔧 Technologies Used

- Python (Jupyter Notebooks)
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn (for visualization)
etc.

## 🧑‍🤝‍🧑 Contributors

- **Chengyi Hua**, 2117289, [chengyi.hua@students.uni-mannheim.de](mailto:chengyi.hua@students.uni-mannheim.de)  
- **Shamalan Rajesvaran**, 2115475, [shamalan.rajesvaran@students.uni-mannheim.de](mailto:shamalan.rajesvaran@students.uni-mannheim.de)  
- **Bahri Selçuk Eşkil**, 2117150, [bahri.eskil@students.uni-mannheim.de](mailto:bahri.eskil@students.uni-mannheim.de)  
- **Nicolas Balzek**, 1709460, [nicolas.balzek@students.uni-mannheim.de](mailto:nicolas.balzek@students.uni-mannheim.de)  
- **Bofan Chen**, 2190412, [bofan.chen@students.uni-mannheim.de](mailto:bofan.chen@students.uni-mannheim.de)

## 🗓️ Last Updated

15.05.2025


