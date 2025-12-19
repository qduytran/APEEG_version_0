import pickle
import os
import streamlit as st
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



def add_label(features_subjects, label):
    features_subjects['Label'] = label
    return features_subjects

def load_features_subjects(file):
    features_subjects = pd.read_csv(file, header=0, index_col=0)
    return features_subjects


def train_ml(X, y, num_folds = 5, num_loops = 10, save_path=None):
    model_results = pd.DataFrame(columns=["Model Name", "Mean", "Std"])
    models = [
        LogisticRegression(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        SVC(),
        # DecisionTreeClassifier(),
        # KNeighborsClassifier(),
        # GaussianNB(),
        # XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        LGBMClassifier()
    ]
    mean_accuracies = np.empty((num_loops,1))

    for model in models:
        model_name = model.__class__.__name__
        mean_accuracies = np.empty((num_loops,1))

        for state in range(1,num_loops+1):
            # Perform K-fold cross-validation and calculate accuracy scores
            stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=state)
            cv_scores = cross_val_score(model, X, y, cv=stratified_kfold)

            # Calculate and print mean accuracy across all folds
            mean_accuracy = cv_scores.mean()
            mean_accuracies[state-1] = mean_accuracy
        
        mean_accuracies_looped = np.mean(mean_accuracies)*100
        std_accuracies_looped  = np.std(mean_accuracies)*100
        model_result = pd.DataFrame([{
            "Model Name": model_name,
            "Mean": mean_accuracies_looped,
            "Std": std_accuracies_looped
        }])
        model_results = pd.concat([model_results, model_result], ignore_index=True)

        if save_path is not None:
            model.fit(X,y)
            model_file = os.path.join(save_path, f"model_{model_name}.pkl")
            with open(model_file, "wb") as f:
                pickle.dump(model, f)

    return model_results

def predict_ml(features_subjects, models):

    predictions = {}

    for name, model in models.items():
        y_pred = model.predict(features_subjects)
        predictions[name] = y_pred

    pred_results = pd.DataFrame(predictions, index=features_subjects.index)

    return pred_results



def ui_load_features_train_groups():
    uploaded_g1 = st.file_uploader("Upload Group 1:", type=["csv"])
    label_g1 = st.text_input("Label for Group 1:", value="")
    uploaded_g2 = st.file_uploader("Upload Group 2:", type=["csv"])
    label_g2 = st.text_input("Label for Group 2:", value="")

    if uploaded_g1 is not None and uploaded_g2 is not None and label_g1.strip() and label_g2.strip():
        try:
            label_g1 = int(label_g1)
            label_g2 = int(label_g2)
        except ValueError:
            st.error("Labels must be numeric.")
            return None

        features_group1 = load_features_subjects(uploaded_g1)
        features_group1 = add_label(features_subjects=features_group1, label=label_g1)
        features_group2 = load_features_subjects(uploaded_g2)
        features_group2 = add_label(features_subjects=features_group2, label=label_g2)

        df = pd.concat([features_group1, features_group2])
        st.markdown("Feature Set:")
        st.dataframe(df)

        X = df.drop(columns=['Label'])
        y = df['Label']

        return X, y
    return None


def ui_load_features_predict_subjects():

    uploaded_subjects = st.file_uploader("Upload Features for Prediction:", type=["csv"])

    if uploaded_subjects:
        features_subjects = load_features_subjects(uploaded_subjects)

        return features_subjects 

def ui_load_models():
    uploaded_models = st.file_uploader("Upload models (.pkl)", type=["pkl"], accept_multiple_files=True)
    models = {}

    if uploaded_models:
        for file in uploaded_models:
            model_name = file.name.replace(".pkl", "")
            models[model_name] = pickle.load(file)

    return models

def UI_train_ml():
    st.sidebar.header("", divider="orange")
    st.sidebar.header(":orange[Classification]")

    st.sidebar.subheader("Classification Adjustments")
    num_folds = st.sidebar.slider("Number of folds:", value=5, min_value=5, max_value=20, step=5)
    num_loops = st.sidebar.slider("Number of loops:", value=5, min_value=5, max_value=100, step=5)
    save_path = st.sidebar.text_input("Type path if you want to save models:", value=None)

    data = ui_load_features_train_groups()
    if data is not None:
        X, y = data
        if not X.empty and not y.empty:

            model_results = train_ml(X=X,y=y,num_folds=num_folds,num_loops=num_loops, save_path=save_path)
            
            st.header("", divider="rainbow")
            st.header(":orange[Classification]")
            st.dataframe(model_results)
    return 

def UI_predict_ml(features_subjects = None):

    st.header("", divider="rainbow")
    st.header(":orange[Prediction]")
    
    if features_subjects is None:
        features_subjects = ui_load_features_predict_subjects()
    models = ui_load_models()

    if features_subjects is not None and not features_subjects.empty and models:
        pred_results = predict_ml(features_subjects=features_subjects, models=models)
        st.dataframe(pred_results) 

    return


# UI_train_ml()

# UI_predict_ml()

