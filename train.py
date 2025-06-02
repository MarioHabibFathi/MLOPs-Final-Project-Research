import pandas as pd
import numpy as np
import json
import pickle
import os
import re
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
from scipy.stats import loguniform
# from scipy import stats
import time
import tracemalloc
import psutil
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from mediapipe.framework.formats import landmark_pb2
# import cv2
# from collections import deque
import joblib

seed = 42
num_points = 21

def preprocess_hand_landmarks(df, img_width=1920, img_height=1080, num_points=21, palm_idx=1, index_tip_idx=13):
    """
    Preprocess hand landmarks:
      1. Normalize coordinates by image resolution.
      2. Recenter coordinates so that the palm (x1,y1) becomes (0,0).
      3. Scale coordinates using the Euclidean distance from palm to middle finger tip (x13,y13).
    """
    df_new = df.copy()
    for i in range(1, num_points+1):
        df_new[f'x{i}'] = df_new[f'x{i}'] / img_width
        df_new[f'y{i}'] = df_new[f'y{i}'] / img_height

    palm_x = df_new[f'x{palm_idx}'].copy()
    palm_y = df_new[f'y{palm_idx}'].copy()
    for i in range(1, num_points+1):
        df_new[f'x{i}'] = df_new[f'x{i}'] - palm_x
        df_new[f'y{i}'] = df_new[f'y{i}'] - palm_y
        
    scale = np.sqrt(df_new[f'x{index_tip_idx}']**2 + df_new[f'y{index_tip_idx}']**2) + 1e-8
    for i in range(1, num_points+1):
        df_new[f'x{i}'] = df_new[f'x{i}'] / scale
        df_new[f'y{i}'] = df_new[f'y{i}'] / scale    
    
    return df_new


def compute_distance_features(df, num_points=21, palm_idx=1):
    """
    For a DataFrame with columns x1, y1, ... x{num_points}, y{num_points},
    compute Euclidean distances from the palm (landmark with index palm_idx)
    and return a new DataFrame with columns: pt1_dist, pt2_dist, ..., pt{num_points}_dist.
    """
    df_dist = df.copy()
    palm_x = df_dist[f'x{palm_idx}']
    palm_y = df_dist[f'y{palm_idx}']

    for i in range(1, num_points+1):
        df_dist[f'pt{i}_dist'] = np.sqrt((df_dist[f'x{i}'] - palm_x)**2 +
                                         (df_dist[f'y{i}'] - palm_y)**2)
    dist_columns = [f'pt{i}_dist' for i in range(1, num_points+1)]
    return df_dist[dist_columns]

def drop_highly_correlated_features(df_dist, threshold=0.99):
    """
    Given a DataFrame of distance features, compute its absolute correlation matrix,
    then iterate through the columns in order. For each new column, if it is highly 
    correlated (>= threshold) with any previously encountered (kept) column, print a 
    message and mark it for dropping.
    
    Returns:
      to_drop : List of feature names (distance columns) to drop.
      corr_matrix : The absolute correlation matrix of df_dist.
    """
    corr_matrix = df_dist.corr().abs()
    kept = []
    to_drop = []
    
    for col in df_dist.columns:
        drop_flag = False
        for kept_col in kept:
            if corr_matrix.loc[col, kept_col] >= threshold:
                print(f"{col} is correlated with {kept_col} (correlation = {corr_matrix.loc[col, kept_col]:.4f}). Dropping {col}.")
                drop_flag = True
                to_drop.append(col)
                break
        if not drop_flag:
            kept.append(col)
    
    return to_drop, corr_matrix

def drop_landmarks_from_data(df_raw, to_drop):
    """
    Given a raw DataFrame with columns x{i} and y{i}, drop the columns corresponding 
    to each distance feature in the to_drop list (e.g., if 'pt5_dist' is in to_drop, drop x5 and y5).
    """
    df_new = df_raw.copy()
    for dist_col in to_drop:
        i = int(dist_col.replace('pt', '').replace('_dist', ''))
        for col in [f'x{i}', f'y{i}']:
            if col in df_new.columns:
                df_new.drop(columns=col, inplace=True)
    distance_cols = [f'pt{i}_dist' for i in range(1, num_points+1)]
    df_new.drop(columns=[col for col in distance_cols if col in df_new.columns], inplace=True)
    return df_new



def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_inference_time(model, X_test, num_runs=10):
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        model.predict(X_test)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)
    return np.mean(times)

def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def refine_bounds(best_val, top_vals, lower_factor=0.9, upper_factor=1.1):
    """
    Given a best value (best_val) and a list of top competitor values (top_vals) for a hyperparameter,
    compute refined lower and upper bounds.
    
    We first compute candidate bounds as:
         candidate_lower = best_val * lower_factor
         candidate_upper = best_val * upper_factor
    Then we look at competitor values:
      - For the lower bound, if any competitor is less than best_val and greater than candidate_lower,
        we set new_lower to the maximum of those competitor values.
      - For the upper bound, if any competitor is greater than best_val and less than candidate_upper,
        we set new_upper to the minimum of those competitor values.
      
    Finally, we ensure that best_val lies within the refined range.
    If not, we revert that bound to the candidate value.
    """
    candidate_lower = best_val * lower_factor
    candidate_upper = best_val * upper_factor
    
    print(f'top vals: {top_vals}')
    competitors_below = [val for val in top_vals if val < best_val]
    print(f'competitors below: {competitors_below}')
    if competitors_below:
        new_lower = max(candidate_lower, max(competitors_below))
    else:
        new_lower = candidate_lower

    competitors_above = [val for val in top_vals if val > best_val]
    print(f'competitors above: {competitors_above}')
    if competitors_above:
        new_upper = min(candidate_upper, min(competitors_above))
    else:
        new_upper = candidate_upper

    if not (new_lower <= best_val <= new_upper):
        new_lower, new_upper = candidate_lower, candidate_upper
        
    return new_lower, new_upper

    
def numpy_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        
def save_tuning_outputs(best_model, best_params, acc, performance_metrics, file_path):
    """
    Saves the outputs from the tuning process to the given file path.
    
    Outputs saved:
      - best_model: Pickle file.
      - best_params: Text file, JSON file, and Pickle file.
      - acc (accuracy): Text file, JSON file, and Pickle file.
      - performance_metrics: Text file, JSON file, and Pickle file.
    
    Parameters:
      best_model: The best estimator returned by your tuning function.
      best_params: The dictionary of best hyperparameters.
      acc: The test accuracy (float).
      performance_metrics: A dictionary of additional metrics (e.g., inference time, memory used, etc.).
      file_path: The directory path where to save the files.
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    model_filename = os.path.join(file_path, "best_model.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(best_model, f)
    
    best_params_txt = os.path.join(file_path, "best_params.txt")
    with open(best_params_txt, "w") as f:
        f.write(str(best_params))
    
    best_params_json = os.path.join(file_path, "best_params.json")
    with open(best_params_json, "w") as f:
        json.dump(best_params, f, indent=4, default=numpy_default)
    
    best_params_pickle = os.path.join(file_path, "best_params.pkl")
    with open(best_params_pickle, "wb") as f:
        pickle.dump(best_params, f)
    
    acc_txt = os.path.join(file_path, "accuracy.txt")
    with open(acc_txt, "w") as f:
        f.write(str(acc))
    
    acc_json = os.path.join(file_path, "accuracy.json")
    with open(acc_json, "w") as f:
        json.dump({"accuracy": acc}, f, indent=4, default=numpy_default)
    
    acc_pickle = os.path.join(file_path, "accuracy.pkl")
    with open(acc_pickle, "wb") as f:
        pickle.dump(acc, f)
     
    perf_txt = os.path.join(file_path, "performance_metrics.txt")
    with open(perf_txt, "w") as f:
        f.write(str(performance_metrics))
    
    perf_json = os.path.join(file_path, "performance_metrics.json")
    with open(perf_json, "w") as f:
        json.dump(performance_metrics, f, indent=4, default=numpy_default)
    
    perf_pickle = os.path.join(file_path, "performance_metrics.pkl")
    with open(perf_pickle, "wb") as f:
        pickle.dump(performance_metrics, f)

def preprocess(df, labels, test_size=0.2, use_preprocessed=True, hand_variant=False, hand_variant_type='palm'):
    
    to_drop = None
    if use_preprocessed:
        df = preprocess_hand_landmarks(df)
    
    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=test_size, random_state=seed, stratify=labels)

    if hand_variant: 
        if hand_variant_type == 'palm': 
            X_train_dist = compute_distance_features(X_train , num_points=num_points, palm_idx=1)
        elif hand_variant_type == 'index':
            X_train_dist = compute_distance_features(X_train , num_points=num_points, palm_idx=13)

        to_drop, corr_matrix_train = drop_highly_correlated_features(X_train_dist, threshold=0.98)
        X_train = drop_landmarks_from_data(X_train, to_drop)
        X_test = drop_landmarks_from_data(X_test, to_drop)
        
    
    return X_train, X_test, y_train, y_test, to_drop 
    


def tune_model_with_performance(X_train, X_test, y_train, y_test, variant_name, model_type="SVM", use_GPU=False, seed=seed, run_id=None):
    """
    Multi-stage hyperparameter tuning and performance evaluation for different model types.
    Supported model_type: "SVM", "DecisionTree", RandomForest, "LogisticRegression", "XGBoost".
    
    The function performs:
      - A random search over a wide parameter space.
      - A grid search around the best parameters.
      - A second random search in a refined region.
      - A final grid search over a narrow grid.
    
    It then evaluates the best model on the test set, logging accuracy, precision, recall, f1_score,
    inference time, memory usage, and CPU usage via MLflow.
    
    Prints all performance metrics to the screen.
    
    Returns the best model, best parameters, test accuracy, and a dictionary of performance metrics.
    """
    # Select model and parameter search spaces based on model_type
    if model_type == "SVM":
        base_model = SVC(random_state=seed)
        param_dist_1 = {
            'C': loguniform(1e-3, 1e3),
            'gamma': loguniform(1e-4, 1e1),
            'kernel': ['rbf', 'linear']
        }
        int_params = []
    elif model_type == "DecisionTree":
        base_model = DecisionTreeClassifier(random_state=seed)
        param_dist_1 = {
            'max_depth': [None] + list(np.arange(3, 20)),
            'min_samples_split': np.arange(2, 20),
            'min_samples_leaf': np.arange(1, 20),
            'max_features': ['auto', 'sqrt', 'log2', None],
            'criterion': ['gini', 'entropy']
        }
        int_params = ['max_depth', 'min_samples_split']
        
    elif model_type == "RandomForest":
        base_model = RandomForestClassifier(random_state=seed)
        param_dist_1 = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None] + list(np.arange(3, 20)),
            'min_samples_split': np.arange(2, 20),
            'min_samples_leaf': np.arange(1, 10),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
        }
        int_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']

    elif model_type == "LogisticRegression":
        base_model = LogisticRegression(random_state=seed, max_iter=5000)
        param_dist_1 = {
            'C': loguniform(1e-3, 1e3),
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga'],
            'multi_class': ['ovr', 'multinomial'],              
            'class_weight': [None, 'balanced'],                 
            'tol': loguniform(1e-5, 1e-2),                      
            'fit_intercept': [True, False]
        }
        int_params = []
    elif model_type == "XGBoost":
        if use_GPU:
            base_model = XGBClassifier(random_state=seed, use_label_encoder=False, eval_metric='logloss', tree_method='hist', device='cuda')
        else:
            base_model = XGBClassifier(random_state=seed, use_label_encoder=False, eval_metric='logloss', tree_method='hist')
        param_dist_1 = {
            'n_estimators': [50, 100, 200],
            'max_depth': np.arange(3, 10),
            'learning_rate': loguniform(1e-3, 1e-1),
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]

        }
        int_params = ['n_estimators', 'max_depth']

    else:
        raise ValueError("Unsupported model type. Choose from 'SVM', 'DecisionTree', 'RandomForest', 'LogisticRegression', or 'XGBoost'.")
    
    # Phase 1: Random Search
    random_search_1 = RandomizedSearchCV(base_model, param_distributions=param_dist_1,
                                         n_iter=20, cv=3, random_state=seed, n_jobs=-1)
    random_search_1.fit(X_train, y_train)
    best_params_1 = random_search_1.best_params_
    
    print('Random finished')

    # Phase 2: Grid Search
    sorted_results = sorted(random_search_1.cv_results_['params'], key=lambda p: p.get('mean_test_score', 0), reverse=True)
    top_params = [p for p in sorted_results[:3]]    
    bounded_params = ['subsample', 'colsample_bytree']
    grid_params = {}
    for key, value in best_params_1.items():
        if type(value) is bool:
            grid_params[key] = [value]
        elif isinstance(value, (int, float)):
            top_vals = [p[key] for p in top_params if key in p and p[key] != value]
            new_lower, new_upper = refine_bounds(value, top_vals)
            grid_range = np.linspace(new_lower, new_upper, num=3).tolist()  # 3 values between new bounds
            if key in int_params:
                grid_range = [int(round(x)) for x in grid_range]
            if key in bounded_params:
                grid_range = [min(x, 1.0) for x in grid_range]
            grid_params[key] = grid_range
        else:
            grid_params[key] = [value]
            
    grid_search_2 = GridSearchCV(base_model, param_grid=grid_params, cv=5, n_jobs=-1)
    grid_search_2.fit(X_train, y_train)
    best_params_2 = grid_search_2.best_params_
    best_params_final = grid_search_2.best_params_
    best_model = grid_search_2.best_estimator_
    print('Grid finished')

    # Phase 3: Build best model
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    report = classification_report(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_filename = f"{variant_name}_{model_type}_confusion_matrix.png"
    plt.savefig(cm_filename)
    plt.show()  
    plt.close()
    
    tracemalloc.start()
    memory_before = get_memory_usage()
    inf_time = get_inference_time(best_model, X_test)
    memory_after = get_memory_usage()
    tracemalloc.stop()
    cpu_usage = get_cpu_usage()
    memory_used = memory_after - memory_before
    
    performance_metrics = {
        "inference_time_ms": inf_time,
        "memory_used_mb": memory_used,
        "cpu_usage_percent": cpu_usage,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": acc
    }
    

    report_filename = f"{variant_name}_{model_type}_classification_report.txt"
    with open(report_filename, "w") as f:
        f.write(report)
    
    
    # with mlflow.start_run(run_id = run_id):
    mlflow.log_param("variant", variant_name)
    mlflow.log_param("model_type", model_type)
    mlflow.log_params(best_params_final)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("inference_time_ms", inf_time)
    mlflow.log_metric("memory_used_mb", memory_used)
    mlflow.log_metric("cpu_usage_percent", cpu_usage)
    mlflow.log_artifact(report_filename)
    mlflow.log_artifact(cm_filename)
    mlflow.set_tag("developer", f"Model {variant_name} {model_type}")
    mlflow.sklearn.log_model(best_model, "model", input_example=X_train.iloc[[0]])

    print(f"{variant_name} {model_type} best parameters: {best_params_final}")
    print(f"{variant_name} {model_type} test accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    print(f"Inference time (ms): {inf_time:.2f}")
    print(f"Memory used (MB): {memory_used:.2f}")
    print(f"CPU usage (%): {cpu_usage:.2f}")
    print(report)
    
    return best_model, best_params_final, acc, performance_metrics


def main(variant_name, model_type, model_name="logistic"):
    start_time = time.time()

    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    ### Set the experiment name
    mlflow.set_experiment("Hand-Gesture-Models")

    run = mlflow.start_run(run_name=f"{variant_name}_{model_type}_run")
    run_id = run.info.run_id

    df = pd.read_csv("Data/hand_landmarks_data.csv")
    df = df.drop(['z'+str(x) for x in range(1,22)], axis=1)

    le = LabelEncoder()
    labels = le.fit_transform(df['label'])
    df = df.drop(columns=['label'])
    
    X_train, X_test, y_train, y_test, to_drop = preprocess(df, labels)

    model, params, acc, perf = tune_model_with_performance(X_train, X_test, y_train, y_test, variant_name=variant_name, model_type=model_name, run_id=run_id)
    end_time = time.time()
    training_time = end_time - start_time

    label_enc_path = "outputs/label_encoder.pkl"
    if not os.path.exists(label_enc_path):
        joblib.dump(le, label_enc_path)
    mlflow.log_artifact(label_enc_path, artifact_path="label_enc")

    path = 'outputs/'+model_name+'/'+ model_type + '/'
    save_tuning_outputs(model, params, acc, perf ,path)

    if to_drop:
        feature_drop_path = f"outputs/{model_name}/{model_type}/features_to_drop.pkl"
        joblib.dump(to_drop, feature_drop_path)
        mlflow.log_artifact(feature_drop_path, artifact_path="features_to_drop")

    mlflow.log_metric("training_time_sec", training_time)
    mlflow.end_run()


if __name__ == "__main__":
    main(variant_name = 'Preprocessed_Variant', model_type = 'preprocessed', model_name='DecisionTree')