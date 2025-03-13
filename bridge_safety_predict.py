import pandas as pd
import numpy as np
from typing import Optional, Union, List

import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
import optuna

# Suppress Optuna logs except for errors
optuna.logging.set_verbosity(optuna.logging.ERROR)

import logging

# Configure logging to show warnings and higher level logs
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Feature engineering: split columns like max_g into multiple columns
def split_into_columns(df, column_name):
    # remove the trailing comma, remove space, then split by comma
    split_columns = df[column_name].str.replace(' ','').str.rstrip(',').str.split(',', expand=True)
    split_columns = split_columns.astype(float)
    split_columns.columns = [f'{column_name}_{i+1}' for i in range(split_columns.shape[1])]
    df = pd.concat([df, split_columns],axis=1)
    return df

# Main Function of safety prediction
def process_and_train_model(input_df, device_id, parameter_finetune='bayesian', log_verbose=True, 
                            accuracy_verbose=True, feature_importance_verbose=True, 
                            positive_weight_lower_bound=0.05, positive_weight_upper_bound=20):

    # Time-related feature engineering
    input_df['time'] = pd.to_datetime(input_df['time'])
    tmp = input_df[input_df['device_id'] == device_id].copy()
    tmp.index = np.arange(len(tmp))
    tmp['hours'] = tmp['time'].dt.hour
    tmp['weekday'] = tmp['time'].dt.weekday  # Monday=0, Sunday=6
    tmp['month'] = tmp['time'].dt.month

    # Time-related features encoding
    tmp['hour_bucket'] = pd.cut(tmp['hours'], bins=[-1, 5, 11, 17, 23], labels=[1, 2, 3, 4])
    encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' to avoid multicollinearity

    # Encoding features
    for feature in ['weekday', 'month', 'hour_bucket']:
        encoded = encoder.fit_transform(tmp[[feature]])
        encoded_df = pd.DataFrame(encoded, columns=[f"{feature}_{int(i)}" for i in range(1, encoded.shape[1] + 1)])
        tmp = pd.concat([tmp, encoded_df], axis=1)

    # Intensity-related features engineering
    for column in ['intensity', 'max_g', 'max_vibration_speed', 'max_movement']:
        tmp = split_into_columns(tmp, column)
    
    tmp.drop(columns=['weekday', 'month', 'hours', 'intensity', 'max_g', 'max_vibration_speed', 'max_movement'], inplace=True)

    # Output preparation
    tmp['safe_or_not'] = tmp['safety_score'].apply(lambda x: 0 if x in [1, 2] else 1)
    tmp['unsafe_type'] = tmp['safety_score'].apply(lambda x: 0 if x == 1 else 1)

    # Sort the input by time
    tmp = tmp.sort_values('time')

    items_to_remove = ['device_id', 'time', 'safety_score', 'hour_bucket', 'safe_or_not']

    def train_model(tmp, output_col, log_verbose):
        
        # The last record (lastest record) will be used as input for future prediction
        #### This part needs to be updated when connecting with customers ####
        y = tmp.iloc[:-1,:][output_col]
        X = tmp.iloc[:-1,:][[item for item in tmp.columns if item not in items_to_remove]]

        if log_verbose:
            print(f"Shape of X is {X.shape}")
            print(f"Shape of y is {y.shape}")
            print("The distribution of output is:")
            display(y.value_counts().reset_index())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=126)
        if log_verbose:
            print(f"Shape of X_train is {X_train.shape}")
            print(f"Shape of y_train is {y_train.shape}")
            print(f"Shape of X_test is {X_test.shape}")
            print(f"Shape of y_test is {y_test.shape}")

        # Calculate scale_pos_weight
        positive_weight = sum(y_train == 1) / sum(y_train == 0)
        if log_verbose:
            print(f"The adj factor for unbalanced dataset is: {positive_weight}")
        if (positive_weight < positive_weight_lower_bound or positive_weight > positive_weight_upper_bound) and log_verbose:
            print(f"WARNING: The input data is too unbalanced! With input ratio {positive_weight}. The model might not perform well.")

        def objective(trial):
            # Suggest hyperparameters to tune
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_loguniform('gamma', 0.1, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10),
                'scale_pos_weight': trial.suggest_loguniform('scale_pos_weight', min(1, positive_weight/2), positive_weight + 5)  # here, positive_weight will almost > 1
            }
            
            # Train the model with the suggested hyperparameters
            model = xgb.XGBClassifier(**param, use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)
            
            # Make predictions on the test set
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            return accuracy

        # Parameter tuning
        if parameter_finetune == 'bayesian':
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50)
            best_params = study.best_params
            model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)
            if log_verbose:
                print("Bayesian parameter finetune done!")
        else:
            model = xgb.XGBClassifier()
            model.fit(X_train, y_train)
            if log_verbose:
                print("No parameter finetune is used.")

        # Model evaluation
        if accuracy_verbose:
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy}")
            cm = confusion_matrix(y_test, y_pred)
            print("Confusion Matrix:")
            display(cm)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.show()
            print("Classification Report:")
            print(classification_report(y_test, y_pred))

        if feature_importance_verbose:
            try:
                feature_importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': feature_importances
                }).sort_values(by='Importance', ascending=False)
                print("Feature Importances:")
                display(importance_df)
                plt.figure(figsize=(10, 8))
                xgb.plot_importance(model, max_num_features=10)
                plt.title('Top 10 Feature Importances')
                plt.show()
            except Exception as e:
                print("Eorr occured for feature importance:", e)

        return model

    if log_verbose:
        print("Train first model: classify safe vs non-safe")
    model1 = train_model(tmp, 'safe_or_not', log_verbose)

    if len(tmp.query("safety_score != 3")['safety_score'].unique()) == 1:
        print("Only 1 type of error in the dataset, skip the second model.")
        model2 = None
    else:
        if log_verbose:
            print("Train second model: classify non-safe type")
        model2 = train_model(tmp.query("safety_score != 3"), 'unsafe_type', log_verbose)

    #### Prediction (For now, we assume the lastest record is input) ####
    #### This part needs to be updated when connecting with customers ####
    pred_input = tmp.sort_values('time').iloc[-1:,:]
    x = pred_input[[item for item in pred_input.columns if item not in items_to_remove]]

    model1_output = model1.predict(x)[0]
    if not model1_output:
        if not model2:
            print("Only 1 type of risk in the historical data, no able to predict the risk type.")
            model2_output = None
        else:
            model2_output = model2.predict(x)[0]

    if model1_output == 1:
        return "Safe"
    else:
        if not model2_output:
            return "Unsafe (Not able to tell risk type)"
        else:
            if model2_output == 0:
                return "警惕（建议检查此测点附近结构是否有外表破坏"
            else:
                return "较危险（建议尽快检查此测点附近结构破坏情况)"

def get_bridge_safety_prediction(device_ids: Union[str, List[str]], bridge: Optional[str] = None, time_frame: Optional[str] = None):
    ########################## This part needs to be replaced by RDS query ##########################
    # df = pd.read_csv(r"C:\Users\leela\Downloads\qiangzhen_data_sample.csv")
    df = pd.read_csv('Data/qiangzhen_data_sample.csv')
    df.columns = ['device_id','time','file_path','intensity','max_g','max_vibration_speed','max_movement','safety_score','file_path2']
    df = df.drop(['file_path2','file_path'], axis = 1)
    
    # Data Cleaning
    sorted_safety_score = sorted(df['safety_score'].unique())
    mapping = {
        sorted_safety_score[1]: 1, # type I risk, 警惕（建议检查此测点附近结构是否有外表破坏
        sorted_safety_score[2]: 2, # type II risk, 较危险（建议尽快检查此测点附近结构破坏情况）
        sorted_safety_score[0]: 3 # safe
    }
    df['safety_score'] = df['safety_score'].replace(mapping)
    
    # require that the device id needs to have enough sample size
    SAMPLE_SIZE_THRESHOLD = 100
    filtered_df = df.groupby('device_id').filter(lambda x: len(x) > SAMPLE_SIZE_THRESHOLD)
    
    # have all types of safety scores
    filtered_df_all = filtered_df.groupby('device_id').filter(lambda x: all(score in x['safety_score'].unique() for score in [1,2,3]))
    
    ######################################################################################################## 
    
    # Convert single device_id to list for consistent processing
    if isinstance(device_ids, str):
        device_ids = [device_ids]
    
    results = []
    for device_id in device_ids:
        # Check if the device_id exists in the filtered data
        if device_id not in filtered_df_all['device_id'].unique():
            results.append({"device_id": device_id, "error": f"No data available for device_id: {device_id}"})
            continue
        
        try:
            prediction = process_and_train_model(
                input_df=filtered_df_all,
                device_id=device_id, 
                parameter_finetune='bayesian', 
                log_verbose=False, 
                accuracy_verbose=False,
                feature_importance_verbose=False, 
                positive_weight_lower_bound=0.05, 
                positive_weight_upper_bound=20,
            )
            results.append({"device_id": device_id, "prediction": prediction})
        except Exception as e:
            results.append({"device_id": device_id, "error": str(e)})
    
    return results

if __name__ == '__main__':
    try:
        result = get_bridge_safety_prediction(device_ids='强震采集仪拷机_备货短周期_端口71                                                                                                                                                                        ')
        print(result)
    except ValueError as e:
        print(f"Error: {e}")