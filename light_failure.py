import pandas as pd
import os
import numpy as np
import datetime as dt
#from datetime import datetime as dt
#from datetime import datetime?
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load, Parallel, delayed
from sklearn.metrics import  confusion_matrix, accuracy_score, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sqlalchemy import create_engine, text
import json
import time
import logging

import sys
import os
sys.path.append('../config')
from config_api import get_db_uri

# from query_api import streetlight_failure_prediction_api

def random_sample(array, fraction=0.1, random_state=None):
    """
    Randomly sample a fraction of rows from a NumPy array.
    
    Parameters:
    -----------
    array : numpy.ndarray
        Input array to sample from
    fraction : float, optional (default=0.1)
        Fraction of rows to sample (between 0 and 1)
    random_state : int, optional (default=None)
        Random seed for reproducibility
        
    Returns:
    --------
    numpy.ndarray
        Sampled subset of the input array
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = int(len(array) * fraction)
    random_indices = np.random.choice(len(array), size=n_samples, replace=False)
    return array[random_indices]

def generate_data(data,
                 client_type,
                 bool_columns = [],
                 output_forward_period=1,
                 input_lookback_period=3,
                 input_lookback_split=3):

    # Check if the input data is empty
    if data.empty:
        return pd.DataFrame()

    # Convert client type to lowercase for consistency
    client_type = client_type.lower()
    if client_type not in ['demo', 'qbit']:
        raise ValueError("client_type must be either 'demo' or 'qbit'")

    # Ensure 'eventTime' column is in datetime format
    data['eventTime'] = pd.to_datetime(data['eventTime'])

    # Sort the data by 'eventTime' and create a copy
    data = data.sort_values('eventTime').copy()
    
    # Extract the device ID
    device = data.device.unique()[0]

    # Set time boundaries for the loop
    data_min_eventTime = data['eventTime'].min()
    data_max_eventTime = data['eventTime'].max()
    loop_start_time = data_min_eventTime + dt.timedelta(hours=input_lookback_period)
    loop_end_time = data_max_eventTime - dt.timedelta(hours=output_forward_period)

    # Time step calculation
    time_step = dt.timedelta(hours=input_lookback_period / input_lookback_split)
    
    results = []
    
    if client_type == 'demo':
        # Original logic for demo client
        for cur_eventTime in data['eventTime'][(data['eventTime'] >= loop_start_time) & 
                                             (data['eventTime'] <= loop_end_time)]:
            input_data = []
            
            # Generate input data for the lookback periods
            for period in range(1, input_lookback_split + 1):
                query_start_time = cur_eventTime - dt.timedelta(hours=input_lookback_period) + (period - 1) * time_step
                query_end_time = query_start_time + time_step
                input_selected_data = data[(data['eventTime'] > query_start_time) & 
                                        (data['eventTime'] <= query_end_time)]
                t1 = input_selected_data['meaning'].value_counts()
                t1.index = t1.index.map(lambda x: f"input_prd{period}_" + str(x))
                input_data.append(t1)

            # Generate output data for the forward period
            output_selected_data = data[(data['eventTime'] > cur_eventTime) & 
                                     (data['eventTime'] <= cur_eventTime + dt.timedelta(hours=output_forward_period))]
            t2 = output_selected_data['meaning'].value_counts()
            t2.index = t2.index.map(lambda x: "output_" + str(x))

            # Combine input and output data
            merged = pd.concat(input_data + [t2])
            merged['eventTime'] = cur_eventTime
            results.append(merged)

    else:  # qbit client
        # # Get columns that contain only 0s and 1s for output
        # bool_columns = []
        # for col in data.columns:
        #     if col not in ['eventTime', 'device']:
        #         unique_vals = data[col].unique()
        #         # Check if column contains only 0s and 1s (can be int, float, or bool)
        #         if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        #             bool_columns.append(col)
        ## TODO: for now, the logic above will sometimes mis-label mode column, we hard code the bool columns here
    
        # Get all columns except eventTime and device for input
        input_columns = [col for col in data.columns if col not in ['eventTime', 'device']]
        
        for cur_eventTime in data['eventTime'][(data['eventTime'] >= loop_start_time) & 
                                             (data['eventTime'] <= loop_end_time)]:
            period_data = []
            
            # Generate input data for the lookback periods (be careful about look-ahead bias)
            for period in range(1, input_lookback_split + 1):
                query_start_time = cur_eventTime - dt.timedelta(hours=input_lookback_period) + (period - 1) * time_step
                query_end_time = query_start_time + time_step
                
                input_selected_data = data[(data['eventTime'] > query_start_time) & 
                                        (data['eventTime'] <= query_end_time)]
                
                if not input_selected_data.empty:
                    # Set up aggregation dictionary for different column types
                    agg_dict = {}
                    for col in input_columns:
                        if col in bool_columns:
                            # For boolean columns: return 1 if any 1 exists, 0 otherwise
                            agg_dict[col] = lambda x: 1 if 1 in x.values else 0
                        else:
                            # For numeric columns: use mean
                            agg_dict[col] = 'mean'
                    
                    period_stats = input_selected_data.agg(agg_dict)
                    period_stats.index = period_stats.index.map(lambda x: f"input_prd{period}_" + str(x))
                    period_data.append(period_stats)
                else:
                    # Create empty series with correct index
                    empty_stats = pd.Series(index=[f"input_prd{period}_" + str(x) for x in input_columns])
                    period_data.append(empty_stats)

            # Generate output data for the forward period
            output_selected_data = data[(data['eventTime'] > cur_eventTime) & 
                                     (data['eventTime'] <= cur_eventTime + dt.timedelta(hours=output_forward_period))]
            
            if not output_selected_data.empty:
                # Only include boolean columns in output
                output_stats = output_selected_data[bool_columns].agg(
                    lambda x: 1 if 1 in x.values else 0
                )
                output_stats.index = output_stats.index.map(lambda x: "output_" + str(x))
                
                # Combine input and output data
                merged = pd.concat(period_data + [output_stats])
                merged['eventTime'] = cur_eventTime
                results.append(merged)

    # Combine all results into a DataFrame
    if results:
        final_data = pd.concat(results, axis=1).T
        #final_data = final_data.fillna(0)
        final_data = final_data.infer_objects(copy=False).fillna(0) #changed because pandas will stop silently downcasting object dtype columns
        final_data['device'] = device
        final_data = final_data.set_index(['device', 'eventTime'])
    else:
        final_data = pd.DataFrame()

    return final_data

def get_latest_events(df):
    """
    Select rows with the maximum eventTime for each device.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with columns 'device', 'eventTime', and 'prediction'
    
    Returns:
    pandas.DataFrame: Filtered DataFrame containing only the latest event for each device
    """
    # Sort by device and eventTime to ensure consistent results
    df_sorted = df.sort_values(['device', 'eventTime'])
    
    # Get the rows with maximum eventTime for each device
    latest_events = df.loc[df.groupby('device')['eventTime'].idxmax()]
    
    return latest_events

def applyParallel(dfGrouped,
                  func,
                  kwargs,
                  n_jobs=16):
    ''' Apply a specific function on groups of data, using parallel computing to speed up

    :param dfGrouped: dataframe, contains the log for each device
    :param func: float, number of hours, control how far we look back to collect behavior
    :param kwargs: dictionary, which is the input for the function
    :param n_jobs: int, number of cpu for parallel computing, -1 to use all
    :return: dataframe
    '''
    retLst = Parallel(n_jobs=n_jobs)(delayed(func)(group, **kwargs) for name, group in dfGrouped)
    return pd.concat(retLst)


def model_optimization_perf(kwargs, X_train, y_train, X_test, y_test, input_name_list = None):
    """ Model optimization using Grid CV + Model perf on train and test data sets
    :param kwargs: model kwargs
    :return: grid cv model object, dictionary of model performance
    """
    # Load Model
    if os.path.exists(kwargs['model_path']) and not kwargs['force_rerun']:
        print(f"Model exists at {kwargs['model_path']}! Loading it.")
        gridcv_model = load(kwargs['model_path'])
    else:
        model = kwargs['model_class'](**kwargs['model_init_kwargs'])
        gridcv_model = GridSearchCV(
            model,
            **kwargs['grid_cv_kwargs'])
        gridcv_model.fit(X_train, y_train)
        if kwargs['save_model']:
            dump(gridcv_model, kwargs['model_path'])
            dump(input_name_list, kwargs['info_path'])

    # Model Performance
    y_pred = gridcv_model.best_estimator_.predict(X_test)
    y_pred_tr = gridcv_model.best_estimator_.predict(X_train)
    perf = {
        'test': {
            'accuracy_score': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc_score': roc_auc_score(y_test, y_pred),
            'recall_score': recall_score(y_test, y_pred),
        },
        'train': {
            'accuracy_score': accuracy_score(y_train, y_pred_tr),
            'confusion_matrix': confusion_matrix(y_train, y_pred_tr),
            'roc_auc_score': roc_auc_score(y_train, y_pred_tr),
            'recall_score': recall_score(y_train, y_pred_tr),
        }
    }

    return gridcv_model, perf

def execute_query(query, database=None, parameters=None):
    """
    Execute an SQL query and return the results in a list of dictionaries.
    
    Parameters:
    - query (str): SQL query to be executed.
    - parameters (dict, optional): Parameters for prepared statements.

    Returns:
    - list of dicts: Results of the query, each row as a dictionary.
    """
    connection_uri = get_db_uri(database)  # Get URI from config
    engine = create_engine(connection_uri)

    with engine.connect() as connection:
        result = connection.execute(text(query), parameters) if parameters else connection.execute(text(query))
        # Use the ._asdict() method for RowProxy objects to convert to dictionaries
        result_list = [row._asdict() for row in result]
        return result_list

def data_processing(input_data, 
                    forward_period, 
                    target_failure_meaning, # meaning integer or string type column name
                    input_bool, # If make inputs columns bool format
                    drop_origin_input, # If we drop original columns after we create bool version columns
                    max_min_scaler, # If we want to apply max_min_scaler on input data
                    drop_target_failure_meaning_col, # If we want to drop the historical target meaning/failure columns from input data
                    feature_engineering, # If we want to implement feature engineering
                    client_type, # type of client, e.g. demo, qbit
                    need_y = True, # if we need y values and put it in the final output
                    bool_columns = [] # list of bool columns, provided by users
                    ):

    kwargs = {
        'client_type': client_type,
        'bool_columns': bool_columns,
        'input_lookback_period':forward_period * 3, 
        'output_forward_period':forward_period, 
        'input_lookback_split':3
    }

    final_data = applyParallel(input_data.groupby('device'), 
                            generate_data, 
                            kwargs).fillna(0)
    
    if client_type.lower() == 'demo':
        # Drop unrelated columns and only keep the key columns for prediction
        failure_meanings = [337, 2675, 1639, 2346, 6161, 6686, 6833, 7582, 9451]
        failure_prediction_meanings = [2822, 246, 435, 624]
        related_columns = [x for x in final_data.columns if
                            any([str(m) in x for m in failure_meanings + failure_prediction_meanings])]
        final_data = final_data[related_columns]

    elif client_type.lower() == 'qbit':
        pass

    ######################### This part can be moved to live ETL ################################
    ## Data Manipulation
    if input_bool:
        if bool_columns is None or len(bool_columns) == 0:
            # turn all input columns into bool like if bool_columns not provided
            for column_name in [c for c in final_data.columns if 'input' in c]:
                final_data[f'bool_{column_name}'] = final_data[column_name].apply(lambda x: 1 if x else 0)
            if drop_origin_input:
                final_data.drop([c for c in final_data.columns if 'input' in c and 'bool' not in c], axis=1, inplace=True)
        else:
            # For now, if bool columns are provided, then there is no need to turn more columns into bool like columns
            pass
            
    if drop_target_failure_meaning_col:
        final_data.drop([c for c in final_data.columns if str(target_failure_meaning) in c and 'input' in c], axis=1,
                        inplace=True)

    if feature_engineering:
        if bool_columns is None or len(bool_columns) == 0:
            bool_input = sorted([c for c in final_data.columns if 'input' in c and 'bool' in c])
            non_bool_input = sorted([c for c in final_data.columns if 'input' in c and 'bool' not in c])
        else:
            bool_input = [c for c in final_data.columns if any([y in c for y in bool_columns])]
            non_bool_input = [c for c in final_data.columns if c not in bool_input]
            bool_input = [x for x in bool_input if 'output' not in x]
            non_bool_input = [x for x in non_bool_input if 'output' not in x]
        if len(bool_input):  # if there is no bool columns, we will not apply this, same for non-bool columns
            for i, a in enumerate(bool_input):
                for j, b in enumerate(bool_input):
                    if i > j:
                        final_data[f'{a}_{b}_inter'] = final_data.eval(f"{a} * {b}")

        # This may generate null values
        if len(non_bool_input):
            for i, a in enumerate(non_bool_input):
                for j, b in enumerate(non_bool_input):
                    if i > j:
                        final_data[f'{a}_{b}_imb'] = final_data.eval(f'({a}-{b})/({a}+{b})')

        # TODO: add hours in input data?

    final_data = final_data.replace([np.inf, -np.inf], np.nan)

    if need_y:
        final_data['y'] = final_data[f'output_{target_failure_meaning}'].apply(lambda x: 1 if x else 0)
        X_df, y_df = final_data.filter(like = 'input', axis = 1), final_data['y']
        X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_df, y_df, test_size=0.2)

        y_dist = 100 * final_data['y'].value_counts()/final_data['y'].value_counts().sum()
        mm_ratio = max(y_dist)/min(y_dist)

        # TODO: store the scaler as joblib object
        if max_min_scaler:
            scaler = MinMaxScaler()
            cols = [x for x in X_train_df.columns if 'bool' not in x]
            if len(cols):
                X_train_df[cols] = scaler.fit_transform(X_train_df[cols])
                X_test_df[cols] = scaler.transform(X_test_df[cols])

        # Please note:we did not remove nan values, which may cause error later
        X_train, X_test, y_train, y_test = X_train_df.values, X_test_df.values, y_train_df.values, y_test_df.values

        return X_df, y_df, X_train, X_test, y_train, y_test, mm_ratio, final_data
    else:
        X_df = final_data.filter(like = 'input', axis = 1)
        return X_df, final_data

# Mapping logic needs to be updated
# Source: https://nexuscomputing.atlassian.net/wiki/spaces/~5570583346ae8900d14b19b9c6c1d3ee579b63/pages/41549826/Failures+and+Mapping
mapping = {
    'driverFault': 2675,
    'highPower': 6161,
    'lowPower': 6161,
    'highPF': 6833,
    'lowPF': 6833,
    'lamp_failed': 7582,
    'powerOutage': 9451,
    'relaySticking': 337,
    'relayOpenCircuit': 337,
    'relay_adhesion': 337,
    'drive_communication_error': 6686,

    # The following comes from table error_internal
    # Need to check with team, more to be added
    'lightFault': 7582,
    'flickering': 101,
    'powerOutage': 6161,
}

def data_cleaning(data):
    data['meaning'] = data['controllerfault'].map(mapping)
    data = data.dropna() # Drop unmapped records
    data['meaning'] = data.meaning.astype(int)

    # Data Cleaning
    data.rename(columns = {'deviceid':'device'}, inplace=True)
    data['eventTime'] = pd.to_datetime(data['eventtime'], unit='ms')

    return data

def filter_and_add_columns(df, column_list):
    # Remove columns not in the list
    df = df[[col for col in df.columns if col in column_list]]
    
    # Add new columns with zeros if they're in the list but not in the DataFrame
    for col in column_list:
        if col not in df.columns:
            df[col] = 0
    
    # Ensure the DataFrame only has columns from the list, in the order of the list
    df = df.reindex(columns=column_list, fill_value=0)
    
    return df

def load_qbit_data(extra_filter = '', time_column='eventTime'):

    # Column name mapping, provided by Qbit; columns commented out are columns that are not viz for us
    analog_column_rename_mapping = {
        "Tag1": "Voltage",
        "Tag2": "Current",
        "Tag3": "Watts",
        "Tag4": "Cumulative Kilowatt Hrs",
        "Tag5": "Burn Hrs",
        "Tag6": "Dimming",
        "Tag7": "Power Factor",
        "Tag8": "Lamp Steady Wattage",
        "Tag9": "Mode",
        # "Tag10": "Temperature"
    }

    digital_column_rename_mapping = {
        "Tag1": "Lamp Status",
        "Tag2": "Photocell Status",
        "Tag3": "Voltage Under Over",
        "Tag4": "Lamp Condition",
        # "Tag5": "Photocell Oscillating",
        "Tag6": "Photocell Feedback",
        # "Tag7": "Lamp Cyclic",
        "Tag8": "Communication",
        "Tag9": "Driver",
        # "Tag10": "Abnormal Lamp Condition",
        "Tag11": "Real Time Clock Status",
        # "Tag12": "Event Over Flow",
        "Tag13": "Energy Meter Fault",
        "Tag14": "Relay Weld",
        "Tag15": "Tilt",
        "Tag16": "Day Burning",
        # "Tag17": "Pole Fault",
        # "Tag18": "Photocell Fault",
        # "Tag19": "Controller Fault"
    }
    
    # Load data from database. We uploaded data provided by Qbit.
    analog = pd.DataFrame(execute_query(
        f"select STR_TO_DATE(a.DateTimeField, '%c/%e/%y %H:%i') as {time_column}, a.*  from analog a {extra_filter}",
        database="test",  # This will use QBIT_DEFAULT_DB if not specified
        db_type="qbit"
    ))
    
    digital = pd.DataFrame(execute_query(
        f"select STR_TO_DATE(d.DateTimeField, '%c/%e/%y %H:%i') as {time_column}, d.* from digital d {extra_filter}",
        database="test",  # This will use QBIT_DEFAULT_DB if not specified
        db_type="qbit"
    ))

    # Rename columns and Drop duplicates
    analog_clean = analog.rename(columns = analog_column_rename_mapping)[[time_column, 'RTUNumber'] + list(analog_column_rename_mapping.values()) + ['SLCName']]
    digital_clean = digital.rename(columns = digital_column_rename_mapping)[[time_column, 'RTUNumber'] + list(digital_column_rename_mapping.values())]
    analog_clean_unique = analog_clean.drop_duplicates(subset=[time_column, 'RTUNumber'])
    digital_clean_unique = digital_clean.drop_duplicates(subset=[time_column, 'RTUNumber'])

    # Merge analog & digital
    merged = pd.merge(
        analog_clean_unique, 
        digital_clean_unique, 
        on=[time_column, 'RTUNumber'], 
        how='outer'
    )

    # Data cleaning
    merged[time_column] = pd.to_datetime(merged[time_column])
    merged.rename(columns = {'RTUNumber':'device'}, inplace=True)
    merged = merged.drop('SLCName', axis=1)

    # Flip communication column and rename it
    merged['Communication'] = 1 - merged['Communication']
    merged = merged.rename(columns={'Communication': 'Communication Fault'})

    # Columns names cleaning
    merged.columns = merged.columns.str.replace(' ', '_')

    # Create extra columns for feature engineering
    merged = merged.sort_values(['device', 'eventTime'])
    merged['actual_energy_consumption'] = merged.groupby('device')['Cumulative_Kilowatt_Hrs'].diff().shift(-1)
    merged['actual_burn_hrs'] = merged.groupby('device')['Burn_Hrs'].diff().shift(-1)
    merged['time_diff_minutes'] = merged.groupby('device')['eventTime'].diff().shift(-1).dt.total_seconds() / 60
    merged['actual_hourly_energy_consumption'] = merged.apply(
        lambda row: row['actual_energy_consumption'] * (60 / row['time_diff_minutes'])
        if pd.notnull(row['time_diff_minutes']) and row['time_diff_minutes'] > 0
        else pd.NA,
        axis=1
    )
    merged['actual_hourly_energy_consumption'] = merged['actual_hourly_energy_consumption'].replace([np.inf, -np.inf], pd.NA)
    
    # Only select photocell Mode data
    merged = merged.query("Mode == 1")
    
    # There are very very few records where light is 1 and it should not be on, remove it
    merged.query("Mode == 1").query("Photocell_Feedback != 1 or Lamp_Status != 1")

    # ourage: should light but not
    merged['should_on'] = 0
    merged.loc[merged['Photocell_Feedback'] == 0, 'should_on'] = 1
    merged.loc[merged['Photocell_Feedback'] == 1, 'should_on'] = 0
    merged['outage'] = ((merged['should_on'] == 1) & (merged['Lamp_Status'] == 0)).astype(int)
    
    # rf issue column
    merged['rf_issue'] = ((merged['Lamp_Status'] == 1) & (merged['Power_Factor'] < 0.9)).astype(int)

    # We drop actual_energy_consumption here, because in the feature engineering step, this column
    # together with actual_hourly_energy_consumption column will create lots of inf values
#     merged = merged.drop(['time_diff_minutes','actual_energy_consumption','Photocell_Feedback'],axis=1)
    merged = merged.drop(['time_diff_minutes','actual_energy_consumption','should_on','Photocell_Feedback'],axis=1)

    return merged


def streetlight_failure_predict(
    forward_period=2,
    device_list=None,
    target_failure_meaning=7582,  # by default, this is 7582 for demo; for qbit, you can add the string type of failure
    input_bool=True,
    drop_origin_input=True,
    max_min_scaler=False,
    drop_target_failure_meaning_col=True,
    feature_engineering=True,
    client = 'demo',
    sample_data_pct = None
):
    model_path = f'Model/streetlight_failure_pred_xgb_{forward_period}h_{target_failure_meaning}.joblib'
    if not os.path.exists(model_path):
        logging.warning(f"Model not found at {model_path}")
    info_path = f'Model/streetlight_failure_pred_xgb_{forward_period}h_{target_failure_meaning}_input_name_list.joblib'
    model_perf_path = f'Model/streetlight_failure_pred_xgb_{forward_period}h_{target_failure_meaning}_input_model_perf.joblib'
    bool_columns = []
    
    # Check if a model exists; if not, trigger on-the-fly training
    if not os.path.exists(model_path):
        logging.warning("No preexisting model found. Initiating on-the-fly training.")
        
        # Load data
        if client is None or client.lower() == 'demo':
            logging.warning("Loading demo data!")
            query = """
                SELECT e.*
                FROM error_internal e
            """
            data = pd.DataFrame(execute_query(query))
            data = data_cleaning(data)

            # Generate additional training data to augment small datasets (for demo only)
            combined_df = data.copy()
            for hours in range(1, 25):
                data_copy = data.copy()
                data_copy['eventTime'] = data_copy['eventTime'] - pd.DateOffset(hours=hours)
                combined_df = pd.concat([combined_df, data_copy])

        # if we are testing on qbit data, for now we will use a seperate function to load data & cleaning
        elif client.lower() == 'qbit':
            logging.warning("Loading Qbit data!")
            data = load_qbit_data()
            logging.warning("Qbit data loading complete.")
            bool_columns = ['Lamp Status',
                            'Photocell Status',
                            'Voltage Under Over',
                            'Lamp Condition',
                            'Communication Fault',
                            'Driver',
                            'Real Time Clock Status',
                            'Energy Meter Fault',
                            'Relay Weld',
                            'Tilt',
                            'Day Burning']
            bool_columns = [x.replace(' ','_') for x in bool_columns]
            # TODO: change this back!
            combined_df = data

        # Prepare the data for model training

        # Data cleaning before model training
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.dropna(axis=0, how='any')

        X_df, y_df, X_train, X_test, y_train, y_test, mm_ratio, final_data = data_processing(
            combined_df, forward_period, target_failure_meaning, input_bool, 
            drop_origin_input, max_min_scaler, drop_target_failure_meaning_col, 
            feature_engineering, client, True, bool_columns
        )

        # Define the training parameters for the on-the-fly model
        xgb_kwargs = {
            'name': 'XGBoost',
            'model_class': XGBClassifier,
            'model_init_kwargs': {
                'n_estimators': 200,
                'random_state': 42,
                'n_jobs': 16,
            },
            'grid_cv_kwargs': {
                'param_grid': {"scale_pos_weight": [x for x in range(max(1, int(mm_ratio)-5), int(mm_ratio) + 5)] + [1, 10, 50, 100]},
                'scoring': "f1",
                'cv': 5,
                'n_jobs': 16,
                'refit': True
            },
            'save_model': True,
            'force_rerun': True,
            'model_path': model_path,
            'info_path': info_path
        }

        # Train the model and save it
        if sample_data_pct != None:
            print(f"Using fraction {sample_data_pct*100}% data to train!")
            X_train_sample = random_sample(X_train, fraction=sample_data_pct, random_state=42)
            y_train_sample = random_sample(y_train, fraction=sample_data_pct, random_state=42)
            X_test_sample = random_sample(X_test, fraction=sample_data_pct, random_state=42)
            y_test_sample = random_sample(y_test, fraction=sample_data_pct, random_state=42)
            _, model_perf = model_optimization_perf(xgb_kwargs, X_train_sample, y_train_sample, X_test_sample, y_test_sample, input_name_list=list(X_df.columns))
        else:
            _, model_perf = model_optimization_perf(xgb_kwargs, X_train, y_train, X_test, y_test, input_name_list=list(X_df.columns))
        print("Model trained on-the-fly and saved.")
        print("Model performance:")
        print(model_perf)
        dump(model_perf, model_perf_path)
    else:
        print("Loading preexisting model.")

    # Load the trained model (either preexisting or newly trained on-the-fly)
    gridcv_model = load(model_path)
    model_input_name_list = load(info_path)

    # Inference: Load inference data and process it
    device_filter = ""
    if client is None or client.lower() == 'demo':
        if device_list is None or device_list == []:
            pass
        else:
            device_filter = "and deviceid in ('"+ "','".join(device_list)+"')"

        data_inference = pd.DataFrame(execute_query(f"""
            SELECT e.*
            FROM error_internal e where eventTime >= 1640995200000 {device_filter}
        """))
        data_inference = data_cleaning(data_inference)
    elif client.lower() == 'qbit':
        bool_columns = ['Lamp Status',
                        'Photocell Status',
                        'Voltage Under Over',
                        'Lamp Condition',
                        'Communication Fault',
                        'Driver',
                        'Real Time Clock Status',
                        'Energy Meter Fault',
                        'Relay Weld',
                        'Tilt',
                        'Day Burning']
        bool_columns = [x.replace(' ','_') for x in bool_columns]

        device_filter = ""
        if device_list is None or device_list == []:
            pass
        else:
            device_filter = "where RTUNumber in ("+ ",".join([str(x) for x in device_list])+")"

        data_inference = load_qbit_data(device_filter)

    X_df, final_data = data_processing(
        data_inference, forward_period, target_failure_meaning, input_bool, 
        drop_origin_input, max_min_scaler, drop_target_failure_meaning_col, 
        feature_engineering, client, False, bool_columns
    )

    # Predict using the loaded model
    y_pred = gridcv_model.best_estimator_.predict(filter_and_add_columns(X_df, model_input_name_list).values)
    output = pd.DataFrame(y_pred, index=X_df.index)
    output.columns = ['prediction']
    out = get_latest_events(output.reset_index()) # Select the max date for each device, this logic is only for demo use! TODO: with real data, this logic needs to be updated
    out = out.drop('eventTime', axis=1)
    out['eventTime'] = dt.datetime.today() 

    # Add device_id to the output to avoid KeyError in handle_failure_results
    out = out.rename(columns={'device': 'device_id'})
    out['geozone'] = np.random.choice(['Zone1', 'Zone2', 'Zone3'], size=len(out))  # Example geozones
    out['group'] = np.random.choice(['10087', '12345', '98765'], size=len(out))
    out['streetName'] = np.random.choice(['A street', 'B avenue', 'C road'], size=len(out))
    out_dict = out.to_dict(orient='records')

    return out_dict

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict streetlight failure.')
    parser.add_argument('--forward_period', type=int, default=1,
                        help='The forward period for prediction (default: 1)')
    
    args = parser.parse_args()

    # Example of demo
    print(streetlight_failure_predict(forward_period=12,
                                      device_list=['100880'])) # Need a list of string for device Id, if not provided or [] is provided, then all device ID will be checked
    
    # Example of qbit data
    print(streetlight_failure_predict(
                forward_period=1,
                device_list=[2001, 2003],
                target_failure_meaning='Lamp_Condition',  # by default, this is 7582 for demo; for qbit, you can add the string type of failure
                input_bool=False,
                drop_origin_input=False,
                max_min_scaler=False,
                drop_target_failure_meaning_col=True,
                feature_engineering=True,
                client = 'qbit',
                sample_data_pct=0.1))