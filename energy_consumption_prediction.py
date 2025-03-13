import pandas as pd
import numpy as np
import statsmodels.api as sm
import pmdarima as pm
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sklearn.metrics import mean_squared_error
import logging

import sys
import os
sys.path.append('../config')
from config_api import get_db_uri

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def list_to_tuple_str(lst):
    s = str(tuple(lst)).replace('"', "'")
    return s[:-2] + ')' if len(lst) == 1 else s

def execute_query(query, database=None, parameters=None):
    """
    Execute an SQL query and return the results in a list of dictionaries.
    
    Parameters:
    - query (str): SQL query to be executed.
    - parameters (dict, optional): Parameters for prepared statements.

    Returns:
    - list of dicts: Results of the query, each row as a dictionary.
    """
    # Get connection URI using the energy database configuration
    connection_uri = get_db_uri(database, db_type="energy")
    engine = create_engine(connection_uri)

    with engine.connect() as connection:
        result = connection.execute(text(query), parameters) if parameters else connection.execute(text(query))
        # Use the ._asdict() method for RowProxy objects to convert to dictionaries
        result_list = [row._asdict() for row in result]
        return result_list

def evaluate_model(model_results, test_data):
    """Evaluate model performance using AIC and RMSE"""
    aic = model_results.aic
    predictions = model_results.get_prediction(start=len(test_data)-12)  # Last 12 periods
    rmse = np.sqrt(mean_squared_error(test_data[-12:], predictions.predicted_mean[-12:]))
    return aic, rmse

def arima_forecast(
    input_data,
    name,
    cadance,
    prediction_periods,
    time_col='timestamp',
    energy_col='energyconsumption'
):
    try:
        display_name = name
        logging.info(f"Starting ARIMA forecast for {display_name} with cadence '{cadance}' and {prediction_periods} periods")
        
        tmp = input_data.copy()
        tmp[time_col] = pd.to_datetime(tmp[time_col], unit='ms')
        
        agg_data = tmp.groupby(tmp[time_col].dt.strftime({
            'D': '%Y-%m-%d',
            'W': '%Y-%W',
            'M': '%Y-%m',
            'Y': '%Y'
        }.get(cadance, '%Y-%m-%d')))[energy_col].sum()
        
        agg_data = agg_data.sort_index()
        logging.info(f"Aggregated data: {len(agg_data)} periods of data available")
        
        if len(agg_data) < 2:
            logging.warning(f"Insufficient data for prediction. Need at least 2 periods, got {len(agg_data)}")
            dates = pd.date_range(start=tmp[time_col].max(), 
                                periods=prediction_periods+1, 
                                freq=cadance)[1:]
            return pd.DataFrame(0, index=[f'next {x+1} {cadance}' for x in range(prediction_periods)],
                              columns=[name])

        # Step 1: Try auto-detection of ARIMA orders
        try:
            logging.info("Attempting auto-detection of ARIMA orders")
            auto_arima = pm.auto_arima(
                agg_data,
                seasonal=True,
                m=12,
                start_p=0,
                start_q=0,
                max_p=2,
                max_q=2,
                max_d=1,
                start_P=0,
                start_Q=0,
                max_P=1,
                max_Q=1,
                max_D=1,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            
            results = auto_arima.fit(agg_data)
            pred = auto_arima.predict(n_periods=prediction_periods)
            logging.info(f"Auto ARIMA successful with orders {auto_arima.order}, seasonal_order {auto_arima.seasonal_order}")
            
            output_df = pd.DataFrame(pred)
            output_df.columns = [name]
            output_df.index = [f'next {x+1} {cadance}' for x in range(prediction_periods)]
            return output_df
            
        except Exception as auto_error:
            logging.warning(f"Auto ARIMA failed: {auto_error}")
            
            # Step 2: Try all seasonal configurations and select best performing
            seasonal_configs = [
                ((0,1,1), (0,1,1,12)),
                ((1,1,0), (1,1,0,12)),
                ((1,1,1), (1,1,1,12)),
                ((1,1,0), (0,1,1,12)),
                ((0,1,1), (1,1,0,12)),
                ((1,0,1), (1,1,0,12)),
                ((1,0,1), (0,1,1,12)),
                ((1,1,0), (1,0,1,12)),
                ((0,1,1), (1,0,1,12))
            ]
            
            successful_models = []
            
            for order, seasonal_order in seasonal_configs:
                try:
                    logging.info(f"Trying SARIMA with order {order}, seasonal_order {seasonal_order}")
                    model = sm.tsa.statespace.SARIMAX(
                        agg_data,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    
                    results = model.fit()
                    aic, rmse = evaluate_model(results, agg_data)
                    
                    successful_models.append({
                        'order': order,
                        'seasonal_order': seasonal_order,
                        'results': results,
                        'aic': aic,
                        'rmse': rmse
                    })
                    
                    logging.info(f"SARIMA config {order}, {seasonal_order} successful - AIC: {aic:.2f}, RMSE: {rmse:.2f}")
                    
                except Exception as config_error:
                    logging.warning(f"SARIMA config {order}, {seasonal_order} failed: {config_error}")
                    continue
            
            if successful_models:
                # Select best model based on AIC and RMSE
                best_model = min(successful_models, key=lambda x: (x['aic'], x['rmse']))
                logging.info(f"Selected best model: {best_model['order']}, {best_model['seasonal_order']}")
                
                pred = best_model['results'].get_forecast(steps=prediction_periods)
                output_df = pd.DataFrame(pred.predicted_mean)
                output_df.columns = [name]
                output_df.index = [f'next {x+1} {cadance}' for x in range(prediction_periods)]
                return output_df
            
            # Step 3: Fall back to original simple ARIMA
            logging.info("Falling back to simple ARIMA")
            model = sm.tsa.statespace.SARIMAX(
                agg_data,
                order=(1, 1, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            try:
                results = model.fit()
                pred = results.get_forecast(steps=prediction_periods)
                
                output_df = pd.DataFrame(pred.predicted_mean)
                output_df.columns = [name]
                output_df.index = [f'next {x+1} {cadance}' for x in range(prediction_periods)]
                
                logging.info("Forecast completed with simple ARIMA")
                return output_df
                
            except Exception as model_error:
                logging.error(f"Simple ARIMA failed: {model_error}")
                return pd.DataFrame(0, index=[f'next {x+1} {cadance}' for x in range(prediction_periods)],
                                  columns=[name])
                
    except Exception as e:
        logging.error(f"Unexpected error in ARIMA forecast for {display_name}: {e}")
        raise

def fetch_valid_entries(column_name, values):
    """
    Fetch valid entries based on column name and values.
    Will return the most active group when requested group doesn't exist.
    """
    try:
        if column_name == 'group':
            if values is not None and len(values):
                # Always get most active group
                query = f"""
                    SELECT DISTINCT c.groupname, c.groupidpath, c.numberofdevices
                    FROM collection c 
                    WHERE c.numberofdevices > 0 
                    and c.groupname in {list_to_tuple_str(values)}
                    ORDER BY c.numberofdevices DESC 
                """
                result = execute_query(query)
                if result:
                    return result
                else:
                    return []
            else:
                return []
                
        elif column_name == 'streetname':  # Handle streetname queries
            if values is not None and len(values):
                values_lower_case = [val.lower() for val in values]
                non_empty_values = []
                for val in values_lower_case:
                    pattern = f"lower(devicename) LIKE '%{val}%'"
#                     query = f"SELECT DISTINCT d.devicename FROM device d WHERE {pattern}"
                    # Make sure the streetname has device that also has log data
                    query = f"""
                         SELECT distinct d.devicename
                         FROM log l 
                         JOIN device d ON l.deviceid = d.deviceid WHERE {pattern}
                    """
                    result = execute_query(query)
                    if result:  # if result is not empty
                        non_empty_values.append(val)
                return non_empty_values
            else:
                return []
            
    except Exception as e:
        logging.error(f"Error in fetch_valid_entries: {e}")
        try:
            # Last resort fallback - get any group
            if column_name == 'group':
                query = "SELECT DISTINCT groupname, groupidpath FROM collection LIMIT 1"
                result = execute_query(query)
                if result and values and isinstance(values, list) and values[0]:
                    result[0]['original_input'] = values[0]
                return result
        except Exception as inner_e:
            logging.error(f"Error in fallback query: {inner_e}")
        return []

def parse_time_params(time_str):
    """Parse time string like '3m', '2d' into proper format"""
    if not time_str:
        return 'D', 1  # Default to 1 day
        
    # Extract number and unit
    time_str = time_str.lower().strip()
    number = ''.join(filter(str.isdigit, time_str))
    unit = ''.join(filter(str.isalpha, time_str))
    
    periods = int(number) if number else 1
    
    # Map to proper cadence
    unit_map = {
        'd': ('D', periods),
        'day': ('D', periods),
        'days': ('D', periods),
        'w': ('W', periods),
        'week': ('W', periods),
        'weeks': ('W', periods),
        'm': ('M', periods),
        'month': ('M', periods),
        'months': ('M', periods),
        'y': ('Y', periods),
        'year': ('Y', periods),
        'years': ('Y', periods)
    }
    
    cadence, periods = unit_map.get(unit, ('D', periods))
    return cadence, periods

def clean_strings(groups, words_to_remove=['group', 'street']):
    # Check if input is None
    if groups is None:
        return []
    
    # Check if list is empty
    if not groups:
        return []
    
    # Clean each string by removing specified words
    cleaned = []
    for s in groups:
        words = s.split()
        cleaned_words = [word for word in words if word.lower() not in words_to_remove]
        cleaned.append(' '.join(cleaned_words))
    
    return cleaned

def streetlight_energy_prediction(
    groups=None,
    streetnames=None,
    predict_target='group',
    prediction_cadence='month',
    prediction_periods=1
):
    """
    Predict energy consumption for streetlights.
    """
    
    # Michael Note: if question is asking street gsgl, it will make 'street gsgl' the street name, which 'gsgl' is what we want
    groups = clean_strings(groups)
    streetnames = clean_strings(streetnames)
    
    try:
        # Parse time parameters
        if isinstance(prediction_cadence, str):
            if any(c.isdigit() for c in prediction_cadence):
                # Handle format like '3m'
                cadence, periods = parse_time_params(prediction_cadence)
                logging.info(f"Parsed time parameters: cadence={cadence}, periods={periods}")
            else:
                # Handle format like 'month' with separate periods
                cadence = prediction_cadence[0].upper()
                periods = prediction_periods
                logging.info(f"Using standard parameters: cadence={cadence}, periods={periods}")
        else:
            cadence, periods = 'D', prediction_periods  # Default to daily

        # Get valid entries
        valid_entries = fetch_valid_entries(
            predict_target,
            groups if predict_target == 'group' else streetnames
        )
        logging.info(f"Fetched {len(valid_entries)} valid entries for {predict_target}")

        if not valid_entries and groups:
            logging.warning("No valid entries found")
            return pd.DataFrame()

        # For group predictions
        if predict_target == 'group':

            if valid_entries:
                group_paths = [entry['groupidpath'] for entry in valid_entries]
                conditions = [f"position('{path}' in d.groupidpath) > 0" for path in group_paths]
                where_clause = 'WHERE ' + ' OR '.join(conditions)
            else:
                where_clause = ''

            sql_query = f"""
            SELECT l.timestamp, l.deviceid, l.energyconsumption, d.groupidpath 
            FROM log l 
            JOIN device d ON l.deviceid = d.deviceid 
            {where_clause}"""
            
            logging.info(f"SQL QUERY: {sql_query}")
            df = pd.DataFrame(execute_query(sql_query))
            
            if df.empty:
                logging.warning(f"No data returned from the query.")
                return pd.DataFrame()

            # Process timestamps
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df.columns = df.columns.str.lower()
            
            # Decide the final output column name
            display_name = ''
            if len(groups) and groups:
                display_name = f'{len(groups)} Group'
            else:
                display_name = 'All Group'

            try:
                output = arima_forecast(
                    input_data=df,
#                     type='groupidpath',
                    name=display_name,
                    cadance=cadence,
#                     m=12 if cadence == 'M' else 365,  # 12 months in year or 365 days
                    prediction_periods=periods,
                    time_col='timestamp',
                    energy_col='energyconsumption'
                )
                return output
                    
            except Exception as e:
                logging.error(f"Error in prediction for group: {e}")
                return pd.DataFrame()

        else:  # streetname predictions
            if valid_entries:
                where_clause = 'WHERE ' + ' OR '.join(f"lower(d.devicename) LIKE '%{val}%'" for val in valid_entries)
            else:
                where_clause = ''

            sql_query = f"""
                SELECT l.timestamp, l.deviceid, l.energyconsumption, d.devicename 
                FROM log l 
                JOIN device d ON l.deviceid = d.deviceid 
                {where_clause}
            """
            logging.info(f"SQL QUERY: {sql_query}")

            df = pd.DataFrame(execute_query(sql_query))

            if df.empty:
                logging.warning(f"No data returned from the query.")
                return pd.DataFrame()

            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df.columns = df.columns.str.lower()
            
            # Decide the final output column name
            display_name = ''
            if len(streetnames) and streetnames:
                display_name = f'{len(streetnames)} Street'
            else:
                display_name = 'All Street'
                
            try:
                output = arima_forecast(
                    input_data=df,
#                     type='groupidpath',
                    name=display_name,
                    cadance=cadence,
#                     m=12 if cadence == 'M' else 365,  # 12 months in year or 365 days
                    prediction_periods=periods,
                    time_col='timestamp',
                    energy_col='energyconsumption'
                )
                return output
                    
            except Exception as e:
                logging.error(f"Error in prediction for streetlight: {e}")
                return pd.DataFrame()
            
            return pd.DataFrame()

    except Exception as e:
        logging.error(f"Unexpected error in streetlight_energy_prediction: {e}")
        raise

if __name__ == '__main__':
    # Example usage
    print(streetlight_energy_prediction(
        groups=['project1','project2','project7'],
        streetnames=None,
        predict_target='group',
        prediction_cadence='month',
        prediction_periods=2
    ))