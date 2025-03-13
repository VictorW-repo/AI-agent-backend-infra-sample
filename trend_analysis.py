import pandas as pd
import os
import numpy as np
#import datetime as dt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load, Parallel, delayed
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, recall_score
from sqlalchemy import create_engine, text
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Any, Dict, Union, List, Set, Tuple
import warnings
import re
from datetime import datetime, timedelta
from scipy import stats as scipy_stats

# import seaborn as sns


# - Added this function as it was missing and is used in 'calculate_basic_stats'.
def get_trend_description(slope: float) -> str:
    """
    Provide a textual description of the trend based on the slope.

    Parameters:
        slope (float): The slope of the trend.

    Returns:
        str: Description of the trend.
    """
    if slope > 0:
        return "Increasing trend"
    elif slope < 0:
        return "Decreasing trend"
    else:
        return "No trend"

def execute_query(query: str, parameters: Dict = None) -> List[Dict]:
    """
    Execute an SQL query and return the results in a list of dictionaries.

    Parameters:
        query (str): SQL query to be executed.
        parameters (dict, optional): Parameters for prepared statements.

    Returns:
        List[Dict]: Results of the query, each row as a dictionary.
    """
    # Database connection details should be securely provided, not hard-coded
    db_user = os.environ.get("DB_USER")
    db_password = os.environ.get("DB_PASSWORD")
    db_host = os.environ.get("DB_HOST")
    db_name = os.environ.get("DB_NAME")
    db_port = os.environ.get("DB_PORT", "3306")

    if not all([db_user, db_password, db_host, db_name]):
        raise ValueError("Database credentials are not fully provided.")

    connection_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(connection_uri)

    with engine.connect() as connection:
        try:
            result = connection.execute(text(query), parameters) if parameters else connection.execute(text(query))
            result_list = [dict(row) for row in result]
            return result_list
        except Exception as e:
            logging.error(f"Error executing query: {e}")
            return []

def identify_columns_to_exclude(df: pd.DataFrame) -> Set[str]:
    """
    Identify columns that should be excluded from abnormal behavior analysis.

    Parameters:
        df (pd.DataFrame): Input data

    Returns:
        Set[str]: Set of column names to exclude
    """
    exclude_cols = set()

    for col in df.columns:
        series = df[col]

        # Skip columns with all null values
        if series.isna().all():
            exclude_cols.add(col)
            continue

        # Check if column is datetime type or contains datetime-like strings
        if pd.api.types.is_datetime64_any_dtype(series):
            exclude_cols.add(col)
            continue

        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            sample = series.dropna().head(100)

            # Try to convert to datetime
            try:
                pd.to_datetime(sample, errors='raise')
                exclude_cols.add(col)
                continue
            except (ValueError, TypeError):
                pass

            # Check for common patterns indicating IDs, locations, or timestamps
            patterns = {
                'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                'device_id': r'^[A-Za-z0-9\-_]{10,}$',
                'ip_address': r'^\d{1,3}(?:\.\d{1,3}){3}$',
                'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                'coordinates': r'^-?\d+\.?\d*,\s*-?\d+\.?\d*$',
                'timestamp_patterns': [
                    r'\d{2}:\d{2}(?::\d{2})?',  # Time patterns
                    r'\d{4}-\d{2}-\d{2}',       # Date patterns
                    r'\d{2}/\d{2}/\d{4}'
                ]
            }

            # Check if majority of non-null values match any pattern
            for pattern in patterns.values():
                if isinstance(pattern, list):
                    matches = any(
                        sample.str.contains(p, regex=True, na=False).mean() > 0.5
                        for p in pattern
                    )
                else:
                    matches = sample.str.match(pattern, na=False).mean() > 0.5

                if matches:
                    exclude_cols.add(col)
                    break

            # Check for location-related columns
            location_indicators = {
                'address', 'street', 'city', 'state', 'country', 'zip', 'postal',
                'latitude', 'longitude', 'lat', 'long', 'lng', 'location'
            }
            if any(indicator in col.lower() for indicator in location_indicators):
                exclude_cols.add(col)
                continue

            # Check for identifier-like columns
            id_indicators = {
                'id', 'uuid', 'guid', 'key', 'hash', 'code', 'number', 'no', 'num',
                'reference', 'ref', 'identifier', 'token'
            }
            if any(indicator in col.lower() for indicator in id_indicators):
                exclude_cols.add(col)
                continue

            # Check for high cardinality relative to row count
            if len(sample.unique()) / len(sample) > 0.8:  # More than 80% unique values
                exclude_cols.add(col)
                continue

        # Check for columns that look like flags or binary indicators
        if is_numeric_column(series):
            unique_vals = series.nunique()
            if unique_vals <= 2:  # Binary columns
                exclude_cols.add(col)
                continue

            # Check for very low cardinality relative to total count
            unique_ratio = unique_vals / len(series)
            if unique_ratio < 0.01:  # Less than 1% unique values
                exclude_cols.add(col)
                continue

        # Check for columns with very low variation
        if is_numeric_column(series) and series.std() == 0:
            exclude_cols.add(col)
            continue

        # Check for sparse columns (mostly zeros or nulls)
        if is_numeric_column(series):
            non_zero_ratio = (series != 0).mean()
            if non_zero_ratio < 0.01:  # Less than 1% non-zero values
                exclude_cols.add(col)
                continue

    return exclude_cols

def detect_time_column(df: pd.DataFrame) -> Tuple[str, bool]:
    """
    Automatically detect the most suitable time column in the DataFrame.
    
    Parameters:
    df : DataFrame
        Input DataFrame to analyze
    
    Returns:
    Tuple[str, bool]: (detected_column, is_found)
        detected_column: Name of the detected time column
        is_found: Boolean indicating whether a suitable time column was found
    """
    if df.empty:
        return "", False
        
    # Common time-related column names
    time_related_names = {
        'timestamp', 'time', 'date', 'datetime', 'created_at', 'updated_at', 
        'start_time', 'end_time', 'creation_date', 'modification_date',
        'stamp', 'time stamp'  # Added variations
    }
    
    # Function to score column names based on likelihood of being time-related
    def get_time_name_score(col_name: str) -> int:
        col_lower = col_name.lower()
        score = 0
        
        # Direct matches with common time column names
        if col_lower in time_related_names:
            score += 3
            
        # Partial matches with time-related terms
        for term in ['time', 'date', 'day', 'month', 'year']:
            if term in col_lower:
                score += 1
                
        return score
    
    candidates = []
    
    for col in df.columns:
        series = df[col]
        name_score = get_time_name_score(col)
        
        # Skip columns with all null values
        if series.isna().all():
            continue
            
        # Check if column is already datetime type
        if pd.api.types.is_datetime64_any_dtype(series):
            candidates.append((col, name_score + 4))
            continue
            
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            sample = series.dropna().head(100)
            
            # Try to convert to datetime
            try:
                pd.to_datetime(sample)
                candidates.append((col, name_score + 3))
                continue
            except (ValueError, TypeError):
                pass
            
            # Check for common datetime patterns
            patterns = [
                r'\d{4}-\d{2}-\d{2}',                    # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',                    # DD/MM/YYYY or MM/DD/YYYY
                r'\d{4}/\d{2}/\d{2}',                    # YYYY/MM/DD
                r'\d{2}-\d{2}-\d{4}',                    # DD-MM-YYYY or MM-DD-YYYY
                r'\d{2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}',  # DD MMM YYYY
                r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
                r'\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}'   # DD/MM/YYYY HH:MM:SS
            ]
            
            for pattern in patterns:
                if sample.str.match(pattern, na=False).mean() > 0.5:
                    candidates.append((col, name_score + 2))
                    break

    if not candidates:
        return "", False
        
    # Sort candidates by score (higher is better) and position in DataFrame (earlier is better)
    scored_candidates = [
        (col, score, list(df.columns).index(col))
        for col, score in candidates
    ]
    
    # Sort by score (descending) and position (ascending)
    sorted_candidates = sorted(
        scored_candidates,
        key=lambda x: (-x[1], x[2])
    )
    
    return sorted_candidates[0][0], True


def convert_to_dataframe(data: Union[pd.DataFrame, List[Dict], Any]) -> Union[pd.DataFrame, None]:
    """Convert input data to pandas DataFrame if possible."""
    try:
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return pd.DataFrame(data)
        else:
            return None
    except Exception as e:
        warnings.warn(f"Failed to convert input to DataFrame: {str(e)}")
        return None

def is_numeric_column(series: pd.Series) -> bool:
    """Check if a column is numeric and has non-null values."""
    return pd.api.types.is_numeric_dtype(series) and not series.isna().all()

def can_be_aggregated(df: pd.DataFrame, col: str) -> bool:
    """Check if a column can be meaningfully aggregated."""
    if is_numeric_column(df[col]):
        return True
    elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
        return True
    return False

def validate_input(df: Union[pd.DataFrame, Any]) -> Tuple[bool, str]:
    """
    Validate input DataFrame and parameters.

    Parameters:
        df (Union[pd.DataFrame, Any]): The DataFrame to validate.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating validity and an error message.
    """
    if df is None:
        return False, "Input could not be converted to DataFrame"

    if not isinstance(df, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"

    if df.empty:
        return False, "DataFrame is empty"

    # Check if any time column can be detected
    time_col, found = detect_time_column(df)
    if not found:
        return False, "No suitable time column found in DataFrame"

    try:
        pd.to_datetime(df[time_col])
    except Exception as e:
        return False, f"Could not convert detected time column '{time_col}' to datetime: {str(e)}"

    analyzable_cols = [col for col in df.columns if can_be_aggregated(df, col)]
    if not analyzable_cols:
        return False, "No columns suitable for abnormal behavior analysis found"

    return True, ""

def get_aggregation_method(df: pd.DataFrame, col: str) -> str:
    """
    Determine appropriate aggregation method for a column based on its data type and content.
    
    Parameters:
    df : DataFrame
        Input DataFrame
    col : str
        Column name to analyze
    
    Returns:
    str: Appropriate aggregation method ('sum', 'mean', 'count', etc.)
    """
    series = df[col]
    
    # For numeric columns, determine if it's a continuous or discrete measure
    if is_numeric_column(series):
        # Check if the column contains primarily integers
        is_integer = pd.api.types.is_integer_dtype(series) or (
            series.dropna().apply(lambda x: float(x).is_integer()).all()
        )
        
        # Check if the column appears to be a counter or quantity
        unique_values = series.nunique()
        total_values = len(series.dropna())
        unique_ratio = unique_values / total_values if total_values > 0 else 0
        
        # Check for monetary values (looking for decimal points)
        has_decimals = not is_integer
        
        # Determine the appropriate method based on the data characteristics
        if is_integer and unique_ratio < 0.1:  # Low cardinality integers are likely categories
            return 'count'
        elif any(indicator in col.lower() for indicator in ['price', 'cost', 'revenue', 'amount', 'fee', 'salary']):
            return 'sum'  # Monetary values should typically be summed
        elif is_integer and series.min() >= 0:  # Non-negative integers are often counts
            return 'sum'
        else:  # For other numeric values, use mean
            return 'mean'
    
    # For categorical or string columns
    elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        return 'count'
    
    # For boolean columns
    elif pd.api.types.is_bool_dtype(series):
        return 'sum'  # Sum of booleans gives count of True values
    
    # For datetime columns (though these should typically be excluded)
    elif pd.api.types.is_datetime64_any_dtype(series):
        return 'count'
    
    # Default to count for any other data types
    return 'count'

def calculate_basic_stats(series: pd.Series, recent_window: int = 30) -> Dict:
    """
    Calculate basic statistics for a time series.

    Parameters:
        series (pd.Series): Time series data
        recent_window (int): Number of periods to consider for recent statistics

    Returns:
        Dict: A dictionary containing various statistics
    """
    # Get recent data
    recent_data = series.tail(recent_window)
    all_time_data = series

    # Handle empty series
    if recent_data.empty:
        raise ValueError("Recent data is empty. Cannot calculate statistics.")

    if all_time_data.empty:
        raise ValueError("All time data is empty. Cannot calculate statistics.")

    # Calculate basic statistics
    recent_mean = recent_data.mean()
    recent_std = recent_data.std()

    statistics = {
        'recent_stats': {
            'mean': recent_mean,
            'median': recent_data.median(),
            'std': recent_std,
            'min': recent_data.min(),
            'max': recent_data.max(),
            'sum': recent_data.sum(),
            'count': len(recent_data),
            'non_zero_days': (recent_data > 0).sum(),
            'coefficient_of_variation': recent_std / recent_mean if recent_mean != 0 else 0
        },
        'all_time_stats': {
            'mean': all_time_data.mean(),
            'median': all_time_data.median(),
            'std': all_time_data.std(),
            'min': all_time_data.min(),
            'max': all_time_data.max(),
            'sum': all_time_data.sum(),
            'count': len(all_time_data),
            'non_zero_days': (all_time_data > 0).sum()
        }
    }

    # Calculate trend indicators
    try:
        # Recent trend
        if len(recent_data) > 1:
            recent_days = np.arange(len(recent_data))
            recent_slope, recent_intercept, r_value, p_value, std_err = scipy_stats.linregress(recent_days, recent_data)
        else:
            recent_slope = 0
            r_value = 0
            p_value = 1

        # All-time trend
        if len(all_time_data) > 1:
            all_days = np.arange(len(all_time_data))
            all_slope, all_intercept, all_r_value, all_p_value, all_std_err = scipy_stats.linregress(all_days, all_time_data)
        else:
            all_slope = 0
            all_r_value = 0
            all_p_value = 1

        statistics['trend_analysis'] = {
            'recent': {
                'slope': recent_slope,
                'description': get_trend_description(recent_slope),
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            },
            'all_time': {
                'slope': all_slope,
                'description': get_trend_description(all_slope),
                'r_squared': all_r_value ** 2,
                'p_value': all_p_value,
                'is_significant': all_p_value < 0.05
            }
        }
    except ValueError as e:
        # Handle cases where linregress cannot be computed
        statistics['trend_analysis'] = None
    except Exception as e:
        statistics['trend_analysis'] = None

    # Calculate momentum (rate of change)
    try:
        if len(recent_data) >= 2 and recent_data.iloc[0] != 0:
            recent_momentum = (recent_data.iloc[-1] - recent_data.iloc[0]) / recent_data.iloc[0]
        else:
            recent_momentum = 0
        statistics['recent_stats']['momentum'] = recent_momentum
    except (IndexError, ZeroDivisionError):
        # Handle cases where recent_data is too short or division by zero occurs
        statistics['recent_stats']['momentum'] = 0

    # Calculate volatility measures
    if len(recent_data) > 1:
        daily_returns = recent_data.pct_change().dropna()
        statistics['recent_stats']['volatility'] = daily_returns.std()
        statistics['recent_stats']['avg_daily_change'] = daily_returns.mean()
    else:
        statistics['recent_stats']['volatility'] = 0
        statistics['recent_stats']['avg_daily_change'] = 0

    return statistics

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def analyze_abnormal_behaviors(
    data: Union[pd.DataFrame, List[Dict], Any],
    exclude_cols: List[str] = None,
    z_score_threshold: float = 3.0,
    min_occurrences: int = 5,
    show_counts: bool = False,
    auto_exclude: bool = True,
    recent_window: int = 30,
    return_agg_df: bool = False
) -> Union[Dict, Tuple[Dict, pd.DataFrame]]:
    """
    Analyze abnormal behaviors across all suitable columns in the data.

    Parameters:
        data (Union[pd.DataFrame, List[Dict], Any]): Input data.
        exclude_cols (List[str], optional): Additional columns to exclude from analysis.
        z_score_threshold (float, optional): Number of standard deviations to consider abnormal.
        min_occurrences (int, optional): Minimum number of occurrences required for analysis.
        show_counts (bool, optional): Whether to display daily counts.
        auto_exclude (bool, optional): Whether to automatically identify columns to exclude.
        recent_window (int, optional): Number of days to consider for recent statistics.
        return_agg_df (bool, optional): Whether to return the aggregated DataFrame along with results.

    Returns:
        Union[Dict, Tuple[Dict, pd.DataFrame]]: Analysis results and optionally the aggregated DataFrame
    """
    # Parameter validation
    if not isinstance(z_score_threshold, (int, float)):
        raise ValueError("z_score_threshold must be a numeric value.")

    if not isinstance(min_occurrences, int) or min_occurrences < 0:
        raise ValueError("min_occurrences must be a non-negative integer.")

    if not isinstance(recent_window, int) or recent_window <= 0:
        raise ValueError("recent_window must be a positive integer.")

    # Convert input to DataFrame if necessary
    df = convert_to_dataframe(data)

    # Input validation
    is_valid, error_message = validate_input(df)
    if not is_valid:
        logging.warning(f"Invalid input: {error_message}")
        return {"analysis": {}, "detected_time_column": None}

    # Detect time column
    time_col, found = detect_time_column(df)
    if not found:
        logging.warning("No suitable time column found.")
        return {
            "analysis": {},
            "detected_time_column": None,
            "message": "No time column available for trend analysis."
        }

    # Initialize parameters
    exclude_cols = set(exclude_cols or [])
    exclude_cols.add(time_col)

    # Auto-identify columns to exclude if enabled
    if auto_exclude:
        auto_excluded = identify_columns_to_exclude(df)
        if show_counts:
            logging.info(f"Auto-excluded columns: {sorted(auto_excluded)}")
        exclude_cols.update(auto_excluded)

    # Convert time column to datetime
    df = df.copy()
    try:
        # First try Unix timestamp in milliseconds
        df[time_col] = pd.to_datetime(df[time_col], unit='ms')
    except (ValueError, TypeError):
        try:
            # Then try standard datetime string formats
            df[time_col] = pd.to_datetime(df[time_col])
        except (ValueError, TypeError):
            try:
                # Try Unix timestamp in seconds (another common format)
                df[time_col] = pd.to_datetime(df[time_col], unit='s')
            except (ValueError, TypeError) as e:
                logging.error(f"Failed to convert time column '{time_col}' to datetime: {e}")
                return {
                    "analysis": {},
                    "detected_time_column": time_col,
                    "message": f"Time column '{time_col}' could not be converted to datetime. Expected formats: Unix timestamp in ms/s or datetime string."
                }

    # Filter data within the recent window
    end_date = df[time_col].max()
    start_date = end_date - pd.Timedelta(days=recent_window)
    df_recent = df[df[time_col] >= start_date]

    # Get analyzable columns
    analyzable_cols = [
        col for col in df.columns
        if col not in exclude_cols and can_be_aggregated(df, col)
    ]

    if not analyzable_cols:
        logging.warning("No analyzable columns found.")
        return {
            "analysis": {},
            "detected_time_column": time_col,
            "message": "No analyzable columns available for trend analysis."
        }

    results = {
        "detected_time_column": time_col,
        "analysis": {}
    }

    # Initialize an empty dictionary to store daily aggregations for each column
    all_daily_data = {}

    for col in analyzable_cols:
        agg_method = get_aggregation_method(df, col)

        # Create daily aggregations
        try:
            daily_data = df.groupby(
                pd.Grouper(key=time_col, freq='D')
            ).agg({col: agg_method})
            
            # Store the daily data for this column
            all_daily_data[col] = daily_data[col]
            
        except Exception as e:
            logging.error(f"Error during aggregation for column '{col}': {e}")
            continue

        if show_counts:
            logging.info(f"\nDaily {agg_method} for {col}:")
            logging.info(daily_data)

        series = daily_data[col].dropna()

        # Skip if there's not enough data
        if series.sum() < min_occurrences:
            logging.info(f"Skipping column '{col}' due to insufficient occurrences.")
            continue

        try:
            # Calculate basic statistics
            basic_stats = calculate_basic_stats(series, recent_window)
        except ValueError as e:
            logging.warning(f"Skipping column '{col}': {e}")
            continue
        except Exception as e:
            logging.error(f"Error calculating statistics for column '{col}': {e}")
            continue

        # Initialize result dictionary for this column
        results["analysis"][col] = {
            "basic_stats": basic_stats,
            "aggregation_method": agg_method,
            "time_range": {
                "start_date": daily_data.index.min().strftime('%Y-%m-%d'),
                "end_date": daily_data.index.max().strftime('%Y-%m-%d'),
                "total_days": len(daily_data)
            }
        }

        # Calculate anomalies if there's enough variation in the data
        if series.std() > 0:
            try:
                z_scores = scipy_stats.zscore(series)
                abnormal_indices = np.where(z_scores > z_score_threshold)[0]
                abnormal_dates = series.iloc[abnormal_indices]

                if not abnormal_dates.empty:
                    results["analysis"][col]["abnormal_behavior"] = {
                        "abnormal_dates": {
                            date.strftime('%Y-%m-%d'): {
                                "value": value,
                                "z_score": z_scores[i],
                                "times_above_mean": value / series.mean() if series.mean() > 0 else float('inf')
                            }
                            for i, (date, value) in zip(abnormal_indices, abnormal_dates.items())
                        }
                    }
            except Exception as e:
                logging.error(f"Error detecting anomalies for column '{col}': {e}")
                continue

    # Check if analysis is empty and provide a message
    if not results["analysis"]:
        results["message"] = "No significant abnormal behaviors detected or insufficient data for analysis."

    # If return_agg_df is True, combine all daily data and return both results and DataFrame
    if return_agg_df:
        if all_daily_data:
            agg_df = pd.DataFrame(all_daily_data)
            return results, agg_df
        else:
            return results, pd.DataFrame()

    return results

def generate_summary(results: Dict) -> str:
    """Generate a comprehensive text summary of both stats summary(always) and abnormal summary."""
    if not results or 'analysis' not in results:
        return "No analysis results available or invalid input provided."
    
    #summary = [f"Detected time column: {results['detected_time_column']}\n"]
    summary = []
    
    # changes: Only add time column info if it's valid
    if results.get('detected_time_column'):
        summary.append(f"Detected time column: {results['detected_time_column']}\n")

    analysis_results = results['analysis']
    if not analysis_results:
        #return "\n".join(summary + ["No analyzable columns found in the dataset."])
        return "\n".join(summary + [""])

    # Get overall date range from first column
    first_col = next(iter(analysis_results.values()))
    time_range = first_col['time_range']
    summary.append(
        f"Analysis period: {time_range['start_date']} to {time_range['end_date']}\n"
    )
    
    for col_name, data in analysis_results.items():
        summary.append(f"Analysis for {col_name} ({data['aggregation_method']}):")
        
        # Recent statistics
        recent_stats = data['basic_stats']['recent_stats']
        summary.append("Recent statistics (last 30 days):")
        summary.append(f"• Average: {recent_stats['mean']:.2f}")
        summary.append(f"• Median: {recent_stats['median']:.2f}")
        summary.append(f"• Volatility (std): {recent_stats['std']:.2f}")
        if 'volatility' in recent_stats:
            summary.append(f"• Daily volatility: {recent_stats['volatility']:.2%}")
        if 'momentum' in recent_stats:
            summary.append(f"• Momentum: {recent_stats['momentum']:.2%}")
        
        # Trend analysis
        if data['basic_stats']['trend_analysis']:
            trend = data['basic_stats']['trend_analysis']['recent']
            summary.append(f"• Recent trend: {trend['description']}")
            if trend['is_significant']:
                summary.append(f"  - Statistically significant (p < 0.05)")
                summary.append(f"  - R-squared: {trend['r_squared']:.3f}")
        
        # Anomalies
        if 'abnormal_behavior' in data:
            abnormal_dates = data['abnormal_behavior']['abnormal_dates']
            summary.append(f"• Detected {len(abnormal_dates)} anomalies:")
            
            # Show top 3 anomalies
            sorted_dates = sorted(
                abnormal_dates.items(),
                key=lambda x: x[1]['z_score'],
                reverse=True
            )[:3]
            
            for date, info in sorted_dates:
                summary.append(
                    f"  - {date}: value: {info['value']:.1f}, "
                    f"{info['times_above_mean']:.1f}x above mean"
                )
        else:
            summary.append("• No significant anomalies detected")
        
        summary.append("")  # Empty line between columns
    
    return "\n".join(summary)

import random

def generate_extended_device_data(start_date_str: str, end_date_str: str) -> List[Dict]:
    """
    Generate extended device data between two dates.

    Parameters:
        start_date_str (str): Start date in 'YYYY-MM-DD' format.
        end_date_str (str): End date in 'YYYY-MM-DD' format.

    Returns:
        List[Dict]: Generated device data.
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    device_data = []
    
    # Define device configurations
    devices = [
        {
            'Device Id': '1000',
            'base_power': 100,
            'base_voltage': 220,
            'base_dimming': 40,
            'Group Id Path': '0000000077',
            'common_faults': ['', 'luxModuleFault', 'highLoadVoltage']
        },
        {
            'Device Id': '100005',
            'base_power': 60,
            'base_voltage': 230,
            'base_dimming': 95,
            'Group Id Path': None,
            'common_faults': ['lowPF', 'lowPower', 'highLoadVoltage', 'lowPower.highLoadVoltage']
        }
    ]
    
    current_date = start_date
    while current_date < end_date:
        for device in devices:
            # Add some random variation to measurements
            start_power = device['base_power'] + random.randint(-5, 5)
            end_power = start_power + random.randint(-10, 10)
            
            start_voltage = device['base_voltage'] + random.randint(-3, 3)
            end_voltage = start_voltage + random.randint(-5, 5)
            
            start_dimming = device['base_dimming'] + random.randint(-2, 2)
            end_dimming = start_dimming + random.randint(-5, 15)
            
            # Ensure values stay within reasonable ranges
            end_dimming = min(max(end_dimming, 0), 100)
            
            entry = {
                'Device Id': device['Device Id'],
                'Start Timestamp': str(int(current_date.timestamp() * 1000)),
                'Start Time': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                'Start Power': start_power,  # Now an integer
                'Start Voltage': start_voltage,  # Now an integer
                'Start Dimming': start_dimming,  # Now an integer
                'Group Id Path': device['Group Id Path'],
                'Start Alarm Fault': random.choice(device['common_faults']),
                'End Timestamp': '1730585760000',
                'End Time': '2024-11-02 22:16:00',
                'End Power': end_power,  # Now an integer
                'End Voltage': end_voltage,  # Now an integer
                'End Dimming': end_dimming,  # Now an integer
                'End Alarm Fault': random.choice(device['common_faults'])
            }
            device_data.append(entry)
        current_date += timedelta(days=1)
    
    return device_data

import json
import re

def extract_and_fix_json(input_data: Union[str, List[Dict]]) -> List[Dict]:
    """
    Extracts and fixes JSON content from a string or processes directly if input is already a list.
    Handles incomplete JSON, including missing key-value pairs, and automatically determines
    the expected keys from the first complete object.

    Args:
        input_data (str or list): Input string containing JSON-like content or already parsed JSON list.
        
    Returns:
        list: Parsed JSON list
    """
    # If the input is already a list, return it as is
    if isinstance(input_data, list):
        return input_data

    # Otherwise, process input as a string
    # Find content after '['
    start_idx = input_data.find('[')
    if start_idx == -1:
        raise ValueError("No '[' found in input string")
    
    content = input_data[start_idx:]
    
    try:
        # Try to parse as is
        return json.loads(content)
    except json.JSONDecodeError:
        objects = []
        expected_keys = None
        
        # Split the content into individual object strings
        # Remove the leading '[' and trailing characters
        content = content[1:].rstrip('"})]')
        # Split by the pattern that separates objects
        object_strings = content.split('}, {')
        
        for i, obj_str in enumerate(object_strings):
            # Clean up the object string
            if i == 0:
                obj_str = obj_str + '}'
            elif i == len(object_strings) - 1:
                obj_str = '{' + obj_str
                # If it's incomplete, add closing brace
                if not obj_str.endswith('}'):
                    obj_str += '}'
            else:
                obj_str = '{' + obj_str + '}'
            
            try:
                # Try to parse the object
                obj = json.loads(obj_str)
                # If this is the first successfully parsed object, get its keys as template
                if expected_keys is None:
                    expected_keys = list(obj.keys())
                objects.append(obj)
            except json.JSONDecodeError:
                # If we haven't found expected_keys yet, try to extract them from the string
                if expected_keys is None:
                    # Extract keys using regex pattern
                    keys = re.findall(r'\'([^\']+)\'\s*:', obj_str)
                    if keys:
                        expected_keys = keys
                    else:
                        raise ValueError("Cannot determine the structure of objects")
                
                # Extract key-value pairs manually
                obj = {}
                # Extract all key-value pairs using regex
                pairs = re.findall(r'\'([^\']+)\'\s*:\s*(?:\'([^\']+)\'|None|\d+)', obj_str)
                for key, value in pairs:
                    obj[key] = value if value else None
                
                # Ensure all expected keys are present
                for key in expected_keys:
                    if key not in obj:
                        obj[key] = None
                        
                objects.append(obj)
        
        return objects

def calculate_failure_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive failure statistics from failure data.
    
    Args:
        df (pd.DataFrame): DataFrame containing failure data with at least 'Alarm Fault' and 'Occurrence Count' columns
        
    Returns:
        Dict[str, Any]: Structured statistics about failures
    """
    if not all(col in df.columns for col in ['Alarm Fault', 'Occurrence Count']):
        return None
        
    # Convert 'Occurrence Count' to numeric if it's not already
    df['Occurrence Count'] = pd.to_numeric(df['Occurrence Count'])
    
    # Basic counts
    total_failures = df['Occurrence Count'].sum()
    unique_types = len(df['Alarm Fault'].unique())
    
    # Distribution statistics
    distribution_stats = {
        "mean": float(df['Occurrence Count'].mean()),  # Convert to native float
        "median": float(df['Occurrence Count'].median()),
        "std": float(df['Occurrence Count'].std()),
        "max": int(df['Occurrence Count'].max()),  # Convert to native int
        "min": int(df['Occurrence Count'].min()),
        "quartiles": {
            "25th": float(df['Occurrence Count'].quantile(0.25)),
            "75th": float(df['Occurrence Count'].quantile(0.75))
        }
    }
    
    # Most common failures
    most_common = [
        {
            "Alarm Fault": str(record["Alarm Fault"]),
            "Occurrence Count": int(record["Occurrence Count"])
        }
        for record in df.nlargest(5, 'Occurrence Count')[['Alarm Fault', 'Occurrence Count']].to_dict('records')
    ]    
    # Failure patterns
    failure_patterns = {
        "single_faults": len(df[~df['Alarm Fault'].str.contains('\.')]),
        "compound_faults": len(df[df['Alarm Fault'].str.contains('\.')])
    }
    
    # Time-based analysis (if timestamp available)
    time_analysis = {}
    if 'timeStamp' in df.columns:
        df['timeStamp'] = pd.to_datetime(df['timeStamp'])
        time_analysis = {
            "failures_by_hour": {
                str(k): int(v) for k, v in  # Convert hour numbers to str and counts to native int
                df.groupby(df['timeStamp'].dt.hour)['Occurrence Count'].sum().to_dict().items()
            },
            "failures_by_day": {
                str(k): int(v) for k, v in  # Convert day names to str and counts to native int
                df.groupby(df['timeStamp'].dt.day_name())['Occurrence Count'].sum().to_dict().items()
            }
        }
    
    # Severity classification
    severity_thresholds = {
        "high": float(df['Occurrence Count'].quantile(0.75)),    # Convert to native float
        "medium": float(df['Occurrence Count'].median()),        # Convert to native float
        "low": float(df['Occurrence Count'].quantile(0.25))     # Convert to native float
    }
    
    # Categorize failures by severity
    def get_severity(count):
        if count >= severity_thresholds["high"]:
            return "high"
        elif count >= severity_thresholds["medium"]:
            return "medium"
        return "low"
    
    severity_distribution = {
        str(k): int(v) for k, v in  # Convert keys to str and values to native int
        df['Occurrence Count'].apply(get_severity).value_counts().to_dict().items()
    }
        
    alarm_stats = {
        "overview": {
            "total_failures": int(total_failures),  # Convert numpy.int64
            "unique_failure_types": int(unique_types),  # Convert numpy.int64
            "failure_rate": float(total_failures / len(df) if len(df) > 0 else 0)
        },
        "distribution": distribution_stats,
        "patterns": {
            "most_common_failures": most_common,
            "failure_patterns": failure_patterns
        },
        "time_analysis": time_analysis,
        "severity": {
            "thresholds": severity_thresholds,
            "distribution": severity_distribution
        }
    }
    
    # Add trend analysis if time data available
    if time_analysis:
        recent_data = df[df['timeStamp'] >= df['timeStamp'].max() - timedelta(days=7)]
        if not recent_data.empty:
            alarm_stats["trends"] = {
                "week_over_week_change": ((recent_data['Occurrence Count'].sum() - 
                    df[df['timeStamp'].between(
                        df['timeStamp'].max() - timedelta(days=14),
                        df['timeStamp'].max() - timedelta(days=7)
                    )]['Occurrence Count'].sum()) / 
                    recent_data['Occurrence Count'].sum() * 100),
                "recent_trend": "increasing" if recent_data['Occurrence Count'].is_monotonic_increasing else 
                               "decreasing" if recent_data['Occurrence Count'].is_monotonic_decreasing else 
                               "fluctuating"
            }
    
    return alarm_stats


if __name__ == '__main__':
    # Generate data from October 1st to November 1st, 2023
    device_data = generate_extended_device_data('2023-10-01', '2023-11-01')

    # Make some fake abnormal behavior
    device_data[-1]['Start Power'] = 9999999

    # Ensure the test data includes 'timeStamp' consistently
    device_data = [
        {
            'Device Name': 'gsgl-60W-A008-Test',
            'Group Control Fault': None,
            'Meter Fault': None,
            'Driver Fault': None,
            'Light Fault': None,
            'timeStamp': 1730585760000  # Unix timestamp in milliseconds
        },
        {
            'Device Name': 'gsgl-60W-A008-8',
            'Group Control Fault': '0',
            'Meter Fault': '0',
            'Driver Fault': '0',
            'Light Fault': '0',
            'timeStamp': 1730585760000
        },
        # Add more records as needed with 'timeStamp'
    ]

    test_input = '''[
    {
        "Query for Identify all deviceNames where log.alarmFault.highTemperature is true from all time to all time": [
            {
                "Device Id": "100194",
                "Alarm Fault": "luxModuleFault.lowLoadVoltage.highCurrent",
                "High Load Current Threshold": "950",
                "High Load Current": null,
                "Low Load Current": null,
                "Low Load Current Threshold": "120",
                "timeStamp": "2024-11-02 18:16:00"
            },
            {
                "Device Id": "101034",
                "Alarm Fault": "highCurrent",
                "High Load Current Threshold": "950",
                "High Load Current": null,
                "Low Load Current": null,
                "Low Load Current Threshold": "120",
                "timeStamp": "2024-11-02 18:16:00"
            }
        ]
    },
        {
            "Query for Identify all deviceNames where log.alarmFault.lowPower is true and collection.groupName='5' from all time to all time": [
                {
                    "Device Id": "100194",
                    "Alarm Fault": "luxModuleFault.lowLoadVoltage.highCurrent",
                    "High Load Current Threshold": "950",
                    "High Load Current": null,
                    "Low Load Current": null,
                    "Low Load Current Threshold": "120",
                    "timeStamp": "2024-11-02 18:16:00"
                },
                {
                    "Device Id": "101034",
                    "Alarm Fault": "highCurrent",
                    "High Load Current Threshold": "950",
                    "High Load Current": null,
                    "Low Load Current": null,
                    "Low Load Current Threshold": "120",
                    "timeStamp": "2024-11-02 18:16:00"
                }
            ]
        }
    ]'''
    print(extract_and_fix_json(test_input))

    # results = analyze_abnormal_behaviors(
    #     extract_and_fix_json(test_input),  
    #     z_score_threshold=3.0,  # Change this smaller to make it more sensitive
    #     show_counts=False,       # Change to True to see aggregation results/debug
    # )

    # print(generate_summary(results))

    convert_to_dataframe(extract_and_fix_json(test_input)).to_csv('../../../trend_analysis_df.csv')