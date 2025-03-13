from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import json
from RAG import direct_rag
from query_api import llm_api
import datetime
import requests
import time
from middleware_helper_functions import convert_to_unix_time
from typing import List, Dict, Union, Optional


def fetch_streetlight_data_and_generate_link(base_url, streetlight_ids=None, street_names=None, groups=None, failure_types=None):
    api_url = f'{base_url}/api/streetlights'
    
    params = {}
    if street_names:
        params['street_names'] = street_names
    if groups:
        params['groups'] = groups
    if failure_types:
        params['failure_types'] = failure_types
    if streetlight_ids:
        params['streetlight_ids'] = streetlight_ids
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status() 
        
        link = f'{base_url}?{"&".join(f"{key}={value}" for key, values in params.items() for value in (values if isinstance(values, list) else [values]))}'
        return link
    
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Other Error: {err}")

def fetch_error_map_link(streetlight_ids = None, street_names = None, groups = None, failure_types = None):
    base_url = "http://gpu.cioic.com:2000"
    link_specific = fetch_streetlight_data_and_generate_link(
        base_url, 
        streetlight_ids=streetlight_ids,
        street_names=street_names,
        groups=groups, 
        failure_types=failure_types
    )
    return link_specific

import re
from typing import List, Dict, Union
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from collections import Counter

import sys
import os
sys.path.append('../config')
from config_api import get_db_uri

def camel_to_title(string: str) -> str:
    return re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', string).title()

def query_streetlights(start_date: int, end_date: int, groups: List[str] = None, street_names: List[str] = None, streetlights: List[str] = None) -> Union[List[Dict], None]:
    """
    Query streetlight data for a specific time range.
    
    Args:
        start_date: Unix timestamp in milliseconds
        end_date: Unix timestamp in milliseconds
        groups: Optional list of group identifiers
        street_names: Optional list of street names
        streetlights: Optional list of streetlight IDs
    """
    # UPDATED: Better timestamp logging
    print(f"\n=== Time Range Parameters ===")
    print(f"Start date: {start_date} ({unix_ms_to_datetime_str(start_date)})")
    print(f"End date: {end_date} ({unix_ms_to_datetime_str(end_date)})")
    print(f"Duration: {(end_date - start_date) / (1000 * 60 * 60)} hours")
    print("===========================\n")

     # Get connection URI from config
    connection_uri = get_db_uri("govchat_streetlighting") #hardcoded to streetlighting database 
    print(f"Attempting to connect to database using config settings")

    # Failure statuses
    failure_statuses = [
        "highVoltage", "lightFault", "flickering", "driverCommunicationFault",
        "dayBurn", "relaySticking", "relayOpenCircuit", "lowLoadPower",
        "lowLoadCurrent", "groupControlFault", "linkControlFault", "lowVoltage",
        "luxCommunicationFault", "highCurrent", "lowCurrent", "highPower",
        "lowPower", "gpsFailure", "powerOutage", "noNetwork", "lowLoadVoltage",
        "highLoadVoltage", "highLoadPower", "highPF", "lowTemperature",
        "highLoadCurrent", "lowPF", "highTemperature", "meterFault",
        "luxModuleFault", "driverFault"
    ]

    failure_count_queries = [
        f"""SUM(CASE WHEN POSITION('{status}' IN ll.alarmfault) > 0 THEN 1 ELSE 0 END) AS "{status}_count\""""
        for status in failure_statuses
    ]

    base_query = f"""
            WITH active_devices AS (
                -- Get only devices that have had activity in the specified time range
                SELECT DISTINCT l.deviceid
                FROM log l
                WHERE l.timestamp BETWEEN :start_time AND :end_time
            ),
            latest_log AS (
                SELECT l.*,
                    ROW_NUMBER() OVER (PARTITION BY l.deviceid ORDER BY l.timestamp DESC) as rn
                FROM log l
                JOIN active_devices actdev ON l.deviceid = actdev.deviceid  -- Changed alias
                WHERE l.timestamp BETWEEN :start_time AND :end_time
            ),
            previous_period AS (
                SELECT l.*,
                    ROW_NUMBER() OVER (PARTITION BY l.deviceid ORDER BY l.timestamp DESC) as rn
                FROM log l
                JOIN active_devices actdev ON l.deviceid = actdev.deviceid  -- Changed alias
                WHERE l.timestamp BETWEEN :previous_start_time AND :previous_end_time
            ),
            wattage_counts AS (
                SELECT ll.power, COUNT(*) as count
                FROM latest_log ll
                WHERE ll.rn = 1
                GROUP BY ll.power
            ),
            device_group AS (
                SELECT d.deviceid, c.groupname
                FROM device d
                JOIN active_devices actdev ON d.deviceid = actdev.deviceid  -- Changed alias
                JOIN collection c ON POSITION(c.groupid IN d.groupidpath) > 0
            ),
            current_energy_consumption AS (
                SELECT dg.groupname, SUM(ll.energyconsumption) as current_consumption
                FROM device_group dg
                JOIN latest_log ll ON dg.deviceid = ll.deviceid AND ll.rn = 1
                GROUP BY dg.groupname
            ),
            previous_energy_consumption AS (
                SELECT dg.groupname, SUM(pl.energyconsumption) as previous_consumption
                FROM device_group dg
                JOIN previous_period pl ON dg.deviceid = pl.deviceid AND pl.rn = 1
                GROUP BY dg.groupname
            ),
            hourly_dimming AS (
                SELECT 
                    EXTRACT(HOUR FROM to_timestamp(l.timestamp/1000)) AS hour,
                    l.currentdimming AS dimming_level,
                    COUNT(*) AS frequency,
                    AVG(l.currentdimming) AS avg_dimming
                FROM log l
                JOIN active_devices actdev ON l.deviceid = actdev.deviceid  -- Changed alias
                WHERE l.timestamp BETWEEN :start_time AND :end_time
                GROUP BY EXTRACT(HOUR FROM to_timestamp(l.timestamp/1000)), l.currentdimming
            ),
            mode_dimming AS (
                SELECT hour, dimming_level AS mode_dimming, frequency
                FROM hourly_dimming hd1
                WHERE frequency = (
                    SELECT MAX(frequency)
                    FROM hourly_dimming hd2
                    WHERE hd1.hour = hd2.hour
                )
            ),
            avg_dimming AS (
                SELECT hour, AVG(avg_dimming) AS avg_dimming
                FROM hourly_dimming
                GROUP BY hour
            )
            SELECT 
                STRING_AGG(DISTINCT dg.groupname, '||') AS "Groups",
                STRING_AGG(DISTINCT d.devicename, '||') AS "Street Names",
                STRING_AGG(DISTINCT d.deviceid, '||') AS "Involved Streetlights",
                '(' || MIN(d.latitude) || ' to ' || MAX(d.latitude) || ', ' ||
                MIN(d.longitude) || ' to ' || MAX(d.longitude) || ')' AS "Geographical Coordinates",
                STRING_AGG(DISTINCT d.hardwareversion, '||') AS "Types of Streetlights",
                STRING_AGG(wc.power || ':' || wc.count, '||') AS "Wattage Counts",
                STRING_AGG(DISTINCT 
                    dg.groupname || ':' || cec.current_consumption || ':' || 
                    COALESCE(pec.previous_consumption, 0) || ':' || 
                    COALESCE(cec.current_consumption - pec.previous_consumption, 0), 
                    '||') AS "Energy Consumption by Group",
                STRING_AGG(                                              
                    md.hour || ':' || md.mode_dimming || ':' || avgd.avg_dimming,  -- Changed alias
                    '||' ORDER BY md.hour) AS "Hourly Dimming Levels",
                {', '.join(failure_count_queries)}
            FROM 
                device d
            JOIN 
                active_devices actdev ON d.deviceid = actdev.deviceid  -- Changed alias
            LEFT JOIN 
                device_group dg ON d.deviceid = dg.deviceid
            LEFT JOIN 
                latest_log ll ON d.deviceid = ll.deviceid AND ll.rn = 1
            LEFT JOIN
                wattage_counts wc ON ll.power = wc.power
            LEFT JOIN
                current_energy_consumption cec ON dg.groupname = cec.groupname
            LEFT JOIN
                previous_energy_consumption pec ON dg.groupname = pec.groupname
            LEFT JOIN
                mode_dimming md ON 1=1
            LEFT JOIN
                avg_dimming avgd ON md.hour = avgd.hour  -- Changed alias
            WHERE 1=1
        """


    # UPDATED: Simplified params with direct time assignments
    params = {
        "start_time": start_date,
        "end_time": end_date,
        "previous_start_time": start_date - (end_date - start_date),
        "previous_end_time": start_date
    }
    
    # UPDATED: Better parameter logging
    print("\n=== Query Parameters ===")
    print(json.dumps(params, indent=2))
    print("======================\n")

    # Build WHERE conditions
    conditions = []
    if groups:
        conditions.append("EXISTS (SELECT 1 FROM device_group dg WHERE dg.deviceid = d.deviceid AND dg.groupname = ANY(:groups))")
        params["groups"] = groups
    if street_names:
        conditions.append("d.devicename = ANY(:street_names)")
        params["street_names"] = street_names
    if streetlights:
        conditions.append("d.deviceid = ANY(:streetlights)")
        params["streetlights"] = streetlights

    if conditions:
        base_query += " AND " + " AND ".join(conditions)

    engine = None
    try:
        engine = create_engine(connection_uri)
        
        with engine.connect() as connection:
            result = connection.execute(text(base_query), params)
            row = result.fetchone()
            
            if not row:
                print("No data found for the specified time range")
                return []
            
            # Process failure counts
            failure_counts = {}
            start_index = 8  # Start at column 8 where the failure counts begin
            
            for i, status in enumerate(failure_statuses):
                try:
                    if start_index + i < len(row):
                        value = row[start_index + i]
                        failure_counts[camel_to_title(status)] = int(value or 0)
                except (ValueError, TypeError) as e:
                    print(f"Error processing failure count for {status}: {e}")
                    failure_counts[camel_to_title(status)] = 0

            # Process other data fields
            formatted_result = {
                "Groups/Geozones/Projects": str(row[0] or "").split('||'),
                "Street Names": str(row[1] or "").split('||'),
                "Involved Streetlights": str(row[2] or "").split('||'),
                "Geographical Coordinates": str(row[3] or ""),
                "Types of Streetlights": str(row[4] or "").split('||'),
                "Wattage Counts": _process_wattage_counts(str(row[5] or "")),
                "Energy Consumption by Group": _process_energy_consumption(str(row[6] or "")),
                "Hourly Dimming Levels (%)": _process_hourly_dimming(str(row[7] or "")),
                "Failure Counts": {k: v for k, v in failure_counts.items() if v > 0}
            }
            
            return [formatted_result]

    except SQLAlchemyError as e:
        print(f"Database error: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None
    finally:
        if engine:
            engine.dispose()

# ADDED: Helper functions to process different data types
def _process_wattage_counts(wattage_counts_str: str) -> Dict[str, int]:
    """Process wattage counts string into a dictionary"""
    wattage_counts = {}
    for item in wattage_counts_str.split('||'):
        try:
            if ':' in item:
                wattage, count = item.rsplit(':', 1)
                if wattage.strip() and count.strip():
                    wattage_counts[f"{wattage.strip()}W"] = int(count.strip())
        except (ValueError, TypeError) as e:
            print(f"Error processing wattage count: {e}")
    return wattage_counts

def _process_energy_consumption(energy_consumption_str: str) -> Dict[str, Dict[str, float]]:
    """Process energy consumption string into a dictionary"""
    energy_consumption = {}
    for item in energy_consumption_str.split('||'):
        try:
            if ':' in item:
                group_name, current, previous, diff = item.split(':')
                energy_consumption[group_name] = {
                    "current (kWh)": float(current),
                    "previous (kWh)": float(previous)
                }
        except (ValueError, TypeError) as e:
            print(f"Error processing energy consumption: {e}")
    return energy_consumption

def _process_hourly_dimming(hourly_dimming_str: str) -> Dict[int, Dict[str, float]]:
    """Process hourly dimming string into a dictionary"""
    hourly_dimming = {}
    for item in hourly_dimming_str.split('||'):
        try:
            if ':' in item:
                hour, mode, avg = item.split(':')
                hourly_dimming[int(hour)] = {
                    "mode": float(mode),
                    "average": float(avg)
                }
        except (ValueError, TypeError) as e:
            print(f"Error processing hourly dimming: {e}")
    return hourly_dimming

import json
from typing import List, Dict, Any

def flatten_and_stringify(data: Dict[str, Any]) -> List[Dict[str, str]]:
    result = []
    for key, value in data.items():
        if isinstance(value, list):
            result.append({key: ', '.join(map(str, value))})
        elif isinstance(value, dict):
            if key == "Maintenance Status":
                result.append({key: '\n'.join(f"{sub_key}: {sub_value}" for sub_key, sub_value in value.items())})
            else:
                for sub_key, sub_value in value.items():
                    result.append({f"{key} - {sub_key}": str(sub_value)})
        else:
            result.append({key: str(value)})
    return result

import json
from typing import List, Dict
from datetime import datetime, timedelta


def unix_ms_to_datetime_str(unix_ms: int) -> str:
    """Convert Unix millisecond timestamp to datetime string"""
    unix_seconds = unix_ms / 1000
    dt = datetime.fromtimestamp(unix_seconds)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def internal_report_streetlighting(start_date: Union[int, str, None] = None, 
                                 end_date: Union[int, str, None] = None, 
                                 groups: Optional[List[str]] = None, 
                                 street_names: Optional[List[str]] = None, 
                                 streetlights: Optional[List[str]] = None, 
                                 errors: Optional[List[str]] = None) -> Dict:
    """
    Generate internal report for streetlighting data with proper time range handling.
    """
    try:
        # Get current time in milliseconds
        current_time = int(time.time() * 1000)
        
        # Convert start_date to timestamp
        if start_date is None:
            # Default to 24 hours ago if no start date provided
            start_time = current_time - (24 * 60 * 60 * 1000)
        elif isinstance(start_date, int):
            start_time = start_date
        elif isinstance(start_date, str):
            start_time = convert_to_unix_time(start_date)
        
        # Convert end_date to timestamp
        if end_date is None:
            end_time = current_time
        elif isinstance(end_date, int):
            end_time = end_date
        elif isinstance(end_date, str):
            if end_date in ["0 days ago", "now"]:
                end_time = current_time
            else:
                end_time = convert_to_unix_time(end_date)

        # Verify the time range is valid (end time should be after start time)
        if end_time < start_time:
            raise ValueError("End time cannot be before start time")

        # Debug logging
        print(f"\n=== Time Range Details ===")
        print(f"Start time: {datetime.fromtimestamp(start_time/1000)} ({start_time})")
        print(f"End time: {datetime.fromtimestamp(end_time/1000)} ({end_time})")
        print(f"Duration: {(end_time - start_time)/(1000*60*60)} hours")
        print("========================\n")
            
        # Query the data with the specified time range
        data = query_streetlights(start_time, end_time, groups, street_names, streetlights)
        print(f"Query result: {'Data received' if data else 'No data or error'}")
        
        time1 = unix_ms_to_datetime_str(start_time)
        time2 = unix_ms_to_datetime_str(end_time)
        
        if not data:
            print("\nNo data received, preparing error response...")
            error_message = {
                "Error": "No data found or error occurred while querying streetlight data",
                "Time Period": f"From {time1} to {time2}"
            }
            return {
                "function": "table_vertical",
                "parameters": [error_message]
            }
        
        # Process the successful query result
        flattened_data = []
        
        # Add introduction
        introduction = {
            "Introduction": f"This is a streetlighting report tailored for fault and consumption analysis. The following report covers the time period from {time1} to {time2}"
        }
        flattened_data.append(introduction)
        
        # Process the main data
        for item in data:
            flattened_items = flatten_and_stringify(item)
            flattened_data.extend(flattened_items)
            
            
        
        final_output = {
            "function": "table_vertical",
            "parameters": flattened_data
        }
        
        return final_output
        
    except Exception as e:
        print(f"Error in internal_report_streetlighting: {str(e)}")
        time1 = unix_ms_to_datetime_str(start_time) if 'start_time' in locals() else 'unknown'
        time2 = unix_ms_to_datetime_str(end_time) if 'end_time' in locals() else 'unknown'
        return {
            "function": "table_vertical",
            "parameters": [{
                "Error": f"Error in internal report generation: {str(e)}",
                "Time Period": f"From {time1} to {time2}"
            }]
        }


if __name__ == '__main__':
    start_date = 1619763200000
    end_date = 1743605258366
    
    try:
        result = internal_report_streetlighting(start_date, end_date)
        print("\nScript completed successfully")
    except Exception as e:
        print(f"Error in main execution: {e}")