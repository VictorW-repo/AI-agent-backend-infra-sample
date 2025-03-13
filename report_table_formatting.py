from datetime import datetime, timedelta
import re
from sqlalchemy import create_engine, text
import traceback
from sqlalchemy.exc import SQLAlchemyError
from process_sql import convert_mysql_to_postgresql
import sys
import os
sys.path.append('../config')
from config_api import get_db_uri

POSTGRES_COLUMN_MAPPING = {
    # Device table
    "streetName": "devicename",
    "deviceId": "deviceid",
    "groupIdPath": "groupidpath",
    "hardwareVersion": "hardwareversion",
    # Log table
    "timeStamp": "timestamp",
    "energyConsumption": "energyconsumption",
    # Error internal table
    "eventTime": "eventtime",
    "controllerFault": "controllerfault",
    "startEndFlag": "startendflag",
    "alarmFault": "alarmfault"
}

def map_column_name(column_name):
    """Helper function to map MySQL column names to PostgreSQL column names"""
    if column_name in POSTGRES_COLUMN_MAPPING:
        return POSTGRES_COLUMN_MAPPING[column_name]
    return column_name.lower()  # PostgreSQL defaults to lowercase



def get_unit(condition):
    if '_current' in condition:
        return 'A'
    elif '_voltage' in condition:
        return 'V'
    elif '_power' in condition:
        return 'W'
    elif 'energy_consumption' in condition:
        return 'kWh'
    elif '_temperature' in condition:
        return '°C'
    elif condition in ['total_running_hours', 'remaining_lifetime']:
        return 'hours'
    elif '_rate' in condition or condition == 'relative_illumination_count':
        return '%'
    elif '_threshold' in condition:
        if '_current' in condition:
            return 'A'
        elif '_voltage' in condition:
            return 'V'
        elif '_power' in condition:
            return 'W'
        elif '_temperature' in condition:
            return '°C'
        elif '_power_factor' in condition:
            return 'ratio'
    elif condition == 'absolute_illumination_count':
        return 'count'
    return ""  # Default case if no match is found

def execute_query(query, database = "govchat_streetlighting", parameters=None):

    try:
        connection_uri = get_db_uri(database)  # Get URI from config
        engine = create_engine(connection_uri)

        with engine.connect() as connection:
            result = connection.execute(text(query), parameters) if parameters else connection.execute(text(query))
            result_list = [row._asdict() for row in result]
            return result_list
    except SQLAlchemyError as e:
        error_message = str(e.__dict__['orig'])
        print(f"A database error occurred: {error_message}")
        print(f"Query: {query}")
        if parameters:
            print(f"Parameters: {parameters}")
        print("Traceback:")
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print(f"Query: {query}")
        if parameters:
            print(f"Parameters: {parameters}")
        print("Traceback:")
        traceback.print_exc()
    return []

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def format_title(condition):
    """
    Formats the title by properly handling camel case and underscores.
    """
    parts = condition.split('_')
    
    formatted_parts = []
    for part in parts:
        camel_parts = camel_case_split(part)
        formatted_parts.extend(word.capitalize() for word in camel_parts)
    
    return ' '.join(formatted_parts)

def convert_ms_to_datetime(ms):
    seconds = ms / 1000.0
    date_time = datetime.fromtimestamp(seconds)
    formatted_date_time = date_time.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_date_time

recent_boundary = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)

def is_unix_timestamp(value):
    try:
        num = float(value)
        return 946684800000 <= num <= 253402300799000  # 2000-01-01 to 9999-12-31
    except (ValueError, TypeError):
        return False

def convert_unix_times(data):
    if isinstance(data, dict):
        return {k: convert_unix_times(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_unix_times(item) for item in data]
    elif is_unix_timestamp(data):
        return convert_ms_to_datetime(int(data))
    else:
        return data

def process_query_to_datetime(query_results):
    return [convert_unix_times(item) for item in query_results]

def get_error_column(database):
    return "controllerFault" if database.endswith("streetlighting") else "alarmFault"

project_level_failures = {
    "electrical_switchgear": [
        "loop_failure", "electric_failure", "power_quality_failure",
        "environmental_failure", "equipment_failure", "communication_failure"
    ],
    "electrical_transformer": ["failure"],
    "curtain": ["failure"],
    "elevator": ["failure"],
    "firefighting": ["failure"],
    "hvac_airhandler": ["failure"],
    "hvac_cooling": ["failure"],
    "lighting": ["failure"],
    "security_camera": ["failure"],
    "security_patrol": ["failure"],
    "water_pump": ["failure"],
    "water_tank": ["failure"],
    "water_valve": ["failure"],
    "restroom_paper": ["failure"],
    "restroom_soap": ["failure"],
    "restroom_toilet": ["failure"],
    "environment": ["failure"]
}

device_level_failures = {
    "electrical_switchgear": [
        "lowCurrentPhaseSingle", "highCurrentPhaseA", "highCurrentPhaseB",
        "highCurrentPhaseC", "highCurrentPhaseSingle", "lowVoltagePhaseA",
        "lowVoltagePhaseB", "lowVoltagePhaseC", "lowVoltagePhaseSingle",
        "highVoltagePhaseSingle", "phaseUnbalance", "highCurrent",
        "relayLeakage", "meterlowPowerFactor", "lowPowerFactor",
        "highPower", "overload", "harmonics", "switchFailure",
        "resistor", "insulation", "energyMeteringFault", "wiringFault",
        "waterIngress", "smoke", "dustIngress", "doorOpen",
        "highTemperature", "waterIngressSensorCommunicationFault",
        "dustIngressSensorCommunicationFault", "currentHarvesterCommunicationFailureOne",
        "currentHarvesterCommunicationFailureTwo", "communicationFailure"
    ],
    "curtain": [
        "openFailure", "closeFailure", "remoteControlFailure"
    ],
    "electrical_transformer": [
        "highTemperaturePhaseA", "highTemperaturePhaseB", "highTemperaturePhaseC",
        "highTemperaturePhaseSingle", "fanFailure", "communicationFailure",
        "overload", "insulationFailure"
    ],
    "elevator": [
        "doorFailure", "cableFailure", "communicationError", "weightOverload",
        "highTemperature", "highVoltage", "lowVoltage", "highCurrent",
        "lowCurrent", "highPower", "emergencyStop"
    ],
    "firefighting": [
        "highCO2", "ventilationFailure", "fireEscapeBlocked",
        "emergencyBroadcastFailure", "sprinklerValveFailure", "fireDamperFailure"
    ],
    "hvac_airhandler": [
        "exhaustFanFailure", "highCO2", "highOutputHumidity", "lowOutputHumidity",
        "vfdSpeedFailure", "temperatureFailure", "waterValveFailure",
        "scheduleFailure"
    ],
    "hvac_cooling": [
        "highPower", "highOutputWaterTemperature", "lowOutputWaterTemperature",
        "valveCloseFailure", "valveOpenFailure", "highWaterLoad", "chillerFaultLockout",
        "lowFlowRate", "highFlowRate"
    ],
    "lighting": [
        "lampFailure", "driverFailure", "powerFailure", "lampFlickering"
    ],
    "security_camera": [
        "lensFailure", "powerFailure", "dustContamination",
        "networkFailure", "storageFailure"
    ],
    "security_patrol": [
        "communicationFailure", "responseDelay", "equipmentFailure"
    ],
    "water_pump": [
        "highOutputPressure", "lowOutputPressure", "highOutputTemperature",
        "lowOutputTemperature", "highFlowRate", "lowFlowRate",
        "highMotorFrequency", "lowMotorFrequency"
    ],
    "water_tank": [
        "leakage", "highTemperature", "lowTemperature", "overflow"
    ],
    "water_valve": [
        "openFailure", "closeFailure", "remoteControlFailure"
    ],
    "restroom_paper": [
        "lowPaperLevel", "paperJam", "dispenserFailure", "motorFailure", "sensorBlockage"
    ],
    "restroom_soap": [
        "lowSoapLevel", "dispenserClogged", "pumpFailure", "leakage",
        "sensorFailure", "containerDisconnected", "contaminationDetected"
    ],
    "restroom_toilet": [
        "blockage", "overflowRisk", "flushFailure", "leakage",
        "sensorFailure", "highWaterPressure", "lowWaterPressure", "seatSensorMalfunction"
    ],
    "environment": [
        "highPM2", "lowPM2", "highPM10", "lowPM10", "highHumidity",
        "lowHumidity", "highTemperature", "lowTemperature", "highH2S",
        "lowH2S", "highNH3", "lowNH3", "highCO2", "lowCO2", "highTVOC",
        "lowTVOC", "highCH2O", "lowCH2O", "particleFilterBlocked",
        "communicationFailure", "batteryLow"
    ]
}

def build_sql_conditions(relevant_columns, item, start_time, end_time, database, level):
    error_column = get_error_column(database)
    error_column = map_column_name(error_column)
    sql_conditions = []
    threshold_results = {}
    having_conditions = []
    needs_log_join = False

    for column in relevant_columns:
        if level == "device" and column in item["parameters"]["tables"]["parameters"]:
            description = item["parameters"]["tables"]["parameters"][column]
        else:
            description = item["parameters"].get(column)

        if not description:
            continue

        preposition_index = description.find(' for ') if ' for ' in description else description.find(' from ')
        condition = description[:preposition_index]
        alias = condition.split('_')[0]
        title = format_title(condition)
        unit = get_unit(condition)

        if level in ["group", "geozone", "project"]:
            if condition == 'device_count':
                subquery = f"""
                COUNT(DISTINCT l.deviceid) AS "{title}"
                """
                sql_conditions.append(subquery)
                needs_log_join = True
            elif condition.endswith('_device_count'):
                subquery = f"""
                COUNT(DISTINCT CASE 
                    WHEN ei.{error_column} = '{alias}' AND ei.startendflag = false 
                    THEN ei.deviceid 
                END) AS "{title}"
                """
                sql_conditions.append(subquery)
            elif condition.endswith('_count') or condition.endswith('_sum'):
                subquery = f"""
                COUNT(CASE 
                    WHEN ei.{error_column} = '{alias}' AND ei.startendflag = false 
                    THEN 1 
                END) AS "{title}"
                """
                sql_conditions.append(subquery)
            elif condition.endswith('_start') or condition.endswith('_end'):
                start_end_flag = 'false' if condition.endswith('_start') else 'true'
                subquery = f"""
                MAX(CASE 
                    WHEN ei.{error_column} = '{alias}' AND ei.startendflag = {start_end_flag}
                    THEN ei.eventtime
                END) AS "{title}"
                """
                sql_conditions.append(subquery)
            elif condition.endswith('_rate'):
                subquery = f"""
                ROUND(CAST(
                    COUNT(CASE 
                        WHEN ei.{error_column} = '{alias}' AND ei.startendflag = false 
                        THEN 1 
                    END) AS FLOAT) * 100.0 / 
                    NULLIF(COUNT(DISTINCT l.deviceid), 0)
                , 2) AS "{title}"
                """
                sql_conditions.append(subquery)
                needs_log_join = True
        else:  # Device level
            if condition == 'device_count':
                sql_conditions.append(f'COUNT(DISTINCT l.deviceid) AS "{title}"')
                needs_log_join = True
            elif condition.endswith('_count'):
                count_expr = f"""
                COUNT(CASE 
                    WHEN ei.{error_column} = '{alias}' AND ei.startendflag = false 
                    THEN 1 
                END) AS "{title}"
                """
                sql_conditions.append(count_expr)
                if level == "device":
                    having_conditions.append(f'COUNT(CASE WHEN ei.{error_column} = \'{alias}\' AND ei.startendflag = false THEN 1 END) > 0')
            elif condition.endswith('_start') or condition.endswith('_end'):
                start_end_flag = 'false' if condition.endswith('_start') else 'true'
                time_expr = f"""
                MAX(CASE 
                    WHEN ei.{error_column} = '{alias}' AND ei.startendflag = {start_end_flag}
                    THEN ei.eventtime 
                END) AS "{title}"
                """
                sql_conditions.append(time_expr)
                having_conditions.append(
                    f'MAX(CASE WHEN ei.{error_column} = \'{alias}\' AND ei.startendflag = {start_end_flag} THEN ei.eventtime END) IS NOT NULL'
                )

    return sql_conditions, threshold_results, having_conditions, needs_log_join



#If project and device levels require significantly different logic, consider 
#adding helper functions, e.g., handle_project_level_conditions and handle_device_level_conditions.


def handle_streetlighting_conditions(condition, alias, title, sql_conditions, threshold_results, having_conditions, start_time, end_time, level, database, needs_log_join):
    if condition.startswith('node_failure'):
        sql_conditions, having_conditions = handle_node_failure_conditions(condition, title, sql_conditions, having_conditions, level, start_time, end_time)
    elif condition.endswith('_threshold'):
        threshold_results = handle_threshold_condition(alias, title, threshold_results, database)
    elif condition.startswith('energy_consumption'):
        sql_conditions.append(f"COALESCE(SUM(log.energyConsumption), 0) AS `{title}`")
        needs_log_join = True
    elif condition in ["lights_on_time", "lights_off_time", "power_on_time", "power_off_time"]:
        sql_conditions, needs_log_join = handle_time_conditions(condition, title, sql_conditions)
    elif condition in ["total_running_hours", "remaining_lifetime"]:
        sql_conditions, needs_log_join = handle_lifetime_conditions(condition, title, sql_conditions, start_time, end_time)
    return sql_conditions, threshold_results, having_conditions, needs_log_join

def handle_time_conditions(condition, title, sql_conditions):
    column = "relayStatus" if condition.startswith("lights") else "power"
    value = "1" if condition.endswith("on_time") else "0"
    operator = "=" if condition.startswith("lights") else ("<>" if condition.endswith("on_time") else "=")
    sql_conditions.append(f"MAX(CASE WHEN log.{column} {operator} {value} THEN log.timeStamp END) AS `{title}`")
    return sql_conditions, True

def handle_lifetime_conditions(condition, title, sql_conditions, start_time, end_time):
    operation = "" if condition == "total_running_hours" else "50000 - "
    sql_conditions.append(f"""
    {operation}(SELECT totalLightingHours
    FROM log AS l1
    WHERE l1.deviceId = log.deviceId
    AND l1.timeStamp = (
        SELECT MAX(timeStamp)
        FROM log AS l2
        WHERE l2.deviceId = l1.deviceId
            AND l2.timeStamp BETWEEN '{start_time}' AND '{end_time}'
    )
    ) AS `{title}`
    """)
    return sql_conditions, True

def handle_building_conditions(condition, alias, title, sql_conditions, threshold_results, having_conditions, start_time, end_time, level, database, needs_log_join):
    if condition.endswith('_threshold'):
        threshold_results = handle_threshold_condition(alias, title, threshold_results, database)
    elif condition in ["energy_consumption", "power_usage"]:
        sql_conditions.append(f"COALESCE(SUM(log.{alias}), 0) AS `{title}`")
        needs_log_join = True
    # Add other building-specific conditions as needed
    return sql_conditions, threshold_results, having_conditions, needs_log_join

def handle_node_failure_conditions(condition, title, sql_conditions, having_conditions, level, start_time, end_time):
    if level in ["group", "geozone", "project"]:
        if condition.endswith('_device_count'):
            subquery = f"""
            (SELECT COUNT(DISTINCT ei.deviceId)
             FROM error_internal ei
             WHERE ei.controllerFault IS NOT NULL
               AND ei.startEndFlag = 0
               AND ei.eventTime BETWEEN '{start_time}' AND '{end_time}'
               AND LOCATE(CONCAT('/', c.groupId, '/'), CONCAT('/', ei.groupIdPath, '/')) > 0
            ) AS `{title}`
            """
            sql_conditions.append(subquery)
        elif condition.endswith('_count'):
            subquery = f"""
            (SELECT COUNT(*)
             FROM error_internal ei
             WHERE ei.controllerFault IS NOT NULL
               AND ei.startEndFlag = 0
               AND ei.eventTime BETWEEN '{start_time}' AND '{end_time}'
               AND LOCATE(CONCAT('/', c.groupId, '/'), CONCAT('/', ei.groupIdPath, '/')) > 0
            ) AS `{title}`
            """
            sql_conditions.append(subquery)
        elif condition.endswith('_start') or condition.endswith('_end'):
            start_end_flag = '0' if condition.endswith('_start') else '1'
            subquery = f"""
            (SELECT MAX(ei.eventTime)
             FROM error_internal ei
             WHERE ei.controllerFault IS NOT NULL
               AND ei.startEndFlag = {start_end_flag}
               AND ei.eventTime BETWEEN '{start_time}' AND '{end_time}'
               AND LOCATE(CONCAT('/', c.groupId, '/'), CONCAT('/', ei.groupIdPath, '/')) > 0
            ) AS `{title}`
            """
            sql_conditions.append(subquery)
        elif condition.endswith('_rate'):
            subquery = f"""
            (CAST(
                (SELECT COUNT(*)
                 FROM error_internal ei
                 WHERE ei.controllerFault IS NOT NULL
                   AND ei.startEndFlag = 0
                   AND ei.eventTime BETWEEN '{start_time}' AND '{end_time}'
                   AND LOCATE(CONCAT('/', c.groupId, '/'), CONCAT('/', ei.groupIdPath, '/')) > 0
                ) AS FLOAT) /
             NULLIF(
                (SELECT COUNT(DISTINCT l.deviceId)
                 FROM log l
                 WHERE l.timeStamp BETWEEN '{start_time}' AND '{end_time}'
                   AND LOCATE(CONCAT('/', c.groupId, '/'), CONCAT('/', l.groupIdPath, '/')) > 0
                ), 0
             )
            ) * 100 AS `{title}`
            """
            sql_conditions.append(subquery)
    elif level == "device":
        if condition.endswith('_start') or condition.endswith('_end'):
            start_end_flag = '0' if condition.endswith('_start') else '1'
            sql_conditions.append(f"MAX(CASE WHEN error_internal.controllerFault IS NOT NULL AND error_internal.startEndFlag = {start_end_flag} THEN error_internal.eventTime END) AS `{title}`")
            having_conditions.append(f"`{title}` IS NOT NULL")
        else:
            sql_conditions.append(f"CAST(SUM(CASE WHEN error_internal.controllerFault IS NOT NULL AND error_internal.startEndFlag = 0 THEN 1 ELSE 0 END) AS SIGNED) AS `{title}`")
            having_conditions.append(f"`{title}` > 0")
    return sql_conditions, having_conditions

def handle_threshold_condition(alias, title, threshold_results, database):
    column_name = f"{alias}Threshold"
    threshold_query = f"SELECT {column_name} FROM log ORDER BY timeStamp DESC LIMIT 1"
    threshold_data = execute_query(threshold_query, database)
    if threshold_data:
        threshold_results[title] = threshold_data[0][column_name]
    return threshold_results

def handle_general_conditions(condition, alias, title, error_column, sql_conditions, having_conditions, level, start_time, end_time, database):
    if level in ["group", "geozone", "project"]:
        if condition == 'device_count':
            # Generate subquery for device count
            subquery = f"""
            (SELECT COUNT(DISTINCT l.deviceId)
             FROM log l
             WHERE l.timeStamp BETWEEN '{start_time}' AND '{end_time}'
               AND LOCATE(CONCAT('/', c.groupId, '/'), CONCAT('/', l.groupIdPath, '/')) > 0
            ) AS `{title}`
            """
            sql_conditions.append(subquery)
        elif condition.endswith('_device_count'):
            # Generate subquery for error_internal device count
            subquery = f"""
            (SELECT COUNT(DISTINCT ei.deviceId)
             FROM error_internal ei
             WHERE ei.{error_column} = '{alias}'
               AND ei.startEndFlag = 0
               AND ei.eventTime BETWEEN '{start_time}' AND '{end_time}'
               AND LOCATE(CONCAT('/', c.groupId, '/'), CONCAT('/', ei.groupIdPath, '/')) > 0
            ) AS `{title}`
            """
            sql_conditions.append(subquery)
        elif condition.endswith('_count') or condition.endswith('_sum'):
            # Generate subquery for error count
            subquery = f"""
            (SELECT COUNT(*)
             FROM error_internal ei
             WHERE ei.{error_column} = '{alias}'
               AND ei.startEndFlag = 0
               AND ei.eventTime BETWEEN '{start_time}' AND '{end_time}'
               AND LOCATE(CONCAT('/', c.groupId, '/'), CONCAT('/', ei.groupIdPath, '/')) > 0
            ) AS `{title}`
            """
            sql_conditions.append(subquery)
        elif condition.endswith('_start') or condition.endswith('_end'):
            # Generate subquery for start/end times
            start_end_flag = '0' if condition.endswith('_start') else '1'
            subquery = f"""
            (SELECT MAX(ei.eventTime)
             FROM error_internal ei
             WHERE ei.{error_column} = '{alias}'
               AND ei.startEndFlag = {start_end_flag}
               AND ei.eventTime BETWEEN '{start_time}' AND '{end_time}'
               AND LOCATE(CONCAT('/', c.groupId, '/'), CONCAT('/', ei.groupIdPath, '/')) > 0
            ) AS `{title}`
            """
            sql_conditions.append(subquery)
        elif condition.endswith('_rate'):
            # Generate subquery for error rate
            subquery = f"""
            (CAST(
                (SELECT COUNT(*)
                 FROM error_internal ei
                 WHERE ei.{error_column} = '{alias}'
                   AND ei.startEndFlag = 0
                   AND ei.eventTime BETWEEN '{start_time}' AND '{end_time}'
                   AND LOCATE(CONCAT('/', c.groupId, '/'), CONCAT('/', ei.groupIdPath, '/')) > 0
                ) AS FLOAT) /
             NULLIF(
                (SELECT COUNT(DISTINCT l.deviceId)
                 FROM log l
                 WHERE l.timeStamp BETWEEN '{start_time}' AND '{end_time}'
                   AND LOCATE(CONCAT('/', c.groupId, '/'), CONCAT('/', l.groupIdPath, '/')) > 0
                ), 0
             )
            ) * 100 AS `{title}`
            """
            sql_conditions.append(subquery)
    else:
        # Device level or other databases
        if condition == 'device_count':
            sql_conditions.append(f"COUNT(DISTINCT log.deviceId) AS `Device Count`")
        elif condition.endswith('_device_count'):
            count_expression = f"COUNT(DISTINCT CASE WHEN error_internal.{error_column} = '{alias}' AND error_internal.startEndFlag = 0 THEN error_internal.deviceId END) AS `{title}`"
            sql_conditions.append(count_expression)
        elif condition.endswith('_count') or condition.endswith('_sum'):
            count_expression = f"COUNT(CASE WHEN error_internal.{error_column} = '{alias}' AND error_internal.startEndFlag = 0 THEN 1 END) AS `{title}`"
            sql_conditions.append(count_expression)
            if level == "device":
                having_conditions.append(f"`{title}` > 0")
        elif condition.endswith('_start') or condition.endswith('_end'):
            start_end_flag = '0' if condition.endswith('_start') else '1'
            sql_conditions.append(f"""
                MAX(CASE WHEN error_internal.{error_column} = '{alias}' AND error_internal.startEndFlag = {start_end_flag}
                    THEN error_internal.eventTime
                    ELSE NULL 
                END) AS `{title}`
            """)
            having_conditions.append(f"`{title}` IS NOT NULL")
        elif condition.endswith('_rate'):
            total_expression = f"SUM(CASE WHEN error_internal.{error_column} = '{alias}' AND error_internal.startEndFlag = 0 THEN 1 ELSE 0 END)"
            count_expression = f"(CAST({total_expression} AS FLOAT) / NULLIF(CAST(COUNT(DISTINCT log.deviceId) AS FLOAT), 0) * 100) AS `{title}`"
            sql_conditions.append(count_expression)
        elif condition is not None and level == "device":
            sql_conditions.append(f"MAX(CASE WHEN error_internal.{error_column} = '{alias}' AND error_internal.startEndFlag = 0 THEN 1 ELSE 0 END) AS `{title}`")
            having_conditions.append(f"`{title}` = 1")
    return sql_conditions, having_conditions

#note, now the query is hardcoded. 
def build_sql_query(sql_conditions, start_time, end_time, having_clause, database, level, needs_log_join, selected_groups=None):
    select_conditions = ', '.join(sql_conditions).strip()

    if level in ["group", "geozone", "project"]:
        sql_select = f"""
            SELECT 
                d.groupidpath AS "Group ID Path",
                COUNT(DISTINCT l.deviceid) AS "Device Count",
                COUNT(DISTINCT CASE 
                    WHEN ei.controllerfault = 'powerOutage' AND ei.startendflag = false 
                    THEN ei.deviceid 
                END) AS "Power Outage Device Count",
                COUNT(CASE 
                    WHEN ei.controllerfault = 'powerOutage' AND ei.startendflag = false 
                    THEN 1 
                END) AS "Power Outage Count",
                COUNT(DISTINCT CASE 
                    WHEN ei.controllerfault = 'lux' AND ei.startendflag = false 
                    THEN ei.deviceid 
                END) AS "Lux Communication Error Device Count",
                COUNT(CASE 
                    WHEN ei.controllerfault = 'lowVoltage' AND ei.startendflag = false 
                    THEN 1 
                END) AS "Low Voltage Count",
                COUNT(DISTINCT CASE 
                    WHEN ei.controllerfault = 'highTemperature' AND ei.startendflag = false 
                    THEN ei.deviceid 
                END) AS "High Temperature Device Count",
                COUNT(CASE 
                    WHEN ei.controllerfault = 'driverCommunicationFault' AND ei.startendflag = false 
                    THEN 1 
                END) AS "Driver Communication Fault Count",
                CAST(
                    (CAST(COUNT(CASE 
                        WHEN ei.controllerfault = 'lux' AND ei.startendflag = false 
                        THEN 1 
                    END) AS FLOAT) * 100.0) / 
                    NULLIF(COUNT(DISTINCT l.deviceid), 0)
                AS FLOAT) AS "Lux Communication Error Rate"
        """
        
        from_clause = f"""
            FROM device d
            JOIN log l ON d.deviceid = l.deviceid
            LEFT JOIN error_internal ei 
                ON l.deviceid = ei.deviceid 
                AND ei.eventtime BETWEEN {start_time} AND {end_time}
        """

        where_clause = f"WHERE l.timestamp >= {start_time} AND l.timestamp < {end_time}"
        if selected_groups:
            group_list = ', '.join([f"'{group}'" for group in selected_groups])
            where_clause += f" AND d.groupidpath IN ({group_list})"
        
        group_by_clause = "GROUP BY d.groupidpath"
        order_by_clause = "ORDER BY d.groupidpath"

        query = f"""
            {sql_select}
            {from_clause}
            {where_clause}
            {group_by_clause}
            {order_by_clause}
        """
        return query.strip()

    elif level == "device":
        select_clause = f"""
            SELECT l.deviceid AS "Device ID", 
                COUNT(CASE 
                    WHEN ei.controllerfault = 'flickering' AND ei.startendflag = false 
                    THEN 1 
                END) AS "Flickering Count",
                MAX(CASE 
                    WHEN ei.controllerfault = 'flickering' AND ei.startendflag = false
                    THEN ei.eventtime 
                END) AS "Flickering Start",
                MAX(CASE 
                    WHEN ei.controllerfault = 'flickering' AND ei.startendflag = true
                    THEN ei.eventtime 
                END) AS "Flickering End",
                MAX(CASE 
                    WHEN ei.controllerfault = 'powerOutage' AND ei.startendflag = false
                    THEN ei.eventtime 
                END) AS "Power Outage Start",
                MAX(CASE 
                    WHEN ei.controllerfault = 'powerOutage' AND ei.startendflag = true
                    THEN ei.eventtime 
                END) AS "Power Outage End",
                MAX(CASE 
                    WHEN ei.controllerfault = 'lowVoltage' AND ei.startendflag = false
                    THEN ei.eventtime 
                END) AS "Low Voltage Start",
                MAX(CASE 
                    WHEN ei.controllerfault = 'highVoltage' AND ei.startendflag = false
                    THEN ei.eventtime 
                END) AS "High Voltage Start",
                MAX(CASE 
                    WHEN ei.controllerfault = 'driverCommunicationFault' AND ei.startendflag = false
                    THEN ei.eventtime 
                END) AS "Driver Communication Fault Start"
        """

        from_clause = f"""
            FROM log l
            LEFT JOIN error_internal ei 
                ON l.deviceid = ei.deviceid 
                AND ei.eventtime BETWEEN {start_time} AND {end_time}
        """

        where_clause = f"WHERE l.timestamp >= {start_time} AND l.timestamp < {end_time}"
        group_by_clause = "GROUP BY l.deviceid"
        # EXISTING: Power outage already included in having clause
        having_clause = """
            HAVING (
                MAX(CASE WHEN ei.controllerfault IN ('lowVoltage', 'highVoltage', 'driverCommunicationFault', 'flickering', 'powerOutage') 
                    AND ei.startendflag = false THEN ei.eventtime END) IS NOT NULL
            )
        """

        query = f"""
            {select_clause}
            {from_clause}
            {where_clause}
            {group_by_clause}
            {having_clause}
            ORDER BY "Device ID"
        """
        return query.strip()

    elif level in ["street", "geozone"]:
        summary_name = "Street" if level == "street" else "GeoZone"
        # UPDATED: Added power outage counts to sql_select
        sql_select = f"""
            SELECT 
                d.groupname AS "{summary_name}",
                COUNT(DISTINCT l.deviceid) AS "Device Count",
                COUNT(DISTINCT CASE 
                    WHEN ei_flicker.controllerfault = 'flickering' AND ei_flicker.startendflag = false 
                    THEN ei_flicker.deviceid 
                END) AS "Flickering Device Count",
                COUNT(CASE 
                    WHEN ei_flicker.controllerfault = 'flickering' AND ei_flicker.startendflag = false 
                    THEN 1 
                END) AS "Flickering Count",
                COUNT(DISTINCT CASE 
                    WHEN ei_power.controllerfault = 'powerOutage' AND ei_power.startendflag = false 
                    THEN ei_power.deviceid 
                END) AS "Power Outage Device Count",
                COUNT(CASE 
                    WHEN ei_power.controllerfault = 'powerOutage' AND ei_power.startendflag = false 
                    THEN 1 
                END) AS "Power Outage Count"
        """

        from_clause = f"""
            FROM device d
            JOIN log l ON d.deviceid = l.deviceid
            LEFT JOIN error_internal ei_flicker
                ON l.deviceid = ei_flicker.deviceid 
                AND ei_flicker.eventtime BETWEEN {start_time} AND {end_time}
                AND ei_flicker.controllerfault = 'flickering'
            LEFT JOIN error_internal ei_power
                ON l.deviceid = ei_power.deviceid 
                AND ei_power.eventtime BETWEEN {start_time} AND {end_time}
                AND ei_power.controllerfault = 'powerOutage'
        """

        where_clause = f"WHERE l.timestamp >= {start_time} AND l.timestamp < {end_time}"
        group_by_clause = f'GROUP BY d.groupname'
        order_by_clause = 'ORDER BY d.groupname'

        query = f"""
            {sql_select}
            {from_clause}
            {where_clause}
            {group_by_clause}
            ORDER BY COALESCE(d.groupname, 'Unknown')
        """
        return query.strip()

    else:
        raise ValueError(f"Unsupported level: {level}")



def generic_level(relevant_columns, item, start_time, end_time, type_val, database, level, selected_groups=None):
    print(f"DEBUG: Entering generic_level function for level: {level}")
    sql_conditions, threshold_results, having_conditions, needs_log_join = build_sql_conditions(relevant_columns, item, start_time, end_time, database, level)

    having_clause = ""
    if having_conditions:
        having_conditions = [cond for cond in having_conditions if cond.strip() != '']
        having_clause = f"HAVING {' OR '.join(having_conditions)}" if having_conditions else ""

    sql_query = build_sql_query(sql_conditions, start_time, end_time, having_clause, database, level, needs_log_join, selected_groups)
    print("DEBUG: SQL query:", sql_query)
    query_results = execute_query(sql_query, database)
    processed_results = process_query_to_datetime(query_results)
    print("DEBUG: Processed results:", processed_results)

    if threshold_results:
        for result in processed_results:
            result.update(threshold_results)

    # Update keys for consistency
    for result in processed_results:
        if 'groupName' in result:
            result['Group Name'] = result.pop('groupName')
        if 'Device count' in result:
            result['Device Count'] = result.pop('Device count')
        if 'deviceId' in result:
            result['Device ID'] = result.pop('deviceId')

    # Exclude entries with all zero counts (except name and device count)
    excluded_keys = ['Group Name', 'Total', 'Device Count', 'Device ID', 'Street', 'GeoZone']
    filtered_results = [
        result for result in processed_results
        if any(
            (isinstance(value, (int, float)) and value != 0) or
            (isinstance(value, str) and value.strip() and not value.lower() == 'none')
            for key, value in result.items()
            if key not in excluded_keys and 'threshold' not in key.lower()
        )
    ]
    print("DEBUG: Filtered results:", filtered_results)

    return_value = {
        'function': 'table',
        'parameters': [
            {type_val: ""},
            {f"Start Time: {convert_ms_to_datetime(start_time)}": "", f"End Time: {convert_ms_to_datetime(end_time)}": ""},
            *filtered_results
        ]
    }
    print("DEBUG: Returning from generic_level:", return_value)
    return return_value

def report_to_table(data, database="govchat_streetlighting"):
    print("DEBUG: Entering report_to_table function")
    new_data = []

    for item in data:
        print("DEBUG: Processing item:", item)
        start_time = item["parameters"]["start_time"]
        end_time = item["parameters"]["end_time"]
        if start_time > end_time:
            temp = start_time
            start_time = end_time
            end_time = temp
        type_val = item["parameters"]["type"]

        level_map = {
            "all projects": "project",
            "all geozones": "geozone",
            "all groups": "group",
            "all streets": "street",
            "all devices": "device"
        }
        
        column1 = item["parameters"]["column1"].lower()
        if column1 in level_map:
            level = level_map[column1]
            selected_groups = None
        elif column1.startswith("groups"):
            level = "group"
            groups_str = column1.split("=")[1].strip()
            # Remove square brackets and split by comma
            selected_groups = [group.strip().strip("'\"") for group in groups_str.strip('[]').split(',')]
        else:
            level = "group"
            selected_groups = None
        
        print(f"DEBUG: Processing level: {level}, selected_groups: {selected_groups}")

        relevant_columns = [key for key in item["parameters"] if key.startswith('column') and key[6:].isdigit()]
        print("DEBUG: Relevant columns for first level:", relevant_columns)
        first_level_table = generic_level(relevant_columns, item, start_time, end_time, type_val, database, level, selected_groups)
        print("DEBUG: First level table:", first_level_table)
        new_data.append(first_level_table)
        
        device_relevant_columns = [key for key in item["parameters"]["tables"]["parameters"] if key.startswith('column') and key[6:].isdigit()]
        print("DEBUG: Relevant columns for device level:", device_relevant_columns)
        device_level_table = generic_level(device_relevant_columns, item, start_time, end_time, type_val, database, "device")
        print("DEBUG: Device level table:", device_level_table)
        new_data.append(device_level_table)

    print("DEBUG: Final new_data:", new_data)
    return new_data


###################Below is for testing########################
# Sample JSON data
data1 = [{
    "function": "report",
    "parameters": {
        "type": "power_outage",
        "start_time": 1575093600000,
        "end_time": 1675093600000,
        "column1": "groups = [group31, group42, group53, group64, project1]",
        "column2": "device_count for column1",
        "column3": "powerOutage_device_count for column1",
        "column4": "powerOutage_count for column1",
        "tables": {
            "index": "for column1",
            "parameters": {
                "start_time": 1575093600000,
                "end_time": 1675093600000,
                "column5": "all devices",
                "column6": "powerOutage_time for column5"
            }
        }
    }
}]

data3 = [{
    "function": "report",
    "parameters": {
        "type": "power_outage",
        "start_time": 1575093600000,
        "end_time": 1675093600000,
        "column1": "all projects",
        "column2": "device_count for column1",
        "column3": "powerOutage_device_count for column1",
        "column4": "powerOutage_count for column1",
        "tables": {
            "index": "for column1",
            "parameters": {
                "start_time": 1575093600000,
                "end_time": 1675093600000,
                "column5": "all devices",
                "column6": "powerOutage_time for column5"
            }
        }
    }
}]

data2 = [{
    "function": "streetlight_data_report",
    "parameters": {
        "type": "node_failure",
        "start_time": 1575093600000,
        "end_time": 1701324000000,
        "column1": "all projects",
        "column2": "device_count for column1",
        "column3": "nodeFailure_device_count for column1",
        "column4": "nodeFailure_count for column1",
        "column5": "highVoltage_device_count for column1",
        "column6": "lowVoltage_device_count for column1",
        "column7": "highPower_device_count for column1",
        "column8": "lowPower_device_count for column1",
        "column9": "lowPF_device_count for column1",
        "column10": "highCurrent_device_count for column1",
        "column11": "lowCurrent_device_count for column1",
        "column12": "highTemperature_device_count for column1",
        "column13": "meterFault_device_count for column1",
        "column14": "luxModuleFault_device_count for column1",
        "column15": "driverCommunicationFault_device_count for column1",
        "column16": "lightFault_device_count for column1",
        "column17": "flickering_device_count for column1",
        "column18": "dayBurn_device_count for column1",
        "column19": "relaySticking_device_count for column1",
        "column20": "relayOpenCircuit_device_count for column1",
        "column21": "noNetwork_device_count for column1",
        "column22": "gpsFailure_device_count for column1",
        "tables": {
            "index": "for column1",
            "parameters": {
                "start_time": 1575093600000,
                "end_time": 1701324000000,
                "column23": "all devices",
                "column24": "nodeFailure_count for column23",
                "column25": "highVoltage for column23",
                "column26": "lowVoltage for column23",
                "column27": "highPower for column23",
                "column28": "lowPower for column23",
                "column29": "lowPower_factor for column23",
                "column30": "highCurrent for column23",
                "column31": "lowCurrent for column23",
                "column32": "highTemperature for column23",
                "column33": "meterFault for column23",
                "column34": "luxModuleFault for column23",
                "column35": "driverCommunicationFault for column23",
                "column36": "lightFault for column23",
                "column37": "flickering for column23",
                "column38": "dayBurn for column23",
                "column39": "relaySticking for column23",
                "column40": "relayOpenCircuit for column23",
                "column41": "noNetwork for column23",
                "column42": "gpsFailure for column23",
                "column43": "nodeFailure_start for column23",
                "column44": "nodeFailure_end for column23",
                "column45": "highLoadCurrent_rate for column23"
            }
        }
    }   
}]
if __name__ == '__main__':
    print("Final Table: \n", report_to_table(data1)) 
    # print("Final Table: \n", report_to_table(data2)) 