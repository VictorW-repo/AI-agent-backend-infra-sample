from datetime import datetime 
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError, DataError
from query_api import llm_api
import re
from typing import List, Dict, Union, Any, Tuple, Optional
from collections import Counter
import logging

import sys
import os
sys.path.append('../config')
from config_api import get_db_uri


# Configure logging
logging.basicConfig(level=logging.INFO)

def execute_query_postgres(query: str, database: str, parameters: Optional[Dict] = None) -> List[Dict]:
    """
    Execute an SQL query on PostgreSQL database and return results as a list of dictionaries.
    
    Parameters:
        query (str): The SQL query to execute
        parameters (dict, optional): Parameters for prepared statements
    
    Returns:
        List[Dict]: Query results as a list of dictionaries
    """
    connection_uri = get_db_uri(database)
    engine = create_engine(
        connection_uri,
        pool_pre_ping=True,  # Enable connection health checks
        pool_size=5,         # Set connection pool size
        max_overflow=10      # Maximum number of connections to overflow
    )
    
    try:
        with engine.connect() as connection:
            # Execute query with or without parameters
            result = connection.execute(text(query), parameters) if parameters else connection.execute(text(query))
            
            # Convert results to dictionary format
            result_list = [dict(row._mapping) for row in result]
            
            # Handle empty result set
            if not result_list:
                return ["No streetlighting data available"]
                
            return result_list
            
    except ProgrammingError as e:
        logging.error(f"PostgreSQL Programming Error: {e}")
        return [f"Invalid query or database structure: {str(e)}"]
        
    except DataError as e:
        logging.error(f"PostgreSQL Data Error: {e}")
        return [f"Data type mismatch or invalid data: {str(e)}"]
        
    except SQLAlchemyError as e:
        logging.error(f"SQLAlchemy Error: {e}")
        return [f"Database connection or execution error: {str(e)}"]
        
    except Exception as e:
        logging.error(f"Unexpected error in PostgreSQL query: {e}")
        return [f"An unexpected error occurred: {str(e)}"]
        
    finally:
        engine.dispose()

def convert_mysql_to_postgresql(query):
    """
    Enhanced MySQL to PostgreSQL query converter that preserves query semantics:
    • Adds aggregations only when GROUP BY is present
    • Preserves original column references when no GROUP BY
    • Converts MySQL functions to PostgreSQL equivalents
    """
    # Store original query for debugging
    original_query = query
    
    # Remove any trailing semicolon for processing
    has_semicolon = query.rstrip().endswith(';')
    query = query.rstrip(';')
    
    # Convert MySQL backtick quotes to PostgreSQL double quotes
    quote_pattern = r'`([^`]+)`'
    query = re.sub(quote_pattern, r'"\1"', query)
    
    def get_aliases(select_expr):
        """Extract column aliases from SELECT expressions"""
        aliases = set()
        for expr in re.split(r',(?![^(]*\))', select_expr):
            if ' as ' in expr.lower():
                alias = expr.split(' as ')[-1].strip()
                aliases.add(alias)
        return aliases
    
    # Split query into SELECT part and the rest
    select_pattern = r'^SELECT\s+(.*?)\s+FROM\s+(.*)$'
    match = re.match(select_pattern, query, re.IGNORECASE | re.DOTALL)
    
    if match:
        select_columns = match.group(1).strip()
        rest_of_query = match.group(2)
        
        # Check if GROUP BY exists in the query
        has_group_by = bool(re.search(r'\bGROUP BY\b', rest_of_query, re.IGNORECASE))
        
        # Extract GROUP BY columns if present
        group_by_match = re.search(r'GROUP BY\s+(.*?)(?:HAVING|ORDER BY|LIMIT|$)', rest_of_query, re.IGNORECASE | re.DOTALL)
        group_cols = set()
        if group_by_match:
            group_cols = {col.strip() for col in group_by_match.group(1).split(',')}
        
        # Process SELECT columns
        select_parts = re.split(r',(?![^(]*\))', select_columns)
        new_select_cols = []
        
        for col in select_parts:
            col = col.strip()
            
            if not has_group_by:
                # No GROUP BY - keep original column references
                if 'CASE' in col.upper():
                    # Just convert TRUE/FALSE strings to PostgreSQL booleans
                    col = col.replace("'true'", "'TRUE'").replace("'false'", "'FALSE'")
                new_select_cols.append(col)
                continue
                
            needs_agg = True
            
            # Check if column is in GROUP BY
            for group_col in group_cols:
                if group_col.strip() in col:
                    needs_agg = False
                    break
            
            # Don't aggregate if already an aggregate function
            if any(agg in col.lower() for agg in ['sum(', 'count(', 'avg(', 'min(', 'max(', 'string_agg(']):
                needs_agg = False
            
            if needs_agg:
                # Handle CASE expressions
                if 'CASE' in col.upper():
                    if ' as ' in col:
                        base, alias = col.split(' as ', 1)
                        new_col = f"MAX(CASE WHEN POSITION('powerOutage' IN l.alarmFault) > 0 THEN 1 ELSE 0 END::int) = 1 as {alias}"
                    else:
                        new_col = f"MAX(CASE WHEN POSITION('powerOutage' IN l.alarmFault) > 0 THEN 1 ELSE 0 END::int) = 1"
                # Handle timestamps
                elif 'timestamp' in col.lower():
                    if ' as ' in col.lower():
                        base, alias = col.split(' as ', 1)
                        new_col = f"MAX({base}) as {alias}"
                    else:
                        new_col = f"MAX({col}) as event_time"  # Just add alias to MAX
                else:
                    if ' as ' in col.lower():
                        base, alias = col.split(' as ', 1)
                        new_col = f"MAX({base}) as {alias}"
                    else:
                        new_col = f"MAX({col})"
            else:
                new_col = col
                
            new_select_cols.append(new_col)
        
        # Rebuild query
        query = f"SELECT {', '.join(new_select_cols)} FROM {rest_of_query}"
    
    # Convert MySQL functions to PostgreSQL
    replacements = {
        r'LOCATE\((.*?),\s*(.*?)\)': r"POSITION(\1 IN \2)",
        r'GROUP_CONCAT\((.*?)\)': r"STRING_AGG(\1, ',')",
        r'IFNULL\((.*?),\s*(.*?)\)': r"COALESCE(\1, \2)",
        r'CONCAT\((.*?)\)': lambda m: ' || '.join(m.group(1).split(','))
    }
    
    # Apply replacements
    for pattern, replacement in replacements.items():
        if callable(replacement):
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
        else:
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    
    # Handle MySQL-style true/false
    query = re.sub(r'\btrue\b', 'TRUE', query, flags=re.IGNORECASE)
    query = re.sub(r'\bfalse\b', 'FALSE', query, flags=re.IGNORECASE)
    
    # Add back semicolon if it was present
    if has_semicolon:
        query += ';'
    
    return query

def format_key(key: str) -> str:
    """
    Format a key by splitting underscores and capitalizing words.
    
    Parameters:
        key (str): The key to format.
    
    Returns:
        str: Formatted key.
    """
    # Handle specific cases where we want to retain the original key
    if key.lower() in ['timestamp', 'stamp', 'time stamp']:
        return 'timeStamp'
    
    words = key.split('_')
    split_words = []
    for word in words:
        split_words.extend(re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]\d|\d|\W|$)|\d+', word))
    return ' '.join(word.capitalize() for word in split_words)

def convert_to_serializable(obj: Any) -> Any:
    """
    Convert an object to a serializable format.
    
    Parameters:
        obj (Any): The object to convert.
    
    Returns:
        Any: Serializable object.
    """
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)

def query_internal_databases(input_string: str, vertical: str = "streetlighting", max_retries: int = 3) -> Dict[str, Union[List[Dict], str]]:
    """
    Query internal databases using LLM-generated SQL queries.
    
    Parameters:
        input_string (str): User input string to generate SQL queries.
        vertical (str, optional): The vertical/domain to query. Defaults to "streetlighting".
        max_retries (int, optional): Maximum number of retries for failed queries. Defaults to 3.
    
    Returns:
        Dict[str, Union[List[Dict], str]]: Results mapped to the original input string.
    """
    database = "govchat_" + vertical
    original_input = input_string  # Preserve the original input

    for attempt in range(max_retries):
        # Modify the prompt to instruct the LLM to include 'timeStamp' in the SELECT clause
        conversation = [
            {
                "role": "user",
                "content": (
                    f"###System: You are a {vertical}_sql assistant ###User: {input_string}"
                )
            }
        ]
        sql_query_mysql = llm_api(conversation)
        print("###Generated MySQL Query: ", sql_query_mysql, "###")

        # Validate if 'timeStamp' is in the SELECT clause
        if 'SELECT' in sql_query_mysql.upper():
            select_clause = sql_query_mysql.upper().split('SELECT')[1].split('FROM')[0]
            if 'TIMESTAMP' not in select_clause:
                # Append 'timeStamp' to the SELECT clause
                columns = select_clause.strip().rstrip(',')
                columns += ', timeStamp'
                sql_query_mysql = sql_query_mysql.replace(select_clause, columns)
                print("###Modified SQL Query to include timeStamp: ", sql_query_mysql, "###")
        else:
            logging.error("Malformed SQL query: Missing SELECT clause.")
            return {original_input: "Invalid SQL query generated."}

        sql_query_postgres = convert_mysql_to_postgresql(sql_query_mysql)
        print("###Transformed PostgreSQL Query: ", sql_query_postgres, "###")
        results = execute_query_postgres(sql_query_postgres, database)

        if isinstance(results, list) and len(results) == 1 and isinstance(results[0], str):
        # Case 1: No data responses - return immediately without retry
            if results[0] in ["No IoT data available", "No streetlighting data available"]:
                return {
                    "status": "no_data",
                    "message": "No data matches your query criteria in the specified time range.",
                    "query": sql_query_postgres
                }
        # Case 2: Actual errors - modify input and continue loop for retry
            else:
                # Add back this else clause for actual errors
                input_string += f" The previous query resulted in an error: {results[0]}. Please provide a corrected SQL query."
        # Case 3: Successful results - return processed data
        else:
            # Successful query execution
            annotated_results = []
            for row in results:
                annotated_row = {}
                for key, value in row.items():
                    formatted_key = format_key(key)
                    if formatted_key.lower().replace(" ", "") in ['timestamp', 'stamp']:
                        # Normalize all variations to 'timeStamp'
                        formatted_key = 'timeStamp'
                        # Convert Unix timestamp (assuming milliseconds) or datetime string to readable datetime
                        try:
                            if isinstance(value, (int, float)):
                                # Value is an integer timestamp in milliseconds
                                annotated_row[formatted_key] = datetime.fromtimestamp(value / 1000).strftime('%Y-%m-%d %H:%M:%S')
                            elif isinstance(value, str):
                                # Value is a string; attempt to parse it
                                annotated_row[formatted_key] = datetime.strptime(value, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                # Unsupported type
                                annotated_row[formatted_key] = None
                                logging.error(f"Unsupported timeStamp value type: {type(value)}")
                        except Exception as e:
                            logging.error(f"Error converting timeStamp: {e}")
                            annotated_row[formatted_key] = None
                    else:
                        if isinstance(value, (int, float, datetime, str)):
                            annotated_row[formatted_key] = str(value)
                        elif value is None:
                            annotated_row[formatted_key] = None
                        else:
                            annotated_row[formatted_key] = convert_to_serializable(value)
                annotated_results.append(annotated_row)
            return {original_input: annotated_results}

    # After exhausting all retries
    return {original_input: ""}
    #return {original_input: "Maximum retries reached without success."}




def main():
    query = "Query for log.energyConsumption, log.totalLightingHours, log.endFlickeringThreshold for all log from all time to all time"
    vertical = "streetlighting"
    result = query_internal_databases(query, vertical)
    print(result)

if __name__ == '__main__':
    main()
