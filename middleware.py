from light_failure import streetlight_failure_predict
from energy_consumption_prediction import streetlight_energy_prediction
from RAG import classify_task
from handle_results import handle_failure_results, handle_consumption_results
from report_table_formatting import report_to_table
from process_sql import query_internal_databases
from internal_report import internal_report_streetlighting
from middleware_helper_functions import force_json, string_to_list_of_strings, convert_to_unix_time, sql_to_table, extract_summary_and_tokens, convert_query_timestamps_to_readable, downsize_sql_supporting_information, normalize_sql_return, clean_temporal_references, downsize_data_report_information, downsize_internal_report, format_data_report_table, format_internal_report_table
from query_api import llm_api#, sql_classifier_api
from trend_analysis import analyze_abnormal_behaviors, extract_and_fix_json, generate_summary, detect_time_column
import time
import json
import re
import warnings
from datetime import datetime
import pandas as pd
import sys
import traceback
from logger_setup import logger, log_function_call, log_request



warnings.filterwarnings('ignore')
from typing import Tuple, List, Dict, Any, Union, Optional  

import logging
logging.basicConfig(level=logging.INFO)



#############################################text streaming##################################################
import re
import json

def ask_with_streaming(chat):
    """
    Version of ask() that streams simulated thought process but returns final response all at once.
    """
    # First, extract the user query to classify it
    user_message = ""
    if 'messages' in chat and chat['messages']:
        for message in chat['messages']:
            if message.get('role') == 'user':
                user_message = message.get('content', '')
                break
    
    # Get classification - we need this to determine which thought process to show
    context, result_classes = classify_task(user_message)
    
    # Based on classification, stream appropriate thought process
    if "streetlight_sql" in result_classes:
        thought_steps = [
            "Converting your question to SQL query...",
            "Identifying relevant database columns...",
            "Generating optimized SQL query...",
            "Querying streetlight database...",
            "Processing database results...",
            "Analyzing data patterns and trends...",
            "Preparing response and visualization..."
        ]
    elif "streetlight_failure_predict" in result_classes:
        thought_steps = [
            "Analyzing failure prediction request...",
            "Accessing historical failure data...",
            "Loading prediction model...",
            "Running machine learning prediction algorithm...",
            "Calculating failure probabilities...",
            "Analyzing prediction confidence levels...",
            "Generating prediction report..."
        ]
    elif "streetlight_energy_consumption_predict" in result_classes:
        thought_steps = [
            "Processing energy consumption prediction request...",
            "Retrieving historical energy consumption data...",
            "Initializing energy forecasting model...",
            "Analyzing consumption patterns...",
            "Generating energy consumption forecast...",
            "Calculating confidence intervals...",
            "Preparing consumption report..."
        ]
    elif "streetlight_data_report" in result_classes:
        thought_steps = [
            "Processing data report request...",
            "Identifying report parameters...",
            "Querying multiple data sources...",
            "Aggregating report data...",
            "Performing statistical analysis...",
            "Identifying key insights...",
            "Generating comprehensive report..."
        ]
    elif "streetlight_internal_report" in result_classes:
        thought_steps = [
            "Generating internal streetlight report...",
            "Collecting maintenance and operational data...",
            "Analyzing performance metrics...",
            "Evaluating system health indicators...",
            "Identifying areas for optimization...",
            "Compiling recommendations...",
            "Finalizing detailed report..."
        ]
    elif "bridge_safety_predict" in result_classes:
        thought_steps = [
            "Processing bridge safety prediction request...",
            "Retrieving sensor data from bridge monitoring system...",
            "Analyzing structural integrity indicators...",
            "Running bridge safety assessment algorithms...",
            "Evaluating risk factors and thresholds...",
            "Calculating safety confidence levels...",
            "Preparing bridge safety report..."
        ]
    else:
        # Generic thought process for other types of queries
        thought_steps = [
            "Processing your request...",
            "Searching knowledge base...",
            "Analyzing relevant information...",
            "Formulating comprehensive response..."
        ]
    
    # Stream the thought process steps with delays
    for step in thought_steps:
        yield json.dumps({"type": "thinking", "content": step}) + "\n"
        time.sleep(0.5)  # Short delay between steps for visual effect
    
    # Now get the actual response using the regular ask function
    result = ask(chat)
    
    # Extract the final response
    final_response = ""
    function_calls = []
    
    if 'messages' in result and result['messages']:
        for message in result['messages']:
            if message.get('role') == 'assistant':
                content = message.get('content', '')
                
                # Look for function calls using regex
                function_matches = list(re.finditer(r'(\{\s*"function"\s*:.+?\}|\[\s*\{\s*"function"\s*:.+?\}\s*\])', content, re.DOTALL))
                
                if function_matches:
                    # Extract text parts and function calls
                    last_pos = 0
                    text_parts = []
                    
                    for match in function_matches:
                        # Add text before function call
                        if match.start() > last_pos:
                            text_parts.append(content[last_pos:match.start()])
                        
                        # Extract function call
                        function_text = match.group(0)
                        try:
                            function_data = json.loads(function_text)
                            function_calls.append(function_data)
                        except json.JSONDecodeError:
                            # If not valid JSON, treat as text
                            text_parts.append(function_text)
                        
                        last_pos = match.end()
                    
                    # Add any remaining text
                    if last_pos < len(content):
                        text_parts.append(content[last_pos:])
                    
                    # Join all text parts
                    final_response = "".join(text_parts)
                else:
                    # No function calls found, use entire content
                    final_response = content
    
    # Stream the complete final response
    if final_response.strip():
        yield json.dumps({"type": "text", "content": final_response.strip()}) + "\n"
    
    # Stream any function calls
    for func_call in function_calls:
        yield json.dumps({"type": "function_call", "content": func_call}) + "\n"
    
    # Signal end of stream
    yield json.dumps({"type": "end"}) + "\n"


#############################################History generation##################################################
"""
about this section:
- we need to summarize the conversation as we approach the token limit to carry over 
the conversation history to avoid having too many tokens (too long context) 
that would crash our llm. the safe token limit for us is around 2000 
"""

def should_include_history(current_query: str, previous_query: str, result_classes: List[str]) -> bool:
    """
    Determine if a user needs history context.
    Things including Independent SQL Queries, First-time Report Generation, Direct Predictions, Simple RAG Queries 
    will all return False
    """
    # Check for empty or None queries
    if not current_query or not previous_query:
        return False
    
    query_lower = current_query.lower()
    prev_lower = previous_query.lower()
    
    # 1. Direct references to previous context
    context_references = {
        'demonstrative': ['that', 'those', 'these'],
        'temporal': ['previous', 'last', 'earlier', 'before'],
        'report': ['report', 'analysis', 'result', 'table'],
        'pronouns': ['it', 'them', 'they']
    }
    
    # Look for demonstrative + noun combinations
    for demonstrative in context_references['demonstrative']:
        for report in context_references['report']:
            if f"{demonstrative} {report}" in query_lower:
                # Only return True if previous query actually contains a report/analysis
                if any(term in prev_lower for term in context_references['report']):
                    return True
                
    # 2. Check for follow-up analysis patterns
    analysis_patterns = [
        'first', 'second', 'third', 'column', 'row',
        'from the', 'based on', 'using the'
    ]
    
    if any(pattern in query_lower for pattern in analysis_patterns):
        # Check if previous query was actually a report AND current query references it
        if (any(cls in result_classes for cls in ['streetlight_data_report', 'internal_report']) and
            any(term in prev_lower for term in context_references['report'])):
            return True
            
    # 3. Check for explicit mentions of comparison
    comparison_terms = ['compare', 'difference', 'versus', 'vs']
    if any(term in query_lower for term in comparison_terms):
        return True
        
    # 4. Check for incomplete queries that rely on context
    if len(query_lower.split()) < 4:  # Very short queries might need context
        if any(pronoun in query_lower for pronoun in context_references['pronouns']):
            # Check if pronouns could refer to something in previous query
            has_potential_reference = any(noun in prev_lower for noun in 
                ['streetlight', 'report', 'data', 'error', 'failure', 'consumption'])
            return has_potential_reference
            
    return False


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.
    This is a simple estimation - actual token count may vary."""
    # Simple estimation: roughly 4 chars per token
    token_count = len(text) // 4
    print(f"\n[DEBUG] Token Estimation:")
    print(f"[DEBUG] Text length: {len(text)} characters")
    print(f"[DEBUG] Estimated tokens: {token_count}")
    return token_count

def manage_conversation_history(chat: Dict, result_classes: List[str], max_history_tokens: int = 1500) -> Dict:
   """
   Manage conversation history using hybrid approach (determine if history is needed + token-based management).
   Two types of token management:
   1. For ≤ 3 messages exceeding token limit: Keep last query and immediate context, maintain existing history
   2. For > 3 messages exceeding token limit: Summarize older messages, keep last 3 exchanges
   
   Args:
       chat (Dict): Chat object containing messages and history
       result_classes (List[str]): Classification results for determining context need
       max_history_tokens (int): Maximum allowed tokens for history (default 1500)
       
   Returns:
       Dict: Updated chat object with managed history
   """
   messages = chat.get('messages', [])
   
   # Safety check for empty messages
   if not messages:
       return chat
       
   history = chat.get('messageSummary', '')
   
   # Get current and previous queries for context detection
   current_query = messages[-1]['content']
   previous_query = messages[-2]['content'] if len(messages) > 1 else ""
   
   # Check if context is needed based on query type and previous context
   needs_context = should_include_history(current_query, previous_query, result_classes)
   
   # Reset history if context not needed
   if not needs_context:
       return {
           'messages': [messages[-1]],
           'messageSummary': ""
       }
       
   # Check total tokens regardless of message count
   total_tokens = sum(estimate_tokens(m['content']) for m in messages)
   
   # Only manage history if approaching token limit (80% of max)
   if total_tokens > max_history_tokens * 0.8:
       try:
           # Special handling for ≤ 3 messages
           if len(messages) <= 3:
               # Keep last query and its immediate context (last 2 messages)
               return {
                   'messages': messages[-2:] if len(messages) >= 2 else messages,
                   'messageSummary': history  # Maintain existing history
               }
           
           # For more than 3 messages, do full summarization
           # Prepare context by combining existing history and older messages
           context_to_summarize = history.strip()
           if context_to_summarize:
               context_to_summarize += "\n"
           context_to_summarize += "\n".join(m['content'] for m in messages[:-3])
           
           # Create summary prompt
           summary_prompt = [{
               'role': 'user',
               'content': ("###System: Summarize this conversation preserving key data "
                         "references and important context ###User: " + context_to_summarize)
           }]
           
           # Generate new summary using LLM
           new_history = llm_api(summary_prompt)
           if not isinstance(new_history, str):
               new_history = str(new_history)
           
           # Validate and truncate summary if too long (40% of max tokens)
           summary_tokens = estimate_tokens(new_history)
           if summary_tokens > max_history_tokens * 0.4:
               words = new_history.split()
               new_history = " ".join(words[:100]) + "..."
           
           return {
               'messages': messages[-3:],  # Keep last 3 exchanges
               'messageSummary': new_history
           }
           
       except Exception as e:
           logging.error(f"Failed to generate history summary: {e}")
           # Fallback: keep last 3 messages and existing history
           return {
               'messages': messages[-3:],
               'messageSummary': history
           }
   
   # If no summarization needed (tokens within limit), return original chat
   return chat

#############################################Core Utility Functions##################################################
"""
about this section:
- Basic functions used across multiple handlers
- Generic utilities like timestamp conversion, data size calculation, system prompt building
- These are Utility functions that other sections rely on
"""


def llm_query(messages, previous_history=""):
    """
    Query the LLM API and optionally summarize conversation history.
    Args:
        messages (list): List of dictionaries representing the conversation.
        history (str): Optional history to be summarized.
    Returns:
        tuple: LLM response and optional summarized history.
    """
    previous_history = previous_history or ""
    start_time = time.time() 
    response_content = llm_api(messages) # This is direct call to LLM API
    end_time = time.time()
    duration_response = end_time - start_time
    print(f"Response generation took {duration_response:.5f} seconds")
    # Ensure response_content is stringified if it's not already a string
    if isinstance(response_content, dict):
        response_content = json.dumps(response_content)
    elif not isinstance(response_content, str):
        response_content = str(response_content)
    return response_content, previous_history


def parse_list_param(param: str) -> List[str]:
    """Parse list-like parameters from string"""
    if not param or param.lower() == 'all' or param == "":
        return None
    return [str(element) for element in string_to_list_of_strings(param)]

def _build_system_prompt(result_classes: List[str], user_message: str) -> str:
    """Build system prompt based on classification results"""
    system_prompt = ""
    
    if "streetlight_data_report" in result_classes:
        system_prompt += "Call streetlight_data_report "
    if "building_data_report" in result_classes:
        system_prompt += "Call building_data_report "
    if "streetlight_sql" in result_classes:        
        # Format the input for SQL column classification - unchanged
        sql_input = [{
            "role": "user",
            "content": f"###System: You are a sql column classifier assistant ###User: {user_message}"
        }]
        
        # Get column classification using llm_api - unchanged
        columns = llm_api(sql_input)
        
        try:
            if isinstance(columns, str):
                # NEW: Check if response is in troubleshooting format
                # Troubleshooting format example:
                # "This is a troubleshooting instance. We need the following columns to solve the issue.
                # relevant columns: ["log.alarmFault.dayBurn", "log.relayStatus"]"
                if "relevant columns:" in columns:
                    # NEW: Extract just the JSON array part that comes after "relevant columns:"
                    # Split on "relevant columns:" and take the last part
                    array_part = columns.split("relevant columns:")[-1].strip()
                    # Parse the extracted JSON array
                    columns = json.loads(array_part)
                else:
                    # UNCHANGED: Handle original simple format
                    # Simple format example: ["device.devAddr", "log.highLoadVoltageThreshold"]
                    columns = json.loads(columns)
                    
            # UNCHANGED: Add the parsed columns to the system prompt
            system_prompt += f"Call streetlight_sql: {columns}"
        except json.JSONDecodeError:
            # UNCHANGED: Fallback if JSON parsing fails
            print("Warning: Could not parse SQL column classification response")
            system_prompt += "Call streetlight_sql"
    if "bridge_sql" in result_classes:
        system_prompt += "Call bridge_sql "
    if "building_sql" in result_classes:
        system_prompt += "Call building_sql "
    if "streetlight_energy_consumption_predict" in result_classes:
        system_prompt += "Call streetlight_energy_consumption_predict"
    if "streetlight_failure_predict" in result_classes:
        system_prompt += "Call streetlight_failure_predict "
    if "bridge_safety_predict" in result_classes:
        system_prompt += "Call bridge_safety_predict "
    if "streetlight_internal_report" in result_classes:
        system_prompt += "Call streetlight_internal_report "
    if "no function" in result_classes or "RAG" in result_classes:
        system_prompt = "You are a smart city expert "
    if not system_prompt:
        system_prompt = "You are a smart city expert "
        
    return system_prompt

def calculate_data_size(input_list: List[Dict]) -> int:
    """Calculate size of data based on unique timestamps"""
    if not input_list:
        return 0
    
    timestamps = set()
    for item in input_list:
        if isinstance(item, dict) and 'timeStamp' in item:
            timestamps.add(item['timeStamp'])
    
    return len(timestamps)

def convert_timestamps(obj: Any) -> Any:
    """Recursively convert timestamps to strings"""
    if isinstance(obj, dict):
        return {k: convert_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps(i) for i in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    return obj



def handle_sql(params, vertical):
    """
    Handle SQL queries based on parameters and vertical.
    
    Parameters:
        params (Dict): Parameters for the SQL query.
        vertical (str): The vertical/domain to query.
    
    Returns:
        Dict: Processed SQL query results.
    """
    print(f"###Debug: Vertical in handle_sql: {vertical}")

    query = params.get('query', '') # This gets semi-natural sql query text
    start_time = params.get('start_time') # This gets start time in english natural language (may or may not have)
    end_time = params.get('end_time') # This gets end time in english natural language (may or may not have)

    query = clean_temporal_references(query, start_time, end_time) # Having other temporal phrases messes up sql generation

    if start_time or end_time:
        unix_start_time = convert_to_unix_time(start_time) # Converts to unix milisecond for database query
        unix_end_time = convert_to_unix_time(end_time) # Converts to unix milisecond for database query
    else:
        unix_start_time = "all time" # This is the default, which is ignore timestamp
        unix_end_time = "all time" # This is the default, which is ignore timestamp
    
    english_string = f"{query} from {unix_start_time} to {unix_end_time} unix milisecond timestamp" #Remove 'Query for' because it caused 'Query for Query' in SQL
    logging.info(f"###This was sql question input: {english_string} ###End\n")

    raw_result = query_internal_databases(english_string, vertical) # Executes query:
    if isinstance(raw_result, dict) and all(value == '' for value in raw_result.values()):
        logging.warning(f"Query returned no data: {english_string}")

    print(f"###Debug: Raw result from query_internal_databases: {raw_result}")

    logging.info(f"###This is sql raw results: {raw_result} ###End\n")
    
    processed_result = convert_query_timestamps_to_readable(raw_result) # Convert timestamps to datetime format for llm interpretation

    return processed_result

#############################################parse Json section ##################################################
"""
about this section:
- Dedicated to parsing and routing different types of JSON responses
- Each parsing function (_parse_sql_query, _parse_failure_prediction, etc.) is independent
- Changes to one parser won't affect others
"""


def preprocess_json(input_str):
    """
    try to fix Json format
    """ 
    # Remove any whitespace around the JSON
    input_str = input_str.strip()
    
    # Only process if it looks like a JSON array
    if not (input_str.startswith('[') and input_str.endswith(']')):
        return input_str
        
    # Remove the outer brackets
    content = input_str[1:-1].strip()
    
    # Split into individual objects, handling nested structures
    objects = []
    current_obj = ""
    brace_count = 0
    
    for char in content:
        current_obj += char
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                objects.append(current_obj.strip())
                current_obj = ""
        elif char == ',' and brace_count == 0:
            current_obj = ""
    
    # Process each object
    processed_objects = []
    for obj in objects:
        if obj:
            # Remove surrounding whitespace and ensure proper braces
            obj = obj.strip()
            if not obj.startswith('{'): 
                obj = '{' + obj
            if not obj.endswith('}'): 
                obj = obj + '}'
            
            # Replace single-quoted booleans with unquoted
            obj = re.sub(r"'(true|false)'", r"\1", obj)
            
            # Replace single quotes with double quotes for strings
            obj = re.sub(r"(?<!\w)'([^']+)'(?!\w)", r'"\1"', obj)
            
            processed_objects.append(obj)
    
    # Reconstruct the JSON array with proper formatting
    result = '[' + ', '.join(obj for obj in processed_objects if obj) + ']'
    
    return result


def parse_json(json_str: str) -> Tuple[List[Dict], List[Any]]:
    """
    Parse and route JSON responses to appropriate handlers.
    
    Parameters:
        json_str (str): JSON string from the LLM
    
    Returns:
        Tuple[List[Dict], List[Any]]: Parsed data and remaining functions
    """    
    # Handle both SQL query and JSON responses
    try:
        # Try parsing as JSON first
        data = json.loads(json_str)
        print("###Successfully parsed JSON response###")
        
    except json.JSONDecodeError:
        # If JSON parsing fails, check if it's a raw SQL query
        if isinstance(json_str, str) and json_str.strip().upper().startswith('SELECT '):
            print("###Converting raw SQL query to proper format###")
            data = [{
                "function": "streetlight_sql",
                "parameters": {
                    "query": json_str.strip(),
                    "start_time": "1 days ago",
                    "end_time": "now"
                }
            }]
        else:
                    # Log the input before processing
                    print("###Attempting to force parse JSON###")
                    print(f"###Input to json.loads(): {json_str}###")
                    
                    # Preprocess the JSON string
                    json_str = preprocess_json(json_str)
                    print(f"###Preprocessed JSON string: {json_str}###")
                    
                    # Try json.loads() directly after preprocessing
                    try:
                        data = json.loads(json_str)
                        print("###Successfully parsed JSON after preprocessing###")
                    except json.JSONDecodeError as e:
                        print(f"###JSONDecodeError after preprocessing: {e}###")
                        data = force_json(json_str)  # Fallback to force parsing as last resort
    
    # Ensure data is a list
    data = [data] if not isinstance(data, list) and data is not None else data
    
    if data is None:
        logging.error("No valid data to parse")
        return [], []

    parsed_data = []
    functions = []

    for obj in data:
        # Skip invalid objects
        if not isinstance(obj, dict) or "function" not in obj:
            continue

        func = obj.get("function")
        params = obj.get("parameters", {})
        print(f"###Processing function: {func}###")

        try:
            # Handle different function types
            if func.endswith("_sql"):
                result = _parse_sql_query(params, func.split('_')[0])
                parsed_data.append(result) # Add SQL query result
                print("###SQL query processed###")
                
            elif func == "streetlight_failure_predict":
                result, func_result = _parse_failure_prediction(params)
                parsed_data.append(result) # Add failure prediction result
                functions.append(func_result) # Add additional function result
                print("###Failure prediction processed###")
                
            elif func == "streetlight_energy_consumption_predict":
                result, func_result = _parse_energy_prediction(params)
                parsed_data.append(result)
                functions.append(func_result)
                print("###Energy consumption prediction processed###")
                
            elif func.endswith("_data_report"):
                result = _parse_data_report(params, func.split('_')[0])
                parsed_data.append(result)
                print("###Data report processed###")
                
            elif func == "streetlight_internal_report":
                result = _parse_internal_report(params)
                parsed_data.append(result)
                print("###Internal report processed###")
                
            elif func == "bridge_safety_predict":
                result, func_result = _parse_bridge_safety(params)
                parsed_data.append(result)
                functions.append(func_result)
                print("###Bridge safety prediction processed###")
                
            else:
                logging.warning(f"Unknown function '{func}'. Skipping.")
                continue

        except Exception as e:
            logging.error(f"Error processing {func}: {e}")
            parsed_data.append({"error": f"Failed to process {func}"})

    print(f"###Parse JSON completed. Parsed data count: {len(parsed_data)}, Functions count: {len(functions)}###")
    return parsed_data, functions

def _parse_sql_query(params: Dict, vertical: str) -> Dict:
    """Parse SQL query parameters and execute query"""
    print(f"###Debug: Vertical before transformation: {vertical}")
    # Fix the database naming
    if vertical == "streetlight":
        vertical = "streetlighting"  # Change from streetlight to streetlighting
    print(f"###Debug: Vertical after transformation: {vertical}")
    result = handle_sql(params, vertical)
    return result

def _parse_failure_prediction(params: Dict) -> Tuple[str, Dict]:
    """Parse failure prediction parameters and execute prediction"""
    raw_device_id = params.get("device_ids", None) # Extract device IDs
    groups = parse_list_param(params.get("groups")) # Extract group IDs
    street_name = parse_list_param(params.get("street_names")) # Extract street names
    
    time_match = re.search(r'\d+(\.\d+)?', params.get("time", ""))
    time = float(time_match.group()) if time_match else 0
    
    if time >= 48:
        return "Unable to predict beyond 2 days in high accuracy", None
    
    def string_to_list(s):
        if isinstance(s, list):  # If input is already a list, return as is
            return s
        s = s.strip("[]")  # Remove square brackets
        return [item.strip() for item in s.split(',')] if s else []  # Split and remove spaces

    device_id = string_to_list(raw_device_id)  # Convert once and use the converted version
    data = streetlight_failure_predict(forward_period=time, device_list=device_id)
    return handle_failure_results(data, time, device_id, groups, street_name)

def _parse_energy_prediction(params: Dict) -> Tuple[str, Dict]:
    """Parse energy prediction parameters and execute prediction"""
    try:
        # Parse group and street name parameters into lists
        groups = parse_list_param(params.get("groups"))
        street_name = parse_list_param(params.get("street_names"))
        
        # Handle time parameter and extract time unit and number
        time_str = params.get("time", "")
        if isinstance(time_str, str):
            time_unit = time_str[-1] if time_str else 'd'
            time_number = int(re.search(r'\d+', time_str).group()) if re.search(r'\d+', time_str) else 1
        else:
            time_unit = 'd'
            time_number = 1
            
        # Perform energy consumption prediction
        data = streetlight_energy_prediction(
            groups=groups,
            streetnames=street_name,
            prediction_cadence=time_unit,
            prediction_periods=time_number
        ) 
        
        # Check if the resulting DataFrame is valid and not empty
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            return "No energy consumption data available for prediction", None
            
        # Handle and format the prediction results
        result_text, result_json = handle_consumption_results(data)
        return f"{result_text}. Corresponding table has been generated below", result_json
        
    except Exception as e:
        # Handle and log errors encountered during prediction
        print(f"Error in energy prediction: {str(e)}")
        return f"Unable to predict energy consumption: {str(e)}", None

def _parse_data_report(params: Dict, database: str) -> Dict:
    """Parse data report parameters and generate report"""
    # Convert start and end time to Unix timestamp format
    params["start_time"] = convert_to_unix_time(params.get("start_time", "30 days ago"))
    params["end_time"] = convert_to_unix_time(params.get("end_time", "0 days ago"))
    
    # Fix the database naming
    if database == "streetlight":
        database = "streetlighting"  # Change from streetlight to streetlighting
        
    # Generate the data report in table format
    return report_to_table([{"parameters": params}], f"govchat_{database}")


def _parse_internal_report(params: Dict) -> Dict:
    """Parse internal report parameters and generate report"""
    try:
        # Get current time in milliseconds
        current_time = int(time.time() * 1000)
        
        # Handle start time
        if not params.get("start"):
            # Default to 24 hours ago
            start_time = current_time - (24 * 60 * 60 * 1000)
        else:
            # Parse the relative time string
            time_str = params["start"]
            if isinstance(time_str, str) and "days ago" in time_str:
                days = int(time_str.split()[0])
                start_time = current_time - (days * 24 * 60 * 60 * 1000)
            else:
                start_time = convert_to_unix_time(params["start"])
        
        # Handle end time
        if not params.get("end") or params["end"] == "0 days ago":
            end_time = current_time
        else:
            end_time = convert_to_unix_time(params["end"])
            
        # Debug logging
        print(f"\n=== Middleware Time Range Details ===")
        print(f"Start time: {datetime.fromtimestamp(start_time/1000)} ({start_time})")
        print(f"End time: {datetime.fromtimestamp(end_time/1000)} ({end_time})")
        print(f"Duration: {(end_time - start_time)/(1000*60*60)} hours")
        print("=================================\n")
        
        # Generate the internal report
        result = internal_report_streetlighting(
            start_date=start_time,
            end_date=end_time,
            groups=params.get("groups"),
            street_names=params.get("street_names")
        )
        
        # Log the generated report for debugging
        print(f"Internal report result: {result}")
        return result
        
    except ValueError as e:
        print(f"ValueError in internal report parsing: {e}")
        return {
            "function": "table_vertical",
            "parameters": [{
                "Error": f"Error processing internal report: {str(e)}",
                "Time Period": f"From {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}"
            }]
        }
    except Exception as e:
        print(f"Error in internal report parsing: {e}")
        return {
            "function": "table_vertical",
            "parameters": [{
                "Error": "Unexpected error in report generation",
                "Time Period": f"Start: {params.get('start', 'unknown')}, End: {params.get('end', 'unknown')}"
            }]
        }
    
def _parse_bridge_safety(params: Dict) -> Tuple[str, Dict]:
    """Parse bridge safety parameters and execute prediction"""
    
    #extract device IDs, bridge names, time period
    device_ids = parse_list_param(params.get("device_ids"))
    bridges = parse_list_param(params.get("bridges"))
    time = params.get("time")
    
    prediction_data = get_bridge_safety_prediction(
        device_ids=device_ids,
        bridge=bridges[0] if bridges else None,
        time_frame=time
    )
    
    result_text = prediction_data if isinstance(prediction_data, str) else json.dumps(prediction_data, ensure_ascii=False)
    result_json = prediction_data if isinstance(prediction_data, dict) else None
    
    return result_text, result_json

#############################################Response Formatting Utilities for Each Handler##################################################
"""
about this section:
- Focuses solely on preparing responses for different handlers
- Handles data formatting, size reduction, and message structuring
- Separated by response type (SQL, prediction, report)
"""


def _prepare_supporting_info_sql(parsed_response: List, trend_analysis_results: List) -> Dict:
    """Prepare supporting information for SQL responses"""
    # Ensure parsed_response is a list for consistent processing
    if not isinstance(parsed_response, list):
        parsed_response = [parsed_response]

    # Iterate over trend analysis results and add trend summaries to parsed response
    for i, trend_summary in enumerate(trend_analysis_results):
        if trend_summary and i < len(parsed_response):
            # Attach trend summary to the corresponding parsed response item
            parsed_response[i]['trend_summary'] = trend_summary
        elif trend_summary:
            # Warn if there's a trend summary without a corresponding parsed response
            print(f"Warning: Found trend summary {i} but no matching parsed response")

    # Convert timestamps in the parsed response to a readable format
    return convert_timestamps(parsed_response)

def _prepare_supporting_info_prediction(parsed_response: List, trend_analysis_results: List) -> Dict:
    """Prepare supporting information for prediction responses"""
    # Ensure parsed_response is a list for consistent processing
    if not isinstance(parsed_response, list):
        parsed_response = [parsed_response]
        
    # Convert timestamps in the parsed response to a readable format
    processed_response = convert_timestamps(parsed_response)
    
    # Add trend analysis data to processed response if available
    for i, trend_summary in enumerate(trend_analysis_results):
        if trend_summary and i < len(processed_response):
            # Attach trend analysis summary to the corresponding parsed response item
            processed_response[i]['trend_analysis'] = trend_summary
            
    return processed_response

def _handle_size_reduction(supporting_information: Dict, result_classes: List[str]) -> Dict:
    """Handle size reduction for responses.
   
   Parameters:
       supporting_information (Dict): Information to potentially reduce in size
       result_classes (List[str]): List of result classes to determine reduction type
   
   Returns:
       Dict: Size-reduced information if needed
    """
    
    original_size = len(str(supporting_information))
    size_threshold = 2000

    print(f"Original data size: {original_size:,} chars")
   
    if original_size > size_threshold:
        if "streetlight_data_report" in result_classes:
            print("Using downsize_data_report_information")
            print(f"Input format: {type(supporting_information)}")
            print(f"First item format: {type(supporting_information[0]) if isinstance(supporting_information, list) else 'N/A'}")
            supporting_information = [
                downsize_data_report_information(resp) 
                for resp in supporting_information
            ]
        elif "streetlight_internal_report" in result_classes:
            print("Using downsize_internal_report")
            print(f"Input format: {type(supporting_information)}")
            print(f"First item format: {type(supporting_information[0]) if isinstance(supporting_information, list) else 'N/A'}")
            supporting_information = [
                downsize_internal_report(resp) 
                for resp in supporting_information
            ]
        else:  # For SQL queries and others
            print("Using downsize_sql_supporting_information")
            print(f"Input format: {type(supporting_information)}")
            print(f"First item format: {type(supporting_information[0]) if isinstance(supporting_information, list) else 'N/A'}")
            supporting_information = [
                downsize_sql_supporting_information(resp) 
                for resp in supporting_information
            ]
       
        final_size = len(str(supporting_information))
        print(f"Total context reduction: {original_size:,} chars -> "
              f"{final_size:,} chars ({(final_size/original_size)*100:.1f}% of original)")
    else:
        print(f"Data size ({original_size:,} chars) below threshold "
              f"({size_threshold:,} chars). Skipping reduction.")

    return supporting_information


def _format_final_message(messages: List[Dict], supporting_information: Dict, history: str = "") -> str:
    """Format final message for LLM with supporting information and history"""
    user_query = re.split(r'###User: ', messages[-1]['content'])[-1].strip()
    
    # Add history to supporting info if it exists
    if history is not None and history != '':
        if isinstance(supporting_information, list):
            supporting_information.append({"conversation_history": history.strip()})
        else:
            supporting_information = [supporting_information, {"conversation_history": history.strip()}]
            
    return "###System: You are a smart city expert ###User: " + json.dumps({
        "user_query": user_query,
        "supporting_information": supporting_information
    }, ensure_ascii=False)
#############################################Request Handler Functions section##################################################
"""
about this section:
- Each handler is independent and handles one type of request
- Contains the main logic for processing specific request types
- Clear entry and exit points for each request type
- each request has its own second LLM input/output for separation of concerns

Note: need to ensure second llm input/response for each handler works
"""

def handle_title_request(chat: Dict) -> Dict:
    """Handle title generation requests"""
    chat['messages'][-1]['content'] = (
        "###System: Generate a 1-5 word title ###User: " + 
        chat['messages'][-1]['content']
    )
    start_time = time.time()
    title_response = llm_api(chat['messages'], max_new_tokens=15)
    end_time = time.time()
    print(f"Title generation took {end_time - start_time:.2f} seconds")
    return title_response

def handle_general_request(messages: List[Dict], history: str) -> Dict:
    """Handle general conversational requests that don't require specialized processing"""
    gpt_response, new_history = llm_query(messages, history)
    if not isinstance(gpt_response, str):
        gpt_response = json.dumps(gpt_response)
    
    return {
        "messages": [{"role": "assistant", "content": gpt_response}],
        "messageSummary": new_history if new_history.strip() else None
    }

def handle_sql_request(messages: List[Dict], history: str, result_classes: List[str], initial_response: str = None) -> Dict:
    """Handle SQL-related requests"""
    # First LLM call: Use passed-in response instead of making new call
    gpt_response = initial_response or llm_query(messages, history)[0]  # Fallback if no initial response
    print("###This is first LLM return: ", gpt_response, "\n###End\n")
    
    # Parse the response
    parsed_response, _ = parse_json(gpt_response)
    
    if parsed_response == "Invalid LLM function output":
        return handle_fallback(messages, history)
    
    # Process the response
    trend_results = _analyze_trends(parsed_response)
    supporting_info = _prepare_supporting_info_sql(parsed_response, trend_results)
    supporting_info = _handle_size_reduction(supporting_info, result_classes)
    
    # Format and make second LLM call
    messages[-1]['content'] = _format_final_message(messages, supporting_info, history)
    print("###This is second LLM input: ", messages, "\n###End\n")
    
    second_response, new_history = llm_query(messages, history)
    print("###This is second LLM return: ", second_response, "\n###End\n")
    
    if not isinstance(second_response, str):
        second_response = json.dumps(second_response)
    
    # Remove "No history " prefix if it exists
    
    if second_response.startswith("No history "):
        print("###Removing 'No history ' prefix...")
        print("###Second response before processing: ", second_response, "\n###End\n")
        second_response = second_response[11:]
        print("###After removal: ", second_response, "\n###End\n")

    # Add SQL table to response
    second_response += "\nBelow are tables visualizing the queried data:\n" + str(sql_to_table(parsed_response))
    
    return {
        "messages": [{"role": "assistant", "content": second_response}],
        "messageSummary": new_history
    }

def handle_prediction_request(messages: List[Dict], history: str, result_classes: List[str], initial_response: str = None) -> Dict:
    """Handle prediction requests"""
    gpt_response = initial_response or llm_query(messages, history)[0]
    remaining_json = None  # Initialize here
    
    # If it's a natural language energy prediction, use it directly as supporting info
    if isinstance(gpt_response, str) and "energy consumption" in gpt_response.lower():
        supporting_info = [gpt_response]  # Use natural language response directly
    else:
        # Regular JSON parsing path
        parsed_response, remaining_json = parse_json(gpt_response)
        
        if parsed_response == "Invalid LLM function output":
            return handle_fallback(messages, history)
            
        supporting_info = parsed_response


    # For predictions, we don't need trend analysis or size reduction
    messages[-1]['content'] = _format_final_message(messages, supporting_info, history)
    second_response, new_history = llm_query(messages, history)
    
    if not isinstance(second_response, str):
        second_response = json.dumps(second_response)

    if second_response.startswith("No history "):
        print("###Removing 'No history ' prefix...")
        print("###Second response before processing: ", second_response, "\n###End\n")
        second_response = second_response[11:]
        print("###After removal: ", second_response, "\n###End\n")

    if remaining_json:
        second_response += str(remaining_json)
    
    return {
        "messages": [{"role": "assistant", "content": second_response}],
        "messageSummary": new_history
    }

def handle_report_request(messages: List[Dict], history: str, result_classes: List[str], initial_response: str = None) -> Dict:
    """Handle data report generation requests"""
    gpt_response = initial_response or llm_query(messages, history)[0]
    parsed_response, _ = parse_json(gpt_response)
    
    supporting_info = _prepare_supporting_info_sql(parsed_response, [])
    supporting_info = _handle_size_reduction(supporting_info, result_classes)
    
    # Add second LLM call for analysis
    messages[-1]['content'] = _format_final_message(messages, supporting_info, history)
    second_response, new_history = llm_query(messages, history)
    
    if not isinstance(second_response, str):
        second_response = json.dumps(second_response)
    
    if second_response.startswith("No history "):
        print("###Removing 'No history ' prefix...")
        print("###Second response before processing: ", second_response, "\n###End\n")
        second_response = second_response[11:]
        print("###After removal: ", second_response, "\n###End\n")
        
    # Use format_data_report_table for handling the nested list structure
    second_response += "\nBelow are tables visualizing the queried data:\n" + format_data_report_table(parsed_response)
    
    return {
        "messages": [{"role": "assistant", "content": second_response}],
        "messageSummary": new_history
    }

def handle_internal_report_request(messages: List[Dict], history: str, result_classes: List[str], initial_response: str = None) -> Dict:
    """Handle internal report requests"""
    gpt_response = initial_response or llm_query(messages, history)[0]
    parsed_response, _ = parse_json(gpt_response)
    
    supporting_info = _prepare_supporting_info_sql(parsed_response, [])
    supporting_info = _handle_size_reduction(supporting_info, result_classes)
    
    # Add second LLM call for analysis
    messages[-1]['content'] = _format_final_message(messages, supporting_info, history)
    second_response, new_history = llm_query(messages, history)
    
    if not isinstance(second_response, str):
        second_response = json.dumps(second_response)
    
    if second_response.startswith("No history "):
        print("###Removing 'No history ' prefix...")
        print("###Second response before processing: ", second_response, "\n###End\n")
        second_response = second_response[11:]
        print("###After removal: ", second_response, "\n###End\n")
    
    # Extract the analysis part (everything before "Below are tables")
    analysis = second_response.split("\nBelow are tables")[0].strip()
    
    # Modify parsed_response to include the analysis
    if isinstance(parsed_response, list) and len(parsed_response) > 0:
        if "function" in parsed_response[0] and parsed_response[0]["function"] == "table_vertical":
            # Clean the analysis text
            clean_analysis = analysis.strip().replace('\n', ' ').replace('\r', '')
            parsed_response[0]["parameters"].insert(0, {
                "System Analysis and Recommendation": clean_analysis
            })

    # Add parsed response to the analysis
    second_response += "\nBelow are tables visualizing the queried data:\n" + format_internal_report_table(parsed_response)
    
    return {
        "messages": [{"role": "assistant", "content": second_response}],
        "messageSummary": new_history
    }


def handle_bridge_safety_request(messages: List[Dict], history: str, result_classes: List[str], initial_response: str = None) -> Dict:
    """Handle bridge safety prediction requests"""
    gpt_response = initial_response or llm_query(messages, history)[0]
    parsed_response, remaining_json = parse_json(gpt_response)
    
    if parsed_response == "Invalid LLM function output":
        return handle_fallback(messages, history)
        
    # Bridge safety might need its own supporting info preparation
    supporting_info = _prepare_supporting_info_prediction(parsed_response, [])
    supporting_info = _handle_size_reduction(supporting_info)
    
    messages[-1]['content'] = _format_final_message(messages, supporting_info, history)
    second_response, new_history = llm_query(messages, history)
    
    if not isinstance(second_response, str):
        second_response = json.dumps(second_response)
        
    if remaining_json:
        second_response += str(remaining_json)
    
    return {
        "messages": [{"role": "assistant", "content": second_response}],
        "messageSummary": new_history
    }


def handle_fallback(messages: List[Dict], history: str) -> Dict:
    """Handle cases where function call generation fails"""
    print("###Warning: Invalid LLM function output ###\n")
    messages[-1]['content'] = re.sub(
        r'###System:.*?###User:', 
        '###System: You are a smart city expert ###User:', 
        messages[-1]['content']
    )
    gpt_response, new_history = llm_query(messages, history)
    return {
        "messages": [{"role": "assistant", "content": gpt_response}],
        "messageSummary": new_history if new_history.strip() else None
    }

#############################################Main Routing Function: ask() ##################################################
"""
about this section:
- Single responsibility of routing requests to appropriate handlers
- Clean, high-level control flow
- Easy to add new request types for V2 features

"""

def ask(chat: Dict) -> Dict:
    """Main router function for processing chat requests"""
    if chat.get('title'):
        return handle_title_request(chat)
        
    # First extract messages and history
    messages, history = extract_summary_and_tokens(chat)
    
    # Then classify request using extracted messages
    context, result_classes = classify_task(messages[-1]['content'])
    print("###This is classification: ", result_classes, "\n###End\n")
    
    ##############added history generation section#######################
   
    # Then manage history with classification results
    print("\n[DEBUG] Managing conversation history...")
    chat = manage_conversation_history(chat, result_classes=result_classes)
    
    # Get updated messages and history after management
    messages, history = extract_summary_and_tokens(chat)
    print(f"[DEBUG] History after management: {history[:100]}..." if history else "[DEBUG] No history available")
   
    #################################################################

    # Prepare system prompt
    system_prompt = _build_system_prompt(result_classes, messages[-1]['content'])
    messages[-1]['content'] = f"###System: {system_prompt} ###User: {messages[-1]['content']}"
    print("###This is first LLM input: ", messages, "\n###End\n")
    
    # Make LLM call if NOT a general conversation request
    initial_response = None
    if "no function" not in result_classes:
        initial_response, new_history = llm_query(messages, history)
        if not isinstance(initial_response, str):
            initial_response = json.dumps(initial_response)
        print("###This is first LLM return: ", initial_response, "\n###End\n")

    # Route to appropriate handler based on request type
    try:
        if "no function" in result_classes:
            return handle_general_request(messages, history)
        elif "streetlight_sql" in result_classes:
            return handle_sql_request(messages, history, result_classes, initial_response)
        elif "streetlight_failure_predict" in result_classes:
            return handle_prediction_request(messages, history, result_classes, initial_response)
        elif "streetlight_energy_consumption_predict" in result_classes:
            return handle_prediction_request(messages, history, result_classes, initial_response)
        elif "streetlight_data_report" in result_classes or "building_data_report" in result_classes:
            return handle_report_request(messages, history, result_classes, initial_response)
        elif "bridge_safety_predict" in result_classes:
            return handle_bridge_safety_request(messages, history, result_classes, initial_response)
        elif "streetlight_internal_report" in result_classes:
            return handle_internal_report_request(messages, history, result_classes, initial_response)
        else:
            logging.warning(f"No specific handler found for classes: {result_classes}")
            return handle_general_request(messages, history)
    except Exception as e:
        logging.error(f"Error in request handling: {e}")
        return handle_fallback(messages, history)
    
#############################################Trend Analysis Helper##################################################
"""
about this section:
- Isolated analysis functionality
- Can be modified without affecting other parts
"""

def _analyze_trends(parsed_response: List) -> List[str]:
    """Analyze trends in the response data"""
    trend_analysis_results = []
    
    for content in parsed_response:
        transformed_data = normalize_sql_return(content)
        if not transformed_data:
            trend_analysis_results.append("")
            continue

        data_size = calculate_data_size(transformed_data)
        should_analyze = data_size > 0

        if should_analyze:
            results = analyze_abnormal_behaviors(
                transformed_data,
                z_score_threshold=3.0,
                show_counts=False,
            )
            if "analysis" in results and results["analysis"]:
                abnormal_summary = generate_summary(results)
                trend_analysis_results.append(
                    ". Abnormal behavior summary and other stats: " + abnormal_summary
                )
            else:
                trend_analysis_results.append("")
        else:
            trend_analysis_results.append("")
            
        trend_analysis_results.append(generate_summary(transformed_data))
    
    return trend_analysis_results

##############################################################################################################################
#################################################### Below is for Testing ####################################################
##############################################################################################################################

irrelevant_ask = {
    'messages': [
        {
            "role": "user",
            "content": "why is the sky blue"
        },
        {
            'role': 'assistant',
            'content': f"The blue light in the sky is scattered in all directions by the gases and particles in the atmosphere. Blue light has a shorter wavelength than red light and is scattered more than red light, which is why the sky appears blue during the day. This scattering is known as Rayleigh scattering, named after the British scientist Lord Rayleigh, who first described it in the 19th century. The amount of scattering is inversely proportional to the fourth power of the wavelength, so shorter wavelengths are scattered much more than longer wavelengths. This is why the sky is blue and sunsets are red."
        },
        {
            "role": "user",
            "content": "man I hate physics, but I need to know everything about this. i need all the mathematical formulations for this please"
        }],
    'messageSummary': 'The blue color of the sky is caused by Rayleigh scattering, where shorter-wavelength blue light is scattered more than longer-wavelength red light by gases and particles in the atmosphere. The intensity of scattered light is described by the Rayleigh scattering formula, which depends on the wavelength of the incident light, the size of the scattering particles, the refractive index of the medium, and the scattering angle. The formula shows that scattering intensity is inversely proportional to the fourth power of the wavelength and directly proportional to the square of the scattering angle. Rayleigh scattering, first described by British scientist Lord Rayleigh in the 19th century, is a fundamental concept in the study of scattering phenomena and explains the blue color of the sky, the red color of sunsets, and the scattering of light by dust and air pollution.'
}

rag_ask = {"messages": [
    {
        "role": "user",
        "content": "what are methods to combat corrosion of streetlight poles and lights for streetlights close to the shore"
    },
],
    "messageSummary": ""
}

streetlight_data_query_ask1 = {
    'messages': [
        {
            "role": "user",
            "content": """which streetlights, and in which geozone, have reported power outage failures in last 1 month?"""
        }],
    'messageSummary': ''
}

streetlight_data_query_ask2 = {"messages": [
    {
        "role": "user",
        "content": "how many streetlights do i have in my database, and I also want total energy output for all time. Also would like to know all the errors and for which streetlight they occured on in the past month"
    },
],
    "messageSummary": ""
}

streetlight_data_query_ask3 = {
    'messages': [
        {
            "role": "user",
            "content": """can you query for me streetlights have had high temperature errors of all time? Also are any streetlights on 彌敦道 having power issues"""
        }],
    'messageSummary': ''
}

streetlight_data_query_ask4 = {
    'messages': [
        {
            "role": "user",
            "content": """Specifically for project 1, I want to know asset information on the streetlights there. What are their wattages, what model, what communication protocols, how many there are."""
        }],
    'messageSummary': ''
}
streetlight_data_query_ask5 = {"messages": [
    {
        "role": "user",
        "content": "how many total flickering failures have there been in geozone 5 in the past 4 months?"
    },
],
    "messageSummary": ""
}
streetlight_data_query_ask6 = {"messages": [
    {
        "role": "user",
        "content": "which devices have had power failure in the past three months"
    },
],
    "messageSummary": ""
}

streetlight_data_query_ask7 = {"messages": [
    {
        "role": "user",
        "content": "how many power outage events happened in the last 2 hours"
    },
],
    "messageSummary": ""
}

streetlight_failure_prediction_ask = {
    'messages': [
        {
            "role": "user",
            "content": "predict how many streetlights will fail in the next 24 hours/day"
        }],
    'messageSummary': ''
}

streetlight_failure_prediction_ask2 = {
    'messages': [
        {
            "role": "user",
            "content": "how many devices will have issue for device 100880 the next 12 hours?"
        }],
    'messageSummary': ''
}

streetlighting_data_report_ask1 = {
    'messages': [
        {
            "role": "user",
            "content": "can i have a flickering report on over the past 2 days"
        }],
    'messageSummary': ''
}


streetlighting_data_report_ask2 = {
    'messages': [
        {
            "role": "user",
            "content": """ Create a custom streetlight report named spesland_report. At the project level, include communication details (errors and methods), voltage, and temperature data. At the device level, add timestamps for these. Cover the past 4 years up to now."""
        }],
    'messageSummary': ''
}

streetlighting_data_report_ask3 = {
    'messages': [
        {
            "role": "user",
            "content": "i want a power outage report"
        }],
    'messageSummary': ''
}

streetlighting_data_report_ask4 = {
    'messages': [
        {
            "role": "user",
            "content": """how many streetlights do i have in my database, and I also want total energy output for all time. Also would like to know all the errors and for which streetlight they occured on in the past month"""
        }],
    'messageSummary': ''
}

streetlighting_data_report_ask5 = {
    'messages': [
        {
            "role": "user",
            "content": """lets combine flickering report and power outage report, give me that for past 3 months"""
        }],
    'messageSummary': ''
}

streetlighting_failure_troubleshooting_ask = {
    'messages': [
        {
            "role": "user",
            "content": """can you look into the streetlights with the most recent flickering failures, why are they having that issue?"""
        }],
    'messageSummary': ''
}

internal_report_ask = {
    'messages': [
        {
            "role": "user",
            "content": "can i have an internal streetlighting report over the past 3 months"
        }],
    'messageSummary': ''
}

energy_consumption_predict_ask1 = {
    'messages': [
        {
            "role": "user",
            "content": """help me predict my energy consumption data for the next 3 months. Specifically for project 1"""
        }],
    'messageSummary': ''
}

#can you predict the energy consumption for project 1 for next 2 days?
energy_consumption_predict_ask2 = {
    'messages': [
        {
            "role": "user",
            "content": """can you predict the energy consumption for project 1 for next 2 days?"""
        }],
    'messageSummary': ''
}

debugging_ask = {
    'messages': [
        {
            "role": "user",
            "content": """my streetlight has a power outage alarm, and the power seems to be unstable but within a controlled range,
            what might be the issue?"""
        }],
    'messageSummary': ''
}

title_ask = {
    'messages': [
        {
            "role": "user",
            "content": """my streetlight has a power outage alarm, and the power seems to be unstable but within a controlled range,
            what might be the issue?"""
        }],
    'messageSummary': '',
    'title': True
}

title_ask2 = {
    'messages': [
        {
            "role": "user",
            "content": """你好"""
        }],
    'messageSummary': '',
    'title': True
}

nihao_ask = {
    'messages': [
        {
            "role": "user",
            "content": """你好"""
        }],
    'messageSummary': ''
}
chinese_ask = {
    'messages': [
        {
            "role": "user",
            "content": """你好 it is working, but when I asked 未来 12 小时内，设备 100880 是否会出现故障？"""
        }],
    'messageSummary': ''
}
mingzi_ask = {
    'messages': [
        {
            "role": "user",
            "content": """你叫什么名字"""
        }],
    'messageSummary': ''
}

bye_ask = {
    'messages': [
        {
            "role": "user",
            "content": """bye"""
        }],
    'messageSummary': ''
}

trend_analysis_ask1 = {
    'messages': [
        {
            "role": "user",
            "content": """Can you analyze the failure trends for streetlights in the past month?"""
        }],
    'messageSummary': ''
}

trend_analysis_ask2 = {
    'messages': [
        {
            "role": "user",
            "content": """What are the abnormal energy consumption patterns for streetlights in Zone 5?"""
        }],
    'messageSummary': ''
}

trend_analysis_ask3 = {
    'messages': [
        {
            "role": "user",
            "content": """Show me the overall failure trends for streetlights over the past 3 month."""
        }],
    'messageSummary': ''
}

michael_ask = {
    'messages': [
        {
            "role": "user",
            "content": """can i have a flickering report on over the past year"""
        }],
    'messageSummary': ''
}
import re
import json
import sys
import traceback
from datetime import datetime
import os
from io import StringIO


def extract_query_from_input(input_data):
    """
    Extract the query from input data.
    
    Args:
        input_data: The input data for the test
    
    Returns:
        str: The extracted query or empty string if not found
    """
    # Try to extract from complex format
    if isinstance(input_data, dict) and 'messages' in input_data:
        for msg in input_data['messages']:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                # Look for user query in the format {"user_query": "..."}
                user_query_match = re.search(r'"user_query":\s*"([^"]+)"', content)
                if user_query_match:
                    return user_query_match.group(1)
                return content
    
    # If input_data is a string, try to extract user_query
    if isinstance(input_data, str):
        user_query_match = re.search(r'"user_query":\s*"([^"]+)"', input_data)
        if user_query_match:
            return user_query_match.group(1)
    
    # Default case
    return str(input_data)

class OutputCapture:
    def __init__(self):
        """Initialize the output capture - careful not to interfere with logging"""
        self.original_stdout = sys.stdout
        self.captured_output = StringIO()
        # Don't touch stderr to preserve error messages
        
    def start(self):
        """Start capture but only redirect stdout (not stderr)"""
        # Create a new StringIO for each capture to ensure clean state
        self.captured_output = StringIO()
        # Only replace stdout, leave stderr untouched
        sys.stdout = self
        
    def stop(self):
        """Restore original stdout"""
        sys.stdout = self.original_stdout
        
    def get_output(self):
        """Get the captured output"""
        return self.captured_output.getvalue()
    
    # These methods make the class act like a file object
    def write(self, text):
        """Write to both the capture buffer and original stdout"""
        self.captured_output.write(text)
        self.original_stdout.write(text)
        
    def flush(self):
        """Implement flush method to make this a proper file-like object"""
        self.captured_output.flush()
        self.original_stdout.flush()

def create_jsonl_report(test_results, output_dir=r"C:\Users\leela\Downloads", filename=None):
    """
    Create a JSONL report file with simplified test information, properly including function calls.
    
    Args:
        test_results: List of test result tuples (name, input_data, result)
        output_dir: Directory to save the report
        filename: Optional filename for the report
        
    Returns:
        str: Path to the created JSONL file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.jsonl"
    
    jsonl_file = os.path.join(output_dir, filename)
    
    # Prepare JSONL data
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for name, input_data, result in test_results:
            # Extract information using our improved function
            info = extract_key_information(result)
            query = extract_query_from_input(input_data)
            
            # Parse column classification
            columns = []
            if info['relevant_columns'] != 'N/A':
                try:
                    # Try to parse the JSON array of columns
                    columns_str = info['relevant_columns'].strip()
                    if columns_str.startswith('[') and columns_str.endswith(']'):
                        columns = json.loads(columns_str)
                except json.JSONDecodeError:
                    # If parsing fails, try to extract column names with regex
                    column_matches = re.findall(r'"([^"]+)"', info['relevant_columns'])
                    if column_matches:
                        columns = column_matches
            
            # Process function calls - use a more robust approach
            function_calls = []
            for call in info['function_calls']:
                try:
                    # Parse each function call
                    # If it's already a JSON string, parse it
                    func_obj = json.loads(call)
                    function_calls.append(func_obj)
                except json.JSONDecodeError:
                    # If it can't be parsed as JSON, try to extract with regex
                    try:
                        # Look for the function name and parameters
                        func_match = re.search(r'{"function":\s*"([^"]+)",\s*"parameters":\s*({.+})}', call, re.DOTALL)
                        if func_match:
                            func_name = func_match.group(1)
                            params_str = func_match.group(2)
                            
                            # Try to parse parameters
                            try:
                                params = json.loads(params_str)
                                func_obj = {"function": func_name, "parameters": params}
                                function_calls.append(func_obj)
                            except json.JSONDecodeError:
                                # If we can't parse parameters as JSON, use a simplified approach
                                func_obj = {
                                    "function": func_name,
                                    "parameters": {"raw_params": params_str}
                                }
                                function_calls.append(func_obj)
                        else:
                            # If we can't find function/parameters pattern, add as raw data
                            func_obj = {"function": "unknown", "parameters": {"raw_data": call}}
                            function_calls.append(func_obj)
                    except Exception as e:
                        # Last resort fallback
                        print(f"Warning: Failed to process function call: {call}. Error: {str(e)}")
            
            # Create the JSONL entry
            entry = {
                "question": query,
                "actual": {
                    "function_classification": info['classification'],
                    "column_classification": columns,
                    "function_calling": function_calls,
                    "sql_query": info['mysql_queries']
                }
            }
            
            # Write the entry as a JSON line
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"JSONL report saved to: {jsonl_file}")
    return jsonl_file

def extract_key_information(output_text):
    """
    Extract key information from the test output text with improved function call detection
    and appropriate timing labels based on function type.
    
    Args:
        output_text (str): The raw output text from the test
        
    Returns:
        dict: Dictionary containing extracted information and timing data
    """
    # Ensure output_text is a string
    if not isinstance(output_text, str):
        output_text = str(output_text)
    
    info = {
        'classification': 'N/A',
        'relevant_columns': 'N/A',
        'function_calls': [],  # List to store unique function calls
        'mysql_queries': [],   # List to store unique SQL queries
        'timing_data': {}      # Dictionary to store timing information
    }
    
    # First extract classification to help determine timing labels
    classification_match = re.search(r'###This is classification:\s+(.*?)(?:###End|\n)', 
                                   output_text, re.DOTALL)
    if classification_match:
        info['classification'] = classification_match.group(1).strip()
    
    # Extract timing information with more specific categories
    timing_info = {}
    
    # Extract function classification timing
    func_class_time_match = re.search(r'###This is LLM API request:.*?###This is LLM API return:.*?(\d+\.\d+) seconds', 
                                    output_text, re.DOTALL)
    if func_class_time_match:
        timing_info['function_classification'] = float(func_class_time_match.group(1))
    
    # Extract column classification timing
    col_class_request = re.search(r'###System: You are a sql column classifier assistant.*?###This is LLM API return:.*?(\d+\.\d+) seconds', 
                                output_text, re.DOTALL)
    if col_class_request:
        timing_info['column_classification'] = float(col_class_request.group(1))
    
    # Extract generation timing - first try to find function call generation time
    func_call_match = re.search(r'###This is first LLM input:.*?###This is first LLM return:.*?Response generation took (\d+\.\d+) seconds', 
                              output_text, re.DOTALL)
    
    # If not found, try alternative patterns
    if not func_call_match:
        func_call_match = re.search(r'###This is first LLM input:.*?###This is LLM API return:.*?<\|im_end\|>\s*###\s*Response generation took (\d+\.\d+) seconds', 
                                  output_text, re.DOTALL)
    
    if not func_call_match:
        func_call_match = re.search(r'###This is first LLM input:.*?Response generation took (\d+\.\d+) seconds', 
                                  output_text, re.DOTALL)
    
    if func_call_match:
        # Use the appropriate timing label based on function type
        if info['classification'] == 'streetlight_sql':
            timing_info['function_call_generation'] = float(func_call_match.group(1))
        else:
            timing_info['textual_generation'] = float(func_call_match.group(1))
    
    # Extract SQL generation timing
    sql_gen_match = re.search(r'###System: You are a streetlighting_sql assistant.*?(\d+\.\d+) seconds', 
                            output_text, re.DOTALL)
    if sql_gen_match:
        timing_info['mysql_generation'] = float(sql_gen_match.group(1))
    
    # Extract second LLM call timing (response formulation)
    response_match = re.search(r'###This is second LLM input:.*?###This is second LLM return:.*?Response generation took (\d+\.\d+) seconds', 
                             output_text, re.DOTALL)
    if response_match:
        timing_info['response_formulation'] = float(response_match.group(1))
    
    # Fallback to catch any other response generation times
    all_response_times = re.finditer(r'Response generation took (\d+\.\d+) seconds', output_text)
    for idx, match in enumerate(all_response_times):
        time_value = float(match.group(1))
        # Only add if not already captured in a specific category
        if time_value not in timing_info.values():
            # Create a unique key for each unidentified response time
            timing_info[f'processing_step_{idx+1}'] = time_value
    
    # Extract total execution time if available
    total_time_match = re.search(r'Total execution time: (\d+\.\d+) seconds', output_text)
    if total_time_match:
        timing_info['total_execution'] = float(total_time_match.group(1))
    
    # Add any other timing information found
    other_timing_matches = re.finditer(r'(\w+(?:\s+\w+)*) took (\d+\.\d+) seconds', output_text)
    for match in other_timing_matches:
        operation = match.group(1).lower().replace(' ', '_')
        time_value = float(match.group(2))
        # Only add if not already captured in a specific category
        if time_value not in timing_info.values():
            timing_info[operation] = time_value
    
    info['timing_data'] = timing_info
    
    # Extract all MySQL Queries with labeled pattern
    mysql_matches = re.finditer(r'Generated MySQL Query:[ \t]*(SELECT.*?FROM.*?(?:WHERE.*?)?;)', 
                              output_text, re.DOTALL | re.IGNORECASE)
    for match in mysql_matches:
        query = match.group(1).strip()
        if query not in info['mysql_queries']:  # Avoid duplicates
            info['mysql_queries'].append(query)
    
    # Also try to find SQL queries in LLM returns
    mysql_in_returns = re.finditer(r'###This is LLM API return:.*?SELECT.*?FROM.*?(?:WHERE.*?)?;', 
                                output_text, re.DOTALL | re.IGNORECASE)
    for match in mysql_in_returns:
        sql_match = re.search(r'(SELECT.*?FROM.*?(?:WHERE.*?)?;)', match.group(0), re.DOTALL)
        if sql_match and sql_match.group(1).strip() not in info['mysql_queries']:
            info['mysql_queries'].append(sql_match.group(1).strip())
    
    # Improved function call extraction - handle different formats
    
    # Pattern 1: Look for function call in standard JSON format in LLM API returns
    api_returns = re.finditer(r'###This is LLM API return:.*?<\|im_start\|>assistant<\|im_sep\|>(.*?)<\|im_end\|>', 
                              output_text, re.DOTALL)
    
    for api_return in api_returns:
        return_content = api_return.group(1).strip()
        
        # Try to parse as JSON
        try:
            json_obj = json.loads(return_content)
            
            if isinstance(json_obj, dict) and 'function' in json_obj and 'parameters' in json_obj:
                func_str = json.dumps(json_obj, ensure_ascii=False)
                
                if func_str not in info['function_calls']:
                    info['function_calls'].append(func_str)
        except json.JSONDecodeError:
            # If not valid JSON, try pattern matching in next steps
            pass
    
    # Pattern 2: Look for function calls with pattern {"function": "name", "parameters": {...}}
    function_pattern = r'{"function":\s*"([^"]+)",\s*"parameters":\s*({[^}]*})}'
    function_matches = re.finditer(function_pattern, output_text, re.DOTALL)
    
    for match in function_matches:
        func_name = match.group(1)
        params_str = match.group(2)
        
        try:
            # Try to parse parameters as JSON
            params = json.loads(params_str)
            func_obj = {"function": func_name, "parameters": params}
            func_str = json.dumps(func_obj, ensure_ascii=False)
            
            if func_str not in info['function_calls']:
                info['function_calls'].append(func_str)
        except json.JSONDecodeError:
            # If parsing fails, add the raw match
            raw_match = match.group(0)
            if raw_match not in info['function_calls']:
                info['function_calls'].append(raw_match)
    
    # Extract relevant columns
    columns_match = re.search(r'relevant columns:\s*(\[.*?\])', output_text, re.DOTALL | re.IGNORECASE)
    if columns_match:
        info['relevant_columns'] = columns_match.group(1).strip()
    
    return info
def run_test(name, func, input_data):
    """
    Run a single test and capture output while preserving all console output
    
    Args:
        name (str): Name of the test
        func (callable): Function to test
        input_data: Input data for the test
        
    Returns:
        tuple: (name, input_data, result)
    """
    # Create a new output capture for each test
    output_capture = OutputCapture()
    output_capture.start()
    
    print(f"Running test: {name}")
    print(f"Original input: {input_data}")
    
    try:
        # Call the function - logging and stderr will go directly to console
        func(input_data)
        print(f"Test '{name}' completed.")
    except Exception as e:
        error_message = f"Error in test '{name}':\n{traceback.format_exc()}"
        print(error_message, file=sys.stderr)  # Will go directly to stderr
    finally:
        print(f"========== END OF TEST: {name} ==========")
        output_capture.stop()
        
        # Use the captured output
        result = output_capture.get_output()
        
        # Handle potential Unicode issues
        try:
            # Test if result contains characters that might cause issues
            result.encode('ascii', 'strict')
        except UnicodeEncodeError:
            # If the test fails, we know there are non-ASCII characters
            print(f"Warning: Output for test '{name}' contains non-ASCII characters")
        
    return name, input_data, result

def run_selected_tests(tests, *selected_tests, create_reports=False, summary_filename="summary_report.txt", jsonl_filename="test_results.jsonl"):
    """
    Run selected tests or all tests if none selected and optionally create reports
    """
    if not tests:
        print("No tests provided!")
        return []
        
    if selected_tests:
        selected_tests = [test.lower() for test in selected_tests]
        tests_to_run = [test for test in tests if any(selected.lower() in test[0].lower() for selected in selected_tests)]
        if not tests_to_run:
            print(f"No tests matched the selection: {selected_tests}")
            return []
    else:
        tests_to_run = tests

    results = []
    print(f"Running {len(tests_to_run)} tests: {[test[0] for test in tests_to_run]}")
    
    for test_name, test_func, test_arg in tests_to_run:
        # Each test gets its own run with its own output capture
        result = run_test(test_name, test_func, test_arg)
        results.append(result)

    # Only create reports if the flag is set to True
    if create_reports:
        summary_file = create_summary_report(results, filename=summary_filename)
        jsonl_file = create_jsonl_report(results, filename=jsonl_filename)
        
        print(f"Summary report saved to: {summary_file}")
        print(f"JSONL report saved to: {jsonl_file}")
        
        return results, summary_file, jsonl_file
    
    return results

def create_summary_report(test_results, output_dir=r"C:\Users\leela\Downloads", filename=None):
    """
    Create a single summary report with revised format and more detailed timing information.
    
    Args:
        test_results: List of test result tuples (name, input_data, result)
        output_dir: Directory to save the report
        filename: Optional filename for the report
        
    Returns:
        str: Path to the created summary file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_report_{timestamp}.txt"
    
    summary_file = os.path.join(output_dir, filename)
    
    # Use UTF-8 encoding when writing the file
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("TEST SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of tests: {len(test_results)}\n\n")
        
        for i, (name, input_data, result) in enumerate(test_results, 1):
            # Extract query from input data
            query = extract_query_from_input(input_data)
            
            # Extract information from THIS test's output ONLY
            info = extract_key_information(result)
            
            f.write(f"Test #{i}: {name}\n")
            f.write(f"Query: {query}\n\n")
            
            # Write function classification (renamed from just "Classification")
            f.write(f"Function Classification: {info['classification']}\n\n")
            
            # Write relevant columns
            f.write(f"Relevant Columns: {info['relevant_columns']}\n\n")
            
            # Write all function calls
            f.write(f"Function Calls:\n")
            if info['function_calls']:
                for j, call in enumerate(info['function_calls'], 1):
                    # Ensure the call is properly formatted JSON
                    try:
                        # Try to parse and re-dump to ensure proper formatting
                        parsed_call = json.loads(call)
                        formatted_call = json.dumps(parsed_call, ensure_ascii=False, indent=2)
                        f.write(f"{j}. {formatted_call}\n")
                    except json.JSONDecodeError:
                        # If parsing fails, use the original string
                        f.write(f"{j}. {call}\n")
            else:
                f.write("None found\n")
            f.write("\n")
            
            # Write all MySQL queries
            f.write(f"Generated MySQL Queries:\n")
            if info['mysql_queries']:
                for j, query in enumerate(info['mysql_queries'], 1):
                    f.write(f"{j}. {query}\n")
            else:
                f.write("None found\n")
            f.write("\n")
            
            # Write timing information with more descriptive labels
            f.write("Timing Information:\n")
            
            # Define the order and labels for timing information
            # Include both function_call_generation and textual_generation
            timing_order = [
                ('function_classification', 'Function Classification Time'),
                ('column_classification', 'Column Classification Time'),
                ('function_call_generation', 'Function Call Generation Time'),
                ('textual_generation', 'Textual Generation Time'),
                ('mysql_generation', 'MySQL Generation Time'),
                ('response_formulation', 'Final Response Formulation Time'),
                ('total_execution', 'Total Execution Time')
            ]
            
            if info['timing_data']:
                # Display timing data in the specified order
                for key, label in timing_order:
                    if key in info['timing_data']:
                        f.write(f"- {label}: {info['timing_data'][key]:.4f} seconds\n")
                
                # Display any other timing information not in the predefined order
                for key, value in info['timing_data'].items():
                    if key not in [item[0] for item in timing_order]:
                        # Convert snake_case to Title Case with spaces
                        readable_key = ' '.join(word.capitalize() for word in key.split('_'))
                        f.write(f"- {readable_key}: {value:.4f} seconds\n")
            else:
                f.write("No timing information found\n")
            
            f.write("\n" + "-" * 80 + "\n\n")
    
    print(f"Summary report saved to: {summary_file}")
    return summary_file
tests = [
           ("nihao Ask", ask, nihao_ask),
    #   ("Michael Ask", ask, michael_ask),
    #    ("Bye Ask", ask, bye_ask), #issue with classification
    #   ("Irrelevant Ask", ask, irrelevant_ask),
    #   ("RAG Ask", ask, rag_ask),
    #  ("Streetlight Data Query Ask 1", ask, streetlight_data_query_ask1),
    #  ("Streetlight Data Query Ask 2", ask, streetlight_data_query_ask2),
    # ("Streetlight Data Query Ask 3", ask, streetlight_data_query_ask3),
    # ("Streetlight Data Query Ask 4", ask, streetlight_data_query_ask4),
    # ("Streetlight Data Query Ask 5", ask, streetlight_data_query_ask5),
    # ("Streetlight Data Query Ask 6", ask, streetlight_data_query_ask6),
    # ("Streetlight Data Query Ask 7", ask, streetlight_data_query_ask7),
    #("Streetlight Failure Prediction Ask", ask, streetlight_failure_prediction_ask),
    # ("Streetlight Failure Prediction Ask 2", ask, streetlight_failure_prediction_ask2),

    #    ("Streelighting Data Report Ask 1", ask, streetlighting_data_report_ask1),
    #    ("Streelighting Data Report Ask 2", ask, streetlighting_data_report_ask2),
    #    ("Streelighting Data Report Ask 3", ask, streetlighting_data_report_ask3),
    #    ("Streelighting Data Report Ask 5", ask, streetlighting_data_report_ask5),

    #   ("Internal Report Ask", ask, internal_report_ask),#sql wrong
    #  ("Energy Consumption Predict Ask 1", ask, energy_consumption_predict_ask1),#return json as first llm return
    #    ("Energy Consumption Predict Ask 2", ask, energy_consumption_predict_ask2),#return natural language as first llm return
    #  ("Title Ask", ask, title_ask),
    #   ("Streetlight Failure Troubleshooting Ask", ask, streetlighting_failure_troubleshooting_ask),
    #   ("Trend Analysis Ask 1", ask, trend_analysis_ask1), #wrong transformed postGresql
    #    ("Trend Analysis Ask 2", ask, trend_analysis_ask2), #looks good
    #    ("Trend Analysis Ask 3", ask, trend_analysis_ask3),#new model, wrong transformed postGresql
    
]


if __name__ == '__main__':
    results = run_selected_tests(tests, create_reports=False)
    #results = run_selected_tests("Internal Report Ask") #time out error
    #results = run_selected_tests("Trend Analysis Ask 1")#no IOT data, much better without classified col prompt
    #results = run_selected_tests("Trend Analysis Ask 2")#query has issue
    #results = run_selected_tests("Trend Analysis Ask 3")#works, and summary stas look good, context reduced, better with classified col prompt

    #results = run_selected_tests("Hi Ask") 
    #results = run_selected_tests("Streetlight Data Query Ask 1")#no IOT data
    #results = run_selected_tests("Streetlight Data Query Ask 2")


    #results = run_selected_tests("Streelighting Data Report Ask 1")#AttributeError: 'list' object has no attribute 'items'
    #results = run_selected_tests("Streelighting Data Report Ask 2")

    #results = run_selected_tests("Streetlight Failure Prediction Ask")
    #results = run_selected_tests("Energy Consumption Predict Ask 1")#time out classification ? 
    #results = run_selected_tests("Energy Consumption Predict Ask 2")