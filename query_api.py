import requests
import json
import torch
from requests.exceptions import Timeout, HTTPError, RequestException


PORT = 9999
def llm_api(conversation, max_new_tokens=2000, timeout=180, format_type='phi4'):
    url = f"http://gpu.cioic.com:{PORT}/llm_inference"
    # Add charset=utf-8 to Content-Type
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    
    formatted = ""
    for message in conversation:
        if message['role'] == 'user':
            formatted += message['content'] + " "
        elif message['role'] == 'assistant':
            formatted += f"###Assistant: {message['content']} " 
        else:
            formatted += message['content'] + " "
    
    llm_input = formatted.strip()
    input_data = {"content": llm_input, "max_new_tokens": max_new_tokens}
    
    try:
        print("###This is LLM API request: \n", input_data, "\n###")
        # Use json.dumps with ensure_ascii=False and encode to utf-8
        response = requests.post(
            url, 
            headers=headers,
            data=json.dumps(input_data, ensure_ascii=False).encode('utf-8'),
            timeout=timeout
        )
        response.raise_for_status()
        
        # Force UTF-8 encoding for response
        response.encoding = 'utf-8'
        output = response.json()
        
        if "data" in output and "response" in output["data"]:
            raw_response = output['data']['response']
            print("###This is LLM API return: \n", raw_response, "\n###")
            
            if format_type == 'phi4':
                cleaned_response = clean_phi4_response(raw_response)
            elif format_type == 'phi3':
                cleaned_response = clean_phi3_response(raw_response)
            elif format_type == 'mistral':
                cleaned_response = clean_mistral_response(raw_response)
            
            if cleaned_response:
                return cleaned_response
        
        return {"error": "Empty or invalid response", "details": "The API response did not contain the expected data"}

    except Timeout:
        return {"error": "Timeout", "details": f"Request timed out after {timeout} seconds"}
    except HTTPError as http_err:
        return {"error": "HTTP Error", "details": str(http_err)}
    except RequestException as req_err:
        return {"error": "Request Exception", "details": str(req_err)}
    except json.JSONDecodeError as json_err:
        return {"error": "JSON Decode Error", "details": str(json_err)}
    except Exception as e:
        return {"error": "Unexpected Error", "details": str(e)}

def clean_phi4_response(raw_response):
    """Clean response from PHI-4 model output"""
    start_tag = "<|im_start|>assistant<|im_sep|>"
    end_tag = "<|im_end|>"
    
    start_index = raw_response.rfind(start_tag)
    if start_index != -1:
        start_index += len(start_tag)
        end_index = raw_response.find(end_tag, start_index)
        if end_index == -1:
            end_index = raw_response.rfind('\n', start_index)
        if end_index == -1:
            end_index = len(raw_response)
            
        cleaned = raw_response[start_index:end_index].strip()
        
        # Try to parse as JSON and extract response field if present
        try:
            response_json = json.loads(cleaned)
            if isinstance(response_json, dict) and "response" in response_json:
                return response_json["response"]
        except json.JSONDecodeError:
            pass
            
        return cleaned
        
    return raw_response.strip()

def clean_phi3_response(raw_response):
    """Clean response from PHI-3 model output"""
    start_tag = "<|assistant|>"
    end_tag = "<|end|>"
    
    start_index = raw_response.rfind(start_tag)
    if start_index != -1:
        start_index += len(start_tag)
        end_index = raw_response.find(end_tag, start_index)
        if end_index == -1:
            end_index = raw_response.rfind('\n', start_index)
        if end_index == -1:
            end_index = len(raw_response)
        return raw_response[start_index:end_index].strip()
    return raw_response.strip()

def clean_mistral_response(raw_response):
    start_marker = "[/INST]"
    end_marker = "</s>"
    
    start_index = raw_response.find(start_marker)
    if start_index != -1:
        start_index += len(start_marker)
        end_index = raw_response.rfind(end_marker)
        if end_index == -1:
            end_index = len(raw_response)
        return raw_response[start_index:end_index].strip()
    return raw_response.strip()
 
def rag_model_api(input_string, timeout=30):
    url = f'http://gpu.cioic.com:{PORT}/rag_model_inference'
    headers = {'Content-Type': 'text/plain'}
    
    try:
        response = requests.post(url, headers=headers, data=input_string, timeout=timeout)
        response.raise_for_status()
        
        response_data = json.loads(response.text)
        
        if "data" in response_data and "embedding" in response_data["data"]:
            embedding = torch.tensor(response_data["data"]["embedding"])
            return embedding
        
        return None

    except requests.RequestException:
        return None

def rag_tokenizer_api(input_string, timeout=10):
    url = f'http://gpu.cioic.com:{PORT}/rag_tokenizer_encode'
    headers = {'Content-Type': 'text/plain'}
    
    try:
        response = requests.post(url, headers=headers, data=input_string, timeout=timeout)
        response.raise_for_status()
        
        response_data = json.loads(response.text)
        
        if "data" in response_data:
            return response_data["data"]
        
        return None

    except requests.RequestException:
        return None
        

if __name__ == '__main__':
    
    conversation = [{'role': 'user', 'content': "###System: Call streetlight_sql ###User: how many failures are there in my database"}]
    result = llm_api(conversation)
    print("LLM API Test Result:", result)

    print("Classifier API Test Result:", result)

        # def test_rag_model_api(self):
        #     result = rag_model_api("This is a test input for the RAG model.")
        #     print("RAG Model API Test Result:", result)
        #     self.assertIsInstance(result, (torch.Tensor, type(None)))

    # result = streetlight_failure_prediction_api("Test input for streetlight failure prediction")
    # print("Streetlight Failure Prediction API Test Result:", result)
