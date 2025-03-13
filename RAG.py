from document_retrieval import DocumentRetrieval
from query_api import llm_api

doc_retrieval = DocumentRetrieval()
    
def critic_eval(query, context):
    eval_ask = [{'role': 'user', 'content': '###System: You are a LLM critic performing relevance ###User: Task instruction: ' + query + "\nEvidence: " + context}]
    response = llm_api(eval_ask)
    return response

def classify_task(user_query, num_docs=15):  # name changed from rag_call() to reflect actual purpose
    """
    Classifies the user query to determine the appropriate handler.
    The num_docs parameter is kept for backward compatibility but is no longer used.
    
    Args:
        user_query (str): The user's query text
        num_docs (int): Unused parameter, kept for compatibility
        
    Returns:
        tuple: (empty list, classification results)
    """
    function_ask = [{'role': 'user', 'content': '###System: Classify the following function ###User: ' + user_query}]
    class_results = llm_api(function_ask)
    return [], class_results



def direct_rag(user_query, num_docs = 15):
    final_collection = []

    query_vector = doc_retrieval._embed_documents(user_query)
    top_docs_with_scores = doc_retrieval.get_similar_documents(query_vector, num_docs)

    sorted_docs = sorted(top_docs_with_scores, key=lambda x: x[1], reverse=True)
    sorted_doc_scores = {doc: score for doc, score in sorted_docs}
    collected = doc_retrieval.choose_rag_docs(sorted_doc_scores)
    for chunk in collected:
        relevancy = critic_eval(query=user_query, context=chunk)
        if relevancy == "[Relevant]":
            final_collection.append(chunk)
    return final_collection



#  only useful function in this file for now is classify_task(), all others are not deleted because we might get RAG back in the future
