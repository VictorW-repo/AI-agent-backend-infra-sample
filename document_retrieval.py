import warnings
import json
import torch
import numpy as np
from typing import List, Dict
from pymilvus import MilvusClient, MilvusException
import warnings
from query_api import rag_model_api, rag_tokenizer_api


class DocumentRetrieval:    
    def __init__(self):
        self.client = self.connect_to_milvus()
        self.rag_model_api = rag_model_api
        self.rag_tokenizer_api = rag_tokenizer_api

    def connect_to_milvus(self):
        try:
            client = MilvusClient(uri="http://71.228.14.203")
            client.list_collections() 
            return client
        except MilvusException as e:
            warnings.warn(f"Failed to connect to Milvus: {e}. Some features will be unavailable.", RuntimeWarning)
            return None

    def is_milvus_available(self):
        return self.client is not None

    def _embed_documents(self, documents: List[str]) -> torch.Tensor:
        return self.rag_model_api(json.dumps(documents))
    
    @staticmethod
    def _cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(tensor1, tensor2).item()

    def calculate_similarity(self, doc1: str, doc2: str) -> float:
        embeddings = self._embed_documents([doc1, doc2])
        similarity = torch.nn.functional.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
        return similarity.item()
    
    def find_multi_similarities(self, query: str) -> Dict[str, float]:
        """ Calculate similarities between the query and pre-embedded descriptions """
        query_embedding = self._embed_documents([query]).squeeze(0)
        similarities = {}
        for key, emb in self.embedded_descriptions.items():
            similarities[key] = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), emb.unsqueeze(0)).item()
        return similarities

    def get_similar_documents(self, query_vector: np.ndarray, k: int = 10) -> List[str]:
        if not self.is_milvus_available():
            print("Milvus is not available. Similar document retrieval skipped.")
            return []

        query_vector_list = query_vector.cpu().numpy().tolist()

        try:
            search_results = self.client.search(
                collection_name="Prototype5",     
                data=query_vector_list,                
                limit=k,      
                output_fields=['text']     
            )
            
            results = [(result['entity'].get('text', 'No text available'), result['distance']) for result in search_results[0]]
            return results
        except Exception as e:
            print(f"An error occurred while searching Milvus: {e}")
            return []
    
    def choose_rag_docs(self, chunks: List[str], max_chunks: int = 5) -> List[str]:
        response_paragraphs = []
        seen_paragraphs = set()
        seen_embeddings = {}  

        for chunk in chunks:
            if chunk not in seen_paragraphs:
                chunk_embedding = self._embed_documents([chunk])
                seen_paragraphs.add(chunk)
                seen_embeddings[chunk] = chunk_embedding

                is_unique = True
                for seen_chunk, seen_emb in seen_embeddings.items():
                    if seen_chunk != chunk and self._cosine_similarity(chunk_embedding, seen_emb) >= 0.8:
                        is_unique = False
                        break

                if is_unique:
                    response_paragraphs.append(chunk)
                    if len(response_paragraphs) >= max_chunks:
                        break

        return response_paragraphs