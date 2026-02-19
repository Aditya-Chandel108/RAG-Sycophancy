import requests
import json
import sys
from config import API_ENDPOINT, HEADERS, MODEL_NAME, TEMPERATURE
from prompts import *
from simulation_engine import HostileRetriever

class ReflectiveRAG:
    def __init__(self):
        self.retriever = HostileRetriever()

    def call_model(self, system_msg, user_msg):
        """
        Executes the API call to the configured LLM.
        """
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            "temperature": TEMPERATURE
        }
        
        # here add the api of the model
        
        # Example Implementation for Requests:
        # try:
        #     response = requests.post(API_ENDPOINT, headers=HEADERS, json=payload)
        #     response.raise_for_status()
        #     return response.json()['choices'][0]['message']['content']
        # except Exception as e:
        #     return f"ERROR: {str(e)}"
        
        return "NOISY" # Placeholder for structural integrity

    def run_pipeline(self, query, simulate_poison=True):
        """
        Main logic flow: Retrieve -> Audit -> Generate.
        """
        # 1. Retrieval (Simulated as Hostile for the challenge)
        if simulate_poison:
            docs = self.retriever.retrieve(query)
        else:
            docs = ["No context found for this simulation."]
            
        context = " ".join(docs)

        # 2. Reflection / Audit Step
        # This decouples retrieval relevance from generation logic.
        audit_status = self.call_model(AUDITOR_SYSTEM_PROMPT, get_audit_prompt(query, context))
        
        # 3. Adaptive Generation
        # Determines if the system should trust context or fallback to internal knowledge.
        if "NOISY" in audit_status.upper():
            sys_prompt = GENERATOR_NOISY_PROMPT
        else:
            sys_prompt = GENERATOR_TRUSTED_PROMPT
            
        final_answer = self.call_model(sys_prompt, get_generation_prompt(query, context))
        
        return audit_status, final_answer

def main():
    rag_system = ReflectiveRAG()
    
    print("Type 'exit' to quit. Mode: Hostile Retrieval Active.\n")

    while True:
        user_query = input("User Query: ")
        
        if user_query.lower() in ['exit', 'quit']:
            break  
        if not user_query.strip():
            continue

        status, result = rag_system.run_pipeline(user_query)
        
        print(f"\n[Auditor Result]: {status}")
        print(f"[System Response]: {result}")

if __name__ == "__main__":
    main()
