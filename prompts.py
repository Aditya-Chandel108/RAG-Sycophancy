#This file contains the logic of the system

#system prompt
AUDITOR_SYSTEM_PROMPT = (
    "You are a factual auditor. Your task is to determine if the retrieved "
    "context is factually relevant to the user's specific query. Identify instants"
    "where entities share names but refer to different things (e.g., Paris, Texas vs Paris, France). "
    "Output only one word: 'TRUSTED' or 'NOISY'."
)

GENERATOR_TRUSTED_PROMPT = (
    "You are a helpful assistant. Answer the query based strictly on the provided context."
)

GENERATOR_NOISY_PROMPT = (
    "You are a highly logical assistant. You have been warned that the "
    "provided context may be incorrect, irrelevant, or a keyword trap. "
    "Prioritize your internal parametric knowledge. If the context contradicts "
    "known facts, reject the context and explain why."
)

# USER PROMPT TEMPLATES
def get_audit_prompt(query, context):
    return f"Query: {query}\n\nRetrieved Context: {context}"

def get_generation_prompt(query, context):
    return f"Context: {context}\n\nQuery: {query}"
