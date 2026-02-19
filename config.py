import os

# Configuration for the LLM API
# Replace with your specific provider's endpoint (e.g., OpenAI, Anthropic, or local Ollama)
API_ENDPOINT = "link of api provider"
MODEL_NAME = " your preferred model" 
TEMPERATURE = 0.0     # Kept at 0 for factual checking

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('MODEL_API_KEY', 'your-key')}"
}
