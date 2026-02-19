# Reducing Contextual Sycophancy in RAG Pipelines

## Methodology: Audited-Agentic RAG
Addressing without any fine-tuning how we can reduce the context mismatches in RAG pipelines

1. **Auditing Step**: Before generation, the `Auditor` agent evaluates the context for mismatches.
2. **Dynamic Routing**: If the context is flagged as `NOISY`, the generator is switched to a "Critical Thinking" mode that prioritizes internal knowledge over external noise.

### Reproducibility
1. Install dependencies: `pip install requests`
2. Set API Key in `config.py` or as an environment variable.
3. Run: `python main.py`

### Findings
In a "Hostile Retrieval" scenario, the system correctly identifies the context as NOISY and refuses to hallucinate.
