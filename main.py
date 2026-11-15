"""
main.py
=======
Entry point for the RAG-Sycophancy project.

Contains
--------
ReflectiveRAG       – Original sycophancy-detection pipeline (unchanged).
RAGSycophancySystem – New wrapper that combines real RAGPipeline with
                      the sycophancy auditor from ReflectiveRAG.
main()              – Interactive menu offering both modes.
"""

import requests
import json
import sys
from config import API_ENDPOINT, HEADERS, MODEL_NAME, TEMPERATURE
from prompts import *
from simulation_engine import HostileRetriever

# ---------------------------------------------------------------------------
# Import RAGPipeline (graceful fallback if rag_pipeline not yet installed)
# ---------------------------------------------------------------------------
try:
    from rag_pipeline import RAGPipeline, RAGConfig
    RAG_PIPELINE_AVAILABLE = True
except ImportError as _e:
    RAG_PIPELINE_AVAILABLE = False
    print(f"[main] rag_pipeline import failed: {_e}. "
          "Real-RAG mode will be unavailable.")


# ===========================================================================
# ReflectiveRAG  (original – do not modify)
# ===========================================================================

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


# ===========================================================================
# RAGSycophancySystem  (new unified system)
# ===========================================================================

class RAGSycophancySystem:
    """
    Integrates the real RAGPipeline with the ReflectiveRAG sycophancy auditor.

    Flow
    ----
    1. RAGPipeline.query()  retrieves real documents and generates a base answer.
    2. If check_sycophancy=True, the retrieved context is passed through the
       ReflectiveRAG auditor to detect noisy / poisoned context.
    3. The final answer is adjusted based on the audit result.

    Parameters
    ----------
    config : Optional RAGConfig.  If None, defaults are used.
    """

    def __init__(self, config: "RAGConfig | None" = None):
        if not RAG_PIPELINE_AVAILABLE:
            raise ImportError(
                "rag_pipeline is not available. "
                "Make sure the rag_pipeline/ folder exists and all "
                "dependencies are installed."
            )
        self.rag_pipeline = RAGPipeline(config)
        self.sycophancy_checker = ReflectiveRAG()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, source: str) -> dict:
        """
        Ingest documents into the RAG pipeline.

        Parameters
        ----------
        source : Path to a file or directory.

        Returns
        -------
        Ingestion stats dict from RAGPipeline.ingest().
        """
        return self.rag_pipeline.ingest(source)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, user_query: str, check_sycophancy: bool = True) -> dict:
        """
        Full pipeline: retrieve real docs → (optionally) audit → generate.

        Parameters
        ----------
        user_query        : User's natural-language question.
        check_sycophancy  : If True, run the sycophancy auditor on the
                            retrieved context before generating the answer.

        Returns
        -------
        dict with keys:
            answer            : Final answer string.
            audit_status      : 'TRUSTED' | 'NOISY' | 'SKIPPED'.
            retrieved_docs    : List of retrieved chunk dicts.
            sycophancy_check  : dict with auditor details (or None).
            retrieval_mode    : Mode used for retrieval.
            num_retrieved     : Number of retrieved chunks.
        """
        # 1. Retrieve + generate (base answer from RAGPipeline)
        rag_result = self.rag_pipeline.query(user_query, use_history=True)

        base_answer = rag_result["answer"]
        context = rag_result["context_used"]
        retrieved_docs = rag_result["retrieved_docs"]

        audit_status = "SKIPPED"
        sycophancy_check = None
        final_answer = base_answer

        if check_sycophancy and context.strip():
            # 2. Audit the retrieved context for sycophancy / noise
            audit_status = self.sycophancy_checker.call_model(
                AUDITOR_SYSTEM_PROMPT,
                get_audit_prompt(user_query, context),
            )

            sycophancy_check = {
                "audit_status": audit_status,
                "context_audited": context[:500] + ("…" if len(context) > 500 else ""),
            }

            # 3. Re-generate with the appropriate system prompt
            if "NOISY" in audit_status.upper():
                # Context is unreliable — use internal knowledge
                final_answer = self.sycophancy_checker.call_model(
                    GENERATOR_NOISY_PROMPT,
                    get_generation_prompt(user_query, context),
                )
            else:
                # Context is trusted — answer grounded in retrieved docs
                final_answer = self.sycophancy_checker.call_model(
                    GENERATOR_TRUSTED_PROMPT,
                    get_generation_prompt(user_query, context),
                )

        return {
            "answer": final_answer,
            "audit_status": audit_status,
            "retrieved_docs": retrieved_docs,
            "sycophancy_check": sycophancy_check,
            "retrieval_mode": rag_result["retrieval_mode"],
            "num_retrieved": rag_result["num_retrieved"],
        }

    # ------------------------------------------------------------------
    # Simple chat interface
    # ------------------------------------------------------------------

    def chat(self, user_query: str) -> str:
        """
        Submit a query and return the final answer string.

        Parameters
        ----------
        user_query : User's question.

        Returns
        -------
        Answer string.
        """
        result = self.query(user_query, check_sycophancy=True)
        return result["answer"]


# ===========================================================================
# main() – interactive menu
# ===========================================================================

def _print_banner() -> None:
    print("\n" + "=" * 60)
    print("  RAG-Sycophancy Detection System")
    print("=" * 60)
    print("  Mode 1 : Hostile Retrieval Simulation (original)")
    print("  Mode 2 : Real RAG Pipeline + Sycophancy Audit")
    print("  q / quit : Exit")
    print("=" * 60 + "\n")


def _run_mode1() -> None:
    """Run the original sycophancy simulation mode."""
    system = ReflectiveRAG()
    print("\n[Mode 1] Hostile Retrieval Active. Type 'back' to return to menu.\n")
    while True:
        user_query = input("Query: ").strip()
        if user_query.lower() in ("back", "exit", "quit", ""):
            break
        status, result = system.run_pipeline(user_query)
        print(f"\n  [Auditor] : {status}")
        print(f"  [Answer]  : {result}\n")


def _run_mode2() -> None:
    """Run the real RAG + sycophancy audit mode."""
    if not RAG_PIPELINE_AVAILABLE:
        print("\n[Mode 2] rag_pipeline is not available. "
              "Install dependencies and ensure rag_pipeline/ folder exists.\n")
        return

    config = RAGConfig()
    system = RAGSycophancySystem(config)

    print("\n[Mode 2] Real RAG + Sycophancy Audit. Type 'back' to return to menu.\n")
    print("  Commands:")
    print("    ingest <path>  – ingest a file or directory into the vector store")
    print("    stats          – show collection / pipeline stats")
    print("    reset          – clear conversation history")
    print("    <any question> – query the pipeline\n")

    while True:
        raw = input("Query: ").strip()
        if not raw:
            continue
        if raw.lower() in ("back", "exit", "quit"):
            break

        if raw.lower().startswith("ingest "):
            source = raw[7:].strip()
            stats = system.ingest(source)
            print(f"\n  [Ingest] {stats}\n")

        elif raw.lower() == "stats":
            stats = system.rag_pipeline.get_stats()
            print(f"\n  [Stats] {json.dumps(stats, indent=4)}\n")

        elif raw.lower() == "reset":
            system.rag_pipeline.reset_history()

        else:
            result = system.query(raw, check_sycophancy=True)
            print(f"\n  [Audit]   : {result['audit_status']}")
            print(f"  [Answer]  : {result['answer']}")
            print(f"  [Sources] : {result['num_retrieved']} chunk(s) retrieved "
                  f"({result['retrieval_mode']} mode)")
            if result.get("sycophancy_check"):
                sc = result["sycophancy_check"]
                print(f"  [Check]   : audit_status={sc['audit_status']}")
            print()


def main():
    _print_banner()

    while True:
        choice = input("Select mode (1 / 2 / q): ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("Goodbye.")
            sys.exit(0)
        elif choice == "1":
            _run_mode1()
        elif choice == "2":
            _run_mode2()
        else:
            print("  Please enter 1, 2, or q.\n")


if __name__ == "__main__":
    main()
