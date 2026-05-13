# vuln-chat

A RAG pipeline for querying NVD CVE records using natural language. Supports two retrieval backends (FAISS and LightRAG), a LangGraph-based agent with tool calling, and an evaluation framework with synthetic test generation and LLM-as-Judge scoring.

## Quickstart

1. Copy `.env.example` to `.env` and fill in the required values

2. Build and start all services:
    ```bash
    docker compose up --build -d
    ```

3. Load the default CVE dataset:
    ```bash
    curl -X POST http://localhost:8001/load/defaults
    ```

    Or load specific CVEs:
    ```bash
    curl -X POST http://localhost:8001/load \
      -H "Content-Type: application/json" \
      -d '{"cve_ids": ["CVE-2024-3094", "CVE-2023-44487"]}'
    ```

4. Ask a question:
    ```bash
    curl -X POST http://localhost:8002/chat \
      -H "Content-Type: application/json" \
      -d '{"message": "What is the CVSS score and attack vector for Log4Shell?"}'
    ```

## Project Structure

| Directory | Description |
|---|---|
| `vuln_agent/` | LangGraph agent with tool calling. Accepts natural language questions and queries the retrieval backends to produce grounded answers. |
| `vuln_loader/` | CVE ingestion service. Fetches records from the NVD API and loads them into both FAISS and LightRAG. |
| `faiss/` | FAISS vector store service. Pure semantic similarity search over CVE descriptions with metadata filtering. |
| `LightRAG/` | LightRAG hybrid retrieval service. Knowledge graph + vector search for multi-hop and relational queries. |
| `evaluation/` | Evaluation framework. Includes synthetic test generation, LLM-as-Judge scoring, and result outputs. |

## Evaluation

The evaluation framework measures 5 dimensions: Faithfulness, Groundedness, Answer Relevancy, Context Precision, and Context Recall. Test cases include 5 manually written and 9 synthetically generated question/answer pairs across 3 personas and 3 question breadths.

To reproduce:
1. Run `evaluation/generate_testset.ipynb` to generate synthetic test data
2. Run `evaluation/eval.ipynb` to run the evaluation against both retrievers

Results and detailed analysis are in [FINDINGS.md](FINDINGS.md).

## Findings

See [FINDINGS.md](FINDINGS.md) for:
- System architecture and design choices
- Evaluation results with interpretation
- Failure case analysis
- Improvement roadmap (4 hours and 4 weeks)
- Recommendation on using the system for automated vulnerability triage
