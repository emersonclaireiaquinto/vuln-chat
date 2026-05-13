# Findings and Documentation

## System Architecture and Design Choices

### LLM Provider: OpenAI (gpt-5.4)
Industry standard, guaranteed to not be a bottleneck in simple RAG applications.
Initially attempted to use self-hosted models (Gemma 4 E4B and Gemma 4 31b), but hit either a bottleneck in latency or quality, depending on the model used. Switched over to gpt-5.4 as a guaranteed alternative

### Embedding Model: text-embedding-3-large (3072 dimensions)
High-quality embeddings for maximum recall across a small but semantically dense CVE corpus. Similar story as with the LLM, attempted to use Qwen3-embedding-4b, which, while fantastic for unstructured text, struggled with the semi-structured, semantically similar text of the CVEs

### FAISS vs LightRAG
The system has two configurable retrieval backends:

- **FAISS** — pure vector similarity search over CVE descriptions. Fast, deterministic, and good for single-hop factual lookups ("What is the CVSS score for X?"). Metadata (severity, CVSS, attack vector, CWE) is stored alongside embeddings and surfaced to the LLM in the tool response.

- **LightRAG** — hybrid knowledge graph + vector search. Builds entity-relationship graphs from CVE text using LLM extraction, enabling multi-hop reasoning ("Which vulnerabilities share the same CWE?") and broader pattern queries. More expensive to index (requires LLM calls during ingestion) but better at relational queries.


Why did I decide to add an entire extra knowledge base?
Personally, knowledge graphs have been a personal fascination of mine, and despite the downsides (mostly latency), I believe more systems should be using them in place of vector databases. This is one of those scenarios, where the source data (CVEs) contain extremely niche, highly technical language that is difficult for embedding models to exctact semantic info from. Knowledge graphs excel at this type of task, where you have an underlying structure to denote relationships between different vulnerabilities (affected software, threat vectors, etc) We'll see later that yes, a knowledge graph outperforms a vector database, despite the small size of the index. (knowledge graphs have been proven to become more effective compared to vector databases as they both scale.)

The two retrieval backends must be handled differently on ingest. 
Vector databases struggle with structured data due to pollution of shared structure and keys, and the irregularity of the text. Embedding models are tuned to extract semantic information from long form, unstructured text.

Knowledge Graphs, on the other hand, handle structured data well due to the LLM entity extraction step. LLMs are able to parse through structured data well, and we are able to fit entity types and underlying structure together, leading to overall better extraction.

### Agent: LangGraph with tool calling
A custom LangGraph tool loop agent. The LLM decides when to call the `search_cve` tool and can iterate (call the tool, read results, call again or answer). This is more flexible than a linear chain. The agent can reformulate queries or ask follow-up searches when initial results are insufficient.

### Reranking: Qwen3-Rerank-4B (optional)
Cross-encoder reranking via a self-hosted model. Configurable per-request. Improves precision when the initial retrieval returns semantically similar but not contextually relevant results.


### Evaluation Framework
One issue commonly faced when evaluating GenAI applications is the lack of man-hours needed to both create test cases and comprehensively verify output over a wide range of inputs and outputs. To solve for this, there are 2 approaches.

#### Synthetic Test Generation
By leveraging LLMs, we can dynamically create synthetic test data grounded in our knowledge source to supplement, or even replace, manually created test cases. To do this, you pick a sample of your knowledge source and have an LLM generate a set of questions over the source, constraining it only to those selected documents. This alone would produce decent test cases, but on its own would only test a fraction of different styles and perspectives you may see in real world use.

To solve for this, you feed in a persona along with your data, and have the LLM generate questions grounded in your knowledge source as if it were that persona. In this application, I have used 3 different personas for synthetic data: A security analyst, an infrastructure engineer, and a product manager. These represent the primary user archetypes for this application, and thus I'd want to tune our testing to meet their specific perspective.

Further than that, I also simulate different breadths of questions; Single-Hop, creating questions from one document at a time, Multi-Hop, focusing on 2 documents, and Global, focusing on 3 or more documents. This mix of question and answer types gives a more comprehensive overview of what you may expect in a production scenario.


#### LLM as Judge
On the output side, manually verifying answers at scale is just as impractical as manually writing test cases. To solve for this, we use an LLM-as-Judge approach — feeding the original question, retrieved documents, and agent output to a separate LLM and having it score the result against a set of defined metrics.

This approach trades a small amount of evaluation cost (each judgment requires an LLM call) for dramatically better coverage than manual review. In practice, LLM judges are surprisingly consistent, with the main caveat being that your Judge must be at least on par or better than the model you're testing.


---

## Evaluation Results (14 test cases)

### FAISS Retriever

| Metric | Overall | Manual (5) | Synthetic (9) |
|---|---|---|---|
| Faithfulness | 0.993 | 1.000 | 0.989 |
| Groundedness | 0.656 | 0.810 | 0.571 |
| Answer Relevancy | 0.882 | 0.984 | 0.826 |
| Context Precision | 0.564 | 0.720 | 0.478 |
| Context Recall | 0.679 | 0.900 | 0.556 |


### LightRAG Retriever

| Metric | Overall | Manual (5) | Synthetic (9) |
|---|---|---|---|
| Faithfulness | 1.000 | 1.000 | 1.000 |
| Groundedness | 0.841 | 0.918 | 0.799 |
| Answer Relevancy | 0.891 | 0.972 | 0.846 |
| Context Precision | 0.854 | 0.990 | 0.778 |
| Context Recall | 0.791 | 0.880 | 0.742 |

(Raw scores available in /evaluation/eval_results_*.json)

### Interpretation

**Faithfulness (FAISS 0.993, LightRAG 1.0)** Both retrievers produce answers that almost never contradict retrieved context. This is the expected behavior from a well-prompted tool-calling agent.

**Groundedness (FAISS 0.656, LightRAG 0.841)** LightRAG's knowledge graph surfaces richer context, giving the LLM more to ground its answers in. FAISS's lower score reflects the LLM augmenting concise NVD descriptions with its own trained knowledge (e.g., adding remediation steps not present in the retrieved text).

**Answer Relevancy (FAISS 0.882, LightRAG 0.891)** Comparable across both retrievers. High for well-specified questions, drops for ambiguous synthetic queries like "What kind of access could this vulnerability give an attacker?" where no CVE is specified and the agent correctly asks for clarification rather than guessing.

**Context Precision (FAISS 0.564, LightRAG 0.854)** The largest delta between the two retrievers. FAISS returns top-k results ranked purely by vector similarity and filtered by a low rerank threshold, so semantically adjacent but irrelevant documents make it through. LightRAG's entity-relationship graph naturally filters to structurally related documents, producing a more focused response.

**Context Recall (FAISS 0.679, LightRAG 0.791)** LightRAG's graph traversal surfaces related entities that pure vector search misses, particularly on multi-hop queries where the retriever needs to connect multiple CVEs. Both retrievers perform well on single-CVE lookups (often 1.0) but diverge on relational questions.


### Replication steps

1. Set env vars in accordance with the project root `.env.example`
2. Run `docker compose up --build -d` to build the containers and run in daemon mode
3. If wanting to test with default CVE list (as provided), run:

    `curl -X POST http://localhost:8001/load/defaults`. 

    To load arbitrary CVEs, run:

    `curl -X POST http://localhost:8001/load \
    -H "Content-Type: application/json" \
    -d '{"cve_ids": ["CVE-2024-3094", "CVE-2023-44487"]}'`

4. At this point, you'll be able to view the knowledge graph and explore the data you just ingested at http://localhost:9621. You'll be able to see the underlying graph structure with the entities, mapped to the custom entity types I created, are connected to each other by their edges.

5. Run `evaluation/generate_testset.ipynb` to generate synthetic test data
6. Run `evaluation/eval.ipynb` to run the evaluation framework against both retrievers

---

## Part 3: Findings

### 1. Where does the pipeline fail and why?

**Failure case: ambiguous queries without a CVE identifier**

The synthetic test "What kind of access could this vulnerability give an attacker?" scores 0.20 answer relevancy, 0.0 context precision, and 0.0 context recall. This question uses "this vulnerability" as a pronoun referring to CVE-2019-5736 used during the synthetic generation process. The FAISS search for the generic phrase "vulnerability access attacker" returns a random set of unrelated CVEs, and the LLM correctly identifies that it cannot determine which vulnerability is being referenced, so it asks for clarification. To prevent overly-ambiguous or malformed synthetic tests, a tighter system prompt and possibly a separate review pass should be implemented.

**Failure case: retrieval returns CVE but context lacks remediation detail**

For "What remediation steps are recommended for Spring4Shell?", groundedness drops to 0.50. The NVD description for CVE-2022-22965 says it affects "JDK 9+ running on Tomcat as WAR" but does not include specific version upgrade instructions. The LLM fills in "upgrade to Spring Framework 5.3.18+ or 5.2.20+" from its trained knowledge. This is factually correct but not grounded in the retrieved context, leading to increased risk of potentially factually incorrect hallucinated content being generated. Potential remediation is utilizing a service like Azure AI content safety, which provides test-time groundedness checks, ensuring the model is generating factually correct, grounded answers during the response stream.

**Failure case: significant score gap between manual and synthetic test cases**

Across both retrievers, synthetic test cases score consistently lower than manual ones. FAISS groundedness drops from 0.810 (manual) to 0.571 (synthetic), context recall from 0.900 to 0.556. LightRAG shows the same pattern — context precision goes from 0.990 to 0.778. While the cumulative drop is explained by malformed synthetic tests, the difference in score drop between FAISS(-0.344) and LightRAG (-0.212) indicates the issue may lie with multi-hop and global questions. Vector based RAG is historically worse at multi-retrieval questions, where multiple disparate, non-semantically related pieces of information are required. With more time, I could parse the synthetic results further and make a concrete determination here.

### 2. What would you change first?

**With another 4 hours:**

- **System Prompt Tuning** Across the application, there are 12 different system prompts being utilized. 1 for the main chat agent, 3 for synthetic test generation, 5 for evaluation/judge, and 3 within LightRAG. Each of these system prompts will have a dramatic effect on model output. The chat agent system prompt specifically is a bit lacking, and as such would be my primary target for enforcing stricter trained knowledge guidelines to lower the risk of hallucinations. Each of the evaluator system prompts should ideally have hand-crafted examples at different score levels for each metric, throwing golden examples to enforce the meaning behind a score. The synthetic test generator falls into the same bucket, a few more hand crafted examples to show good vs bad question/answer pairs. 

- **Parameter Tuning** Similarly, parameter tuning is incredibly important, especially in RAG applications, where you typically have hard thresholds for document retrieval cutoff. Most notable of these is top-k, which limits how many documents get retrieved during any semantic search. Because these are hard thresholds, you must try to find the balance between too much and too little context, each of which can be extremely harmful for cost, latency, and quality. Reranking alleviates some of this, but it comes with its own set of thresholds that need careful tuning.

    I always suggest starting here, as both of these improvements are high-impact and easily verifiable, with clear outcomes, vs some of the more intense and research heavy "4 week" improvements.

**With another 4 weeks:**

- **Data enrichment** CVEs are particularly difficult to semantically search, due to a high degree of semantically meaningless, yet still important, text. (i.e. an embedding model doesn't know what Spring4Shell means, so it doesn't embed well.) One solution, as I've shown, is to use a hybrid method, leveraging a vector database in conjunction with a knowledge graph. Another potential avenue is LLM enriched data during ingest, leveraging test-time scaling at ingest time to include more semantically relevant information within each CVE chunk to improve search.

- **VectorDB interrogation** This more or less goes hand in hand with data enrichment. When attempting to improve data in a RAG pipeline, it can be difficult to see all ramifications at once. Changing parameters or modifying data can lead to better results in some areas, but worse in others. Several techniques can be used, like PCA and K-means clustering, to interrogate and gain insight into a vector database. 

    For example, K-means shows groupings of data in latent (vector) space. In a well-functioning vector database, one would expect clear delineations between groupings of similar content, including sub-groups within groups. If you see little to no delineation between different groups of unrelated data, that shows that your embedding model is struggling to pull out the relevant pieces of semantic information, and it's very likely that your vector database will struggle to be able to pick out the exact right chunk for any given query. (I would bet that that's exactly what's happening with my FAISS instance, despite having a SOTA embedding model, the type of information we're embedding is difficult to capture semantically.)

- **Other forms of hybrid retrieval** While LightRAG is a great example of hybrid retrieval, that being of knowledge graphs and vector databases, other forms of hybrid retrieval exist, most notably keyword search. This would most likely improve my search results, as we could focus in on specific words or phrases that may be difficult for an embedding model to capture on its own.

- **Fine-tuned embedding model.** Train or fine-tune an embedding model on cybersecurity text (CVE descriptions, advisories, CWE definitions) to improve retrieval relevance over the general-purpose `text-embedding-3-large`. This can potentially solve for the issues I've noted above with embedding models and this form of text. Unfortunately, this can be a large lift to achieve correctly, especially when verification is difficult, which is exactly the case with embedding models. 

### 3. Can we use this to automatically triage incoming vulnerability reports and assign severity?

**Short answer: no, not yet**

**What we'd need to build:**

- **Asset context.** Severity depends on context, on what's deployed. A critical RCE in Apache Struts is irrelevant if you don't run Struts. The system needs an asset inventory integration (CMDB, SBOM) to contextualize severity for a specific organization.

- **Fail-Safe Mechanisms** Via prompting and automated oversight, we would want to push the system towards failing safe. In this context, that would mean marking something as critical severity unless a sufficient standard is proven to deem it less than so. We'd most likely build a system in which everything is proven critical unless there is overwhelming evidence against the contrary. Due to the highly sensitive nature of security vulnerabilities, the cost of a false negative would greatly outweigh the cost of a false positive. 

- **Human-in-the-loop workflow.** At least for a while, auto-triage should produce draft assessments for analyst review, not final decisions. Present the LLM's reasoning alongside retrieved evidence so analysts can verify and override. If we see consistently good outcomes, and very few negatives, we would want to have a discussion of the risk vs reward of entirely autonomous workflows. 

**Key risks:**

- **False Negatives** There is always the potential that this system would mark a critical severity issue as non-critical. Depending on the context, that may not be sustainable. A false "Low" on a critical vulnerability could delay patching, or could obfuscate an issue entirely that would have otherwise been found due to trust in the system.

- **Novel vulnerabilities.** Zero-day or newly-published CVEs won't be in the vector store until ingested. The system would need a near-real-time NVD polling pipeline and re-indexing, which adds operational complexity.

- **Adversarial inputs.** If triaging user-submitted vulnerability reports, the input itself could contain prompt injection attempts designed to manipulate the severity assessment. Input sanitization and output validation are essential.

**Recommendation to the PM:** This system would be viable as a triage assistant that works alongside analysts to draft severity assessments with evidence citations. It should not be positioned as an autonomous decision-maker. Implement fail-safe mechanisms, start with a pilot where the system suggests and analysts confirm, measure agreement rate, and expand automation only where agreement exceeds a defined threshold (e.g., 95%).
