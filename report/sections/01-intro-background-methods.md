# TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning

**Course:** Applied Machine Learning (Spring 2026)
**Author:** Sai Kiran Jagini
**Email:** jaginiudaykiran03@gmail.com
**Date:** April 2026

**Reproduced Paper:**
Zheng et al., "TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning," arXiv:2025. GitHub: https://github.com/kiran-jsk/TableRAG-AppML

---

## 1. Introduction

Open-domain question answering increasingly demands systems capable of reasoning over heterogeneous
documents — sources that interweave free-form text paragraphs with structured tables, as found in
Wikipedia articles about sports leagues, corporate financial filings, and scientific datasets.
Existing retrieval-augmented generation (RAG) systems treat all retrieved content as plain text,
systematically discarding the relational structure inherent in tables (row-column relationships,
data types, and aggregation semantics). This has been termed the "structural information loss"
problem, and it manifests in two well-documented failure modes.

The first failure mode arises in flat-text RAG pipelines, which flatten tables into token sequences
and thereby destroy the relational meaning that gives tables their expressive power. When a table
cell's significance depends on its column header or on values in adjacent columns, linearizing the
table into a plain text stream loses exactly this contextual information. The second failure mode
occurs in program-only approaches (NL2SQL, code generation), which achieve high precision on
single-table structured queries but cannot operate over mixed corpora where evidence is distributed
across both prose passages and table cells. These two failure modes represent complementary
deficiencies: the first trades structure for generality, the second trades generality for
structure.

The HybridQA benchmark (Chen et al., 2020), consisting of 70,000 questions requiring multi-hop
reasoning over Wikipedia tables and linked passages, has emerged as the canonical test bed for
evaluating systems that must handle this heterogeneous data problem. Solving heterogeneous document
QA is not merely an academic exercise: any QA system deployed over enterprise knowledge bases,
financial reports, or regulatory filings will inevitably encounter documents that mix structured
and unstructured content, and failure to reason over that structure translates directly into
degraded answer quality in production.

TableRAG [1] introduces a four-stage pipeline specifically designed to address both failure modes
simultaneously. In the offline phase, documents are parsed so that tables are ingested into a
MySQL relational database and textual passages are embedded using BGE (Beijing General Encoder)
models and stored in a vector index. During online inference, the system first decomposes the
input question into structured sub-questions, then performs hierarchical retrieval moving from
the document level down to the table level and finally to the row level. Row-level retrieval is
accomplished via NL2SQL — the language model generates a SQL query that retrieves precisely the
rows relevant to the sub-question, rather than including entire tables in the prompt.

Retrieved text chunks and matched table rows are then combined and passed to a GPT backbone for
answer generation. If the model's confidence is insufficient, the loop iterates up to five times,
each iteration refining the query based on prior retrieved evidence. The key architectural insight
is that NL2SQL is used purely as a retrieval mechanism, not as an answer generation mechanism,
preserving the LLM's ability to synthesize answers from heterogeneous mixed context. In our
reproduction of this system on the HybridQA benchmark using GPT-4o-mini, we achieve EM=48.9% and
F1=58.52%, closely matching the paper's reported 47.8% EM and 57.1% F1 for the same model
configuration.

This report is structured as follows. Section 2 reviews related work across three categories of
prior approaches and situates TableRAG within the existing literature. Section 3 describes the full
TableRAG methodology, including both offline and online phases and the system's architecture.
Section 4 presents our experimental setup and results, including a comparison against
paper-reported baselines. Section 5 summarizes our findings and conclusions. Section 6 describes
individual team contributions. References are provided at the end.

---

## 2. Background and Related Work

The challenge of heterogeneous document question answering sits at the intersection of three active
research threads: traditional text-based RAG, table-only QA via program synthesis, and multi-hop
hybrid reasoning over mixed corpora. Traditional RAG systems excel at retrieving and synthesizing
from prose but are fundamentally ill-equipped to reason over tables, as they lose relational
structure during tokenization. Program-only approaches such as NL2SQL and DATER achieve high
precision on single-table queries but break down when evidence is distributed across both text
passages and tables. Hybrid approaches attempt to bridge the gap but typically require fine-tuning
dataset-specific components or imposing fixed reasoning chains. TableRAG addresses limitations
across all three categories through a unified, LLM-agnostic pipeline that treats SQL execution as
a retrieval primitive rather than an answer-generation primitive.

### 2.1 Table-Aware Retrieval (arXiv:2411.02059)

**Summary.** This work applies table-aware attention mechanisms to the retrieval stage, enabling
models to attend to row and column context rather than treating a table as an unordered bag of
tokens. By encoding structural metadata (column headers, data types, row boundaries) alongside
cell values, the retriever can surface individual relevant rows rather than entire tables,
improving both recall and context efficiency.

**Pros.** The approach preserves a meaningful portion of structural information at retrieval time
and can retrieve at row granularity rather than table granularity, substantially reducing the
amount of irrelevant context delivered to the downstream LLM. On single-table benchmarks, this
leads to measurable improvements over flat-text baselines. The architecture is modular and can be
integrated with various LLM backends without modifying the generation component.

**Cons.** The method is optimized for single-table scenarios and does not generalize well to
multi-hop questions where evidence crosses table-text boundaries. It requires fine-tuned table
encoders trained on table-specific objectives, which introduces significant compute cost and
reduces generalizability to new domains or table schemas. Furthermore, the fine-tuned encoders
embed structural assumptions that may not hold across diverse real-world table formats, and
maintaining separate table-specific embedding models adds operational complexity in production.

**Connection to TableRAG.** TableRAG's hierarchical retrieval design — proceeding from document
retrieval down to table retrieval and finally row retrieval — is directly motivated by the
row-level granularity insight from table-aware retrieval. However, TableRAG achieves fine-grained
row retrieval not through a fine-tuned encoder but through NL2SQL, allowing it to work with
unmodified off-the-shelf LLMs and arbitrary table schemas without any table-specific training.

### 2.2 Structured Prompting for Table QA (arXiv:2504.01346)

**Summary.** This approach constructs structured prompt templates that present table contents to
LLMs in a consistent, parseable format, improving zero-shot table reasoning without any model
fine-tuning. Tables are linearized according to a fixed schema (header row, typed columns,
delimited cells) and inserted directly into the prompt, allowing general-purpose LLMs to reason
over table contents using their in-context learning capabilities.

**Pros.** The method requires no fine-tuning and is compatible with any instruction-following LLM.
The prompt format is human-readable and inspectable, facilitating debugging and interpretability.
For tables that fit within the LLM's context window, structured prompting can achieve competitive
performance against fine-tuned baselines. The approach is also highly portable — it can be
integrated into existing LLM pipelines with minimal engineering overhead.

**Cons.** Because the full table is included in the prompt, the approach fails for tables exceeding
the model's context length — a common occurrence in real-world documents with hundreds of rows.
There is no retrieval step, which means irrelevant rows from large tables bloat the prompt and can
degrade answer quality by introducing noise. Scaling to documents with multiple large tables is
therefore infeasible with this method. Additionally, as LLM context windows have grown, the cost
of including large tables grows proportionally, making the approach expensive at scale.

**Connection to TableRAG.** TableRAG's NL2SQL retrieval step directly solves the context length
problem identified in structured prompting. By executing a SQL query against the MySQL database
and passing only the rows that satisfy the query to the LLM, TableRAG ensures that the prompt
contains only relevant, high-precision table content regardless of the original table size. This
allows TableRAG to operate effectively on very large tables without any modification to the LLM
or prompt format.

### 2.3 Multi-Hop Heterogeneous Reasoning (arXiv:2410.04739)

**Summary.** This work addresses multi-hop QA where each reasoning step may cross a table-text
boundary — for example, a first hop that retrieves a numerical value from a table cell, followed
by a second hop that retrieves a textual passage about an entity referenced by that value. The
approach models these cross-modality hops explicitly using a bridge entity linker that identifies
which values in retrieved table cells should trigger passage lookups, and vice versa.

**Pros.** The method explicitly models the cross-modality reasoning structure and outperforms
single-modality baselines on HybridQA by a substantial margin. It demonstrates that explicitly
tracking bridge entities between table cells and text passages is a principled approach to the
multi-hop heterogeneous problem. The bridge entity linker provides interpretable reasoning chains
that can be audited and debugged.

**Cons.** The bridge entity linker must be fine-tuned on dataset-specific annotated cross-modality
links, making the approach non-trivially expensive to adapt to new domains. The fixed two-hop
structure (table-to-text or text-to-table) does not generalize to arbitrary reasoning chains, and
the fine-tuning requirement makes the method dataset-dependent rather than general-purpose. The
approach also requires a curated annotation of bridge entity pairs, which is costly to obtain for
new domains.

**Connection to TableRAG.** TableRAG's iterative four-step reasoning loop (up to five iterations)
achieves implicit multi-hop reasoning without requiring a fine-tuned bridge entity linker. At each
iteration, the LLM decides autonomously — based on retrieved context from prior iterations —
whether to issue another SQL query, a vector similarity lookup, or generate a final answer. This
dynamic routing allows TableRAG to perform table-to-text and text-to-table reasoning hops in an
LLM-driven, dataset-agnostic manner.

All three related works illuminate the fundamental difficulty of combining structured and
unstructured retrieval in a single framework: table-aware retrieval [1] requires custom encoders,
structured prompting [2] fails on large tables, and explicit multi-hop reasoning [3] requires
dataset-specific fine-tuning. TableRAG's core contribution is a unified, LLM-agnostic pipeline
that handles both modalities within a single iterative reasoning framework, using SQL execution
for structured lookups and vector retrieval for textual evidence, without requiring any
table-specific training or dataset-specific components.

---

## 3. Methods

TableRAG is organized into two phases: an offline phase that constructs persistent data structures
from the document corpus once before any queries are issued, and an online phase that executes per
query, iterating up to five times to produce a confident answer. This two-phase design decouples
expensive data ingestion and index construction from latency-sensitive query processing, enabling
efficient operation at scale once the offline phase completes.

### 3.1 Offline Phase: Table Parsing and Database Construction

The offline phase transforms a raw document collection into two complementary retrieval
structures: a MySQL relational database for structured table lookups and a BGE vector index for
semantic textual retrieval.

**Input.** The input corpus consists of documents containing both text paragraphs and embedded
tables. In our implementation, we use the HybridQA dataset, which provides 10,672 Wikipedia-
derived tables (distributed as dev_excel.zip) alongside their associated textual passages.

**Step 1 — Table extraction.** Tables are extracted from documents and stored as Excel files
(.xlsx). Each Excel file corresponds to a single Wikipedia table, preserving the original column
headers, row structure, and cell values as they appear in the source document. The use of Excel
as an intermediate format preserves data types and column names, which are critical for
generating correct SQL queries in the online phase.

**Step 2 — MySQL ingestion.** Each Excel table is loaded into a MySQL relational database via the
`data_persistent.py` script in the offline ingestion module. Each table becomes a SQL relation;
column names are used as attribute names and data types are inferred from cell content. This
transformation allows the online phase to issue precise SQL queries against table contents without
loading the full table into the LLM's context.

**Step 3 — Text chunk indexing.** The textual paragraphs associated with each document are split
into chunks and encoded using BGE (Beijing General Encoder) models — specifically `bge-m3` for
embedding and `bge-reranker-v2-m3` for reranking. The resulting dense embeddings are stored in a
vector index that supports cosine similarity retrieval. This index serves the document-level
retrieval step in the online phase. BGE models are hosted locally (no external API calls required
for retrieval), which eliminates retrieval latency variance due to network conditions.

### 3.2 Online Phase: Four-Step Iterative Reasoning

The online phase processes each query through a four-step loop, repeating up to a maximum of five
iterations (`MAX_ITER = 5`, as defined in `main.py`). At each iteration, the system accumulates
retrieved evidence and the LLM determines whether to answer or continue retrieving.

**Step 1 — Question decomposition.** The LLM receives the original question and decomposes it
into sub-questions, identifying which parts require structured table lookups and which parts
require textual passage retrieval. This decomposition step serves as the reasoning plan for the
current iteration and directly guides which retrieval paths are activated in subsequent steps.

**Step 2 — Document-level retrieval.** BGE embeddings are used to retrieve the top-K most relevant
documents from the vector index via cosine similarity search. The reranker model then reorders
the candidates by relevance. This step narrows the search space from all 10,000+ tables and
passages to a small candidate set relevant to the current sub-question, ensuring that subsequent
SQL generation targets tables that are actually relevant to the query.

**Step 3 — Table-level and row-level retrieval via NL2SQL.** For each retrieved document's
associated tables, the LLM generates a SQL query targeting the MySQL database. The query is
executed against the relational table, and the matching rows are returned as structured text. If
the SQL query fails (e.g., due to schema mismatch or unsatisfiable conditions), the system falls
back to vector similarity search over the table cells. This hybrid retrieval strategy combines
SQL precision with vector search robustness: SQL is attempted first for maximum precision, with
vector fallback ensuring graceful degradation when SQL cannot be formulated.

**Step 4 — Answer generation.** The LLM receives the original question, all retrieved text chunks,
and all matched table rows assembled into a unified context. It generates a natural-language final
answer. If confidence is assessed as insufficient — or if the decomposition reveals unresolved
sub-questions — the loop continues with a refined query, up to five total iterations. The
iterative design allows the system to resolve multi-hop questions incrementally, using results
from one retrieval step to inform the next.

The key design insight is that NL2SQL serves purely as a retrieval mechanism rather than an
answer-generation mechanism. The LLM generates the final answer in natural language from the
retrieved mixed context, giving it full flexibility to synthesize evidence from tables and text
simultaneously without being constrained to express its answer as a SQL result set.

### 3.3 Architecture Diagram

```mermaid
flowchart TD
    A["Input Documents\n(Text + Tables)"] --> B["Offline Phase"]
    B --> B1["Table Extraction\n(Excel files)"]
    B --> B2["Text Chunking\n+ BGE Embedding"]
    B1 --> B3["MySQL Ingestion\n(SQL relations)"]
    B2 --> B4["Vector Index\n(cosine similarity)"]

    Q["User Question"] --> C["Online Phase"]
    B3 --> C
    B4 --> C

    C --> C1["Step 1: Question\nDecomposition (LLM)"]
    C1 --> C2["Step 2: Document\nRetrieval (BGE)"]
    C2 --> C3["Step 3: Row Retrieval\n(NL2SQL → MySQL)"]
    C3 --> C4["Step 4: Answer\nGeneration (LLM)"]
    C4 --> C5{"Confidence\nSufficient?"}
    C5 -->|"No (iter < 5)"| C1
    C5 -->|"Yes"| ANS["Final Answer"]
```

**Figure 1: TableRAG architecture.** The offline phase (top) constructs the MySQL table database
and BGE vector index once; the online phase (bottom) executes per-query, iterating up to 5 times
until a confident answer is produced.

### 3.4 Discussion

The decision to use NL2SQL as a retrieval primitive rather than an answer-generation primitive is
architecturally sound for two reasons. First, it preserves SQL's precision for structured lookups
— only the rows satisfying the query are passed to the LLM, eliminating irrelevant table content
from the prompt regardless of the original table size. Second, it gives the LLM full freedom to
synthesize answers from heterogeneous context: the LLM is not constrained to express its answer
as a SQL result set but can incorporate evidence from both retrieved rows and text passages in a
single natural-language generation step. This design also means that TableRAG works with any LLM
backbone without table-specific fine-tuning, which is the key architectural difference from
approaches like table-aware retrieval [1] that require custom-trained encoders to achieve
row-level granularity.

It is worth noting that the results presented in this report were produced using GPT-4o-mini as
the LLM backbone. This is the same configuration used in the paper's primary ablation study,
making our EM=48.9% / F1=58.52% results directly comparable to the paper's reported 47.8% / 57.1%
for the same configuration. The small positive delta (+1.1 EM, +1.42 F1) likely reflects minor
differences in prompt formatting and retrieval threshold settings rather than any fundamental
methodological difference.

---
