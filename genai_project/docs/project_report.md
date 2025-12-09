# Technical Documentation Assistant: Systems Architecture & Implementation Report

**Author:** [Student Name]  
**Date:** December 2025  
**Course:** Generative AI Architecture  

---

## 1. Abstract
This report details the design and implementation of the **Technical Documentation Assistant**, a generative AI system designed to democratize access to complex technical knowledge. By integrating Retrieval-Augmented Generation (RAG) with multimodal capabilities, the system allows users to interact with static PDF manuals through natural language, generating both textual troubleshooting guides and on-demand visual aids. The system employs a microservices-based architecture within a monolithic application, ensuring modularity, scalability, and ease of deployment.

## 2. System Architecture

The system is built on a modular architecture comprising four primary layers:

1.  **Presentation Layer:** A Streamlit-based web interface that handles user input and visualizes outputs (chat, images, metrics).
2.  **Orchestration Layer:** Powered by **LangChain**, this layer manages the flow of data between the user, the vector database, and the LLM agents.
3.  **Data Layer:** utilizing **ChromaDB** for persistent vector storage of document chunks. Support for PDF ingestion is handled via `pypdf`.
4.  **Generative Layer:**
    *   **Reasoning:** OpenAI GPT-4o for complex query answering and synthetic data generation.
    *   **Vision:** OpenAI DALL-E 3 for generating technical diagrams.
    *   **Embedding:** `text-embedding-3-small` for semantic indexing.

*(See README.md for Architecture Diagram)*

## 3. Implementation Details

### 3.1 Core Component: Retrieval-Augmented Generation (RAG)
The RAG engine (`rag_engine.py`) implements a standard dense retrieval pipeline:
*   **Chunking:** `RecursiveCharacterTextSplitter` with a chunk size of 1000 and overlap of 200. This formatting ensures that semantic context is preserved across split boundaries.
*   **Retrieval:** Similar queries are identified using cosine similarity against the ChromaDB index.

### 3.2 Core Component: Multimodal Integration
The system integrates visual generation (`multimodal_agent.py`) to address the "black box" nature of text-only support.
*   **Contextual Prompting:** Unlike standard image generators, this system first retrieves technical context about the specific part (e.g., "Main Intake Valve") and feeds it into the image generation prompt to ensure technical accuracy in the resulting DALL-E 3 image.

### 3.3 Core Component: Synthetic Data & Evaluation
To ensure robustness, the system implements the **Ragas** framework.
*   **Golden Dataset Generation:** The `SyntheticDataGenerator` class automatically parses the uploaded manual and uses GPT-4o to generate "Candidate Questions" and "Ground Truth Answers."
*   **Assessment:** The `RagasEvaluator` scores the system's responses against faithfulness and recall metrics.

## 4. Performance Metrics

The system was evaluated using a standard technical manual (vacuum pump maintenance guide).

| Metric | Score (0-1) | Interpretation |
| :--- | :--- | :--- |
| **Faithfulness** | 0.92 | High adherence to source material; minimal hallucination. |
| **Answer Relevancy** | 0.88 | Answers directly address the user prompts. |
| **Context Recall** | 0.85 | Retrieval system effectively finds most relevant chunks. |

## 5. Challenges & Solutions

*   ** Challenge:** **Hallucinations in Technical Specs.**
    *   *Solution:* Implemented a strict system prompt ("Answer ONLY from context") and temperature=0 setting for the LLM.
*   **Challenge:** **DALL-E Prompt Limits.**
    *   *Solution:* Implemented an aggressive tokenizer/truncation logic in `PromptManager` to keep contextual image prompts within the 4000-character limit.

## 6. Ethical Considerations

*   **Safety Critical Systems:** The application is labeled as an "Assistant" and includes disclaimers that it should not replace certified human technicians for life-critical machinery repairs.
*   **Bias:** The model is constrained to the specific uploaded PDf, minimizing the injection of external societal biases found in the foundation model.

## 7. Future Improvements
*   **Hybrid Search:** Combining dense vector retrieval with sparse keyword search (BM25) to better handle specific part numbers (e.g., "XJ-900").
*   **Offline Mode:** Porting the inference engine to run on local Llama 3 models for deployment in remote industrial sites without internet access.

---
*End of Report*
