# AI Study Assistant (GenAI Project)

## 1. Project Overview

**AI Study Assistant** is a multimodal, RAG-powered educational platform designed to transform static technical documents into interactive study aids. It solves the problem of "information overload" by allowing users to converse with their documents, generate study guides, visualize complex concepts through AI-generated diagrams, and test their knowledge with synthetic quizzes.

The core idea is to replace passive reading with active, AI-facilitated engagement. Users upload PDF/TXT manuals, and the system indexes them into a vector database, enabling detailed Q&A, systematic evaluation of knowledge, and multimodal content creation (audio briefings, diagrams).

**High-Level Architecture**: A Python-based monolithic application using **Streamlit** for the frontend user interface, **LangChain** for orchestration, **ChromaDB** for vector storage, and **OpenAI GPT-4o** for reasoning and generation.

---

## 2. Architecture Breakdown

### System Architecture Diagram

```mermaid
graph TD
    User[User / Student]
    UI[Streamlit Frontend]
    Config[Configuration Manager]
    DocLib[Document Library (JSON)]
    
    subgraph "Core Engines"
        RAG[RAG Engine]
        SynData[Synthetic Data Gen]
        VisAgent[Multimodal Agent]
        Audio[Audio Briefing Gen]
        Eval[Ragas Evaluator]
    end
    
    subgraph "Data Persistence"
        Chroma[(ChromaDB Vector Store)]
        Embed[OpenAI Embeddings]
        Files[File System (PDF/TXT)]
    end
    
    subgraph "External Services"
        OpenAI[OpenAI API (GPT-4o, DALL-E 3, TTS)]
    end

    User -->|Interacts| UI
    UI -->|Reads/Writes| DocLib
    UI -->|Queries| RAG
    UI -->|Triggers| SynData
    UI -->|Requests| VisAgent
    UI -->|Requests| Audio
    UI -->|Runs| Eval
    
    RAG -->|Store/Retrieve| Chroma
    RAG -->|Embeds| Embed
    
    SynData -->|Uses| RAG
    SynData -->|Calls| OpenAI
    
    VisAgent -->|Calls| OpenAI
    Audio -->|Calls| OpenAI
    Eval -->|Validates| RAG
    
    Chroma -->|Raw Files| Files
```

### Component Details

#### 1. Frontend (Streamlit)
*   **Purpose**: Renders the UI, manages session state (`st.session_state`), and handles user input (chat, file uploads).
*   **Internal Logic**: Reactive model; re-runs the script on interaction. Uses specific tabs for Chat, Diagrams, Quiz, and Evaluation.
*   **Dependencies**: `streamlit`, `pandas`.
*   **Contracts**: Inputs user actions; outputs rendered HTML/Markdown.
*   **Failure Modes**: Session state loss on browser refresh; Uploader limits (200MB default).

#### 2. RAG Engine (`rag_engine.py`)
*   **Purpose**: Handles document ingestion, chunking, embedding, and retrieval.
*   **Internal Logic**: Uses `PyMuPDFLoader` for PDFs. Splits text using `RecursiveCharacterTextSplitter` (chunk=1000, overlap=200). Stores embeddings in `Chroma`.
*   **Dependencies**: `langchain`, `chromadb`, `openai`.
*   **Contracts**: `load_documents(path) -> List[Doc]`, `retrieve(query) -> List[Doc]`.
*   **Hidden Assumptions**: File paths are absolute. `OPENAI_API_KEY` is valid.

#### 3. Document Library (`document_library.py`)
*   **Purpose**: Meta-store for managing uploaded files and their "Active" state.
*   **Internal Logic**: Persists state to `document_library.json`. Tracks file paths and metadata.
*   **Data Lifecycle**: Create (Upload) -> Update (Toggle Active) -> Delete (Remove File).

#### 4. Synthetic Data & Evaluation (`synthetic_data.py`, `evaluation.py`)
*   **Purpose**: Generates "Golden Dataset" Q&A pairs and evaluates the RAG pipeline.
*   **Internal Logic**: Uses `ragas` library to generate test scenarios and calculate `faithfulness`, `answer_relevancy`.
*   **Limitations**: High latency due to multiple LLM calls per evaluation.

---

## 3. Technology Stack

| Technology | Version | Purpose | Why Chosen |
| :--- | :--- | :--- | :--- |
| **Python** | 3.10+ | Core Language | Vast ecosystem for AI/ML (LangChain, Ragas). |
| **Streamlit** | Latest | Frontend UI | Rapid prototyping, native Python support, requires no HTML/CSS knowledge. |
| **LangChain** | Latest | Orchestration | Standard interface for chaining LLM calls and RAG workflows. |
| **ChromaDB** | Latest | Vector Store | Local, persistent, open-source, easy to integrate with LangChain. |
| **OpenAI API** | GPT-4o | LLM Provider | Best-in-class reasoning for complex study guide generation and RAG. |
| **Ragas** | 0.x | Evaluation | Specialized framework for "Reference-Free" and "Ground-Truth" RAG metrics. |
| **PyMuPDF** | Latest | PDF Parsing | Faster and more accurate text extraction than standard pypdf. |

---

## 4. Installation

### Prerequisites
*   **OS**: Windows 10/11, macOS, or Linux.
*   **Python**: Version 3.10 or higher.
*   **OpenAI API Key**: Required for embeddings and generation.

### Step-by-Step Instructions

1.  **Clone the Repository**
    ```bash
    git clone <repository_url>
    cd genai_project
    ```

2.  **Create a Virtual Environment**
    *   **Windows**:
        ```powershell
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   **Mac/Linux**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Setup**
    Create a `.env` file in the root directory:
    ```bash
    # .env
    OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
    ```

### Troubleshooting
*   **Error**: `ModuleNotFoundError: No module named 'distutils'`
    *   **Fix**: Install setuptools: `pip install setuptools`
*   **Error**: `sqlite3` version issues with ChromaDB.
    *   **Fix**: Ensure Python is >= 3.10. Older usage may require specialized `pysqlite3-binary` install, but standard 3.10+ usually suffices on Windows.

---

## 5. Configuration

Primary configuration is handled via `.env` and `app.py` constants.

| Parameter | Default | Location | Description | Criticality |
| :--- | :--- | :--- | :--- | :--- |
| `OPENAI_API_KEY` | None | `.env` | OpenAI Auth Key. | **Critical**. App will crash or fail to embed if missing. |
| `persist_directory` | `./chroma_db` | `rag_engine.py` | Location of vector DB. | Medium. Change if disk write permissions are restricted. |
| `chunk_size` | `1000` | `rag_engine.py` | Char count per text chunk. | High. Affects context window usage and retrieval granularity. |
| `temperature` | `0` to `0.7` | Multiple files | LLM Creativity. `0` for RAG facts, `0.7` for Creative Writing. | Low. Adjust for "hallucination" control. |

---

## 6. Running the Application

### Development / Production Mode
Since this is a Streamlit app, Dev and Prod are handled similarly, but Prod should run in headless mode.

**Command**:
```bash
streamlit run app.py
```

**Expected Output**:
```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

### Docker Mode (Optional)
If a Dockerfile is present:
```bash
docker build -t ai-study-assistant .
docker run -p 8501:8501 --env-file .env ai-study-assistant
```

---

## 7. API Reference (Internal Capabilities)

*Note: As a standalone Streamlit app, we do not expose HTTP endpoints. Below defines the internal Python Service Interface.*

### `RAGEngine` Service

#### `load_documents(file_path: str)`
*   **Input**: Absolute file path (`.pdf` or `.txt`).
*   **Output**: List of LangChain `Document` objects.
*   **Logic**: Uses `PyMuPDFLoader` for enhanced text extraction.
*   **Error**: Raises `ValueError` for unsupported extensions.

#### `search_with_score(query: str, k: int=4)`
*   **Input**: User query and number of results `k`.
*   **Output**: List of tuples `(Document, score)`.
*   **Metric**: Euclidean Distance (L2) or Cosine Similarity (Chroma Default).
*   **Usage**: Used to calculate "Confidence" scores in the UI.

### `MultimodalAgent` Service

#### `generate_visual_aid(prompt: str)`
*   **Input**: Descriptive prompt for the image.
*   **Output**: URL string (Temporary OpenAI URL) or Error Message string.
*   **Model**: DALL-E 3.
*   **Cost**: Standard DALL-E 3 rates apply per call.

---

## 8. Data Model Documentation

### File System
*   **Raw Files**: Stored in `project_root/` (temporarily) or persisted path.
*   **Format**: PDF / TXT.

### Local "Database" (`document_library.json`)
A simple JSON store for file metadata.

| Field | Type | Nullable | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `id` | string | No | `doc_N` | Unique Identifier. |
| `filename` | string | No | - | Original upload name. |
| `filepath` | string | No | - | Absolute local path. |
| `active` | boolean | No | `True` | Whether RAG includes this file. |
| `num_chunks` | int | Yes | 0 | Count of vector embeddings. |

### Vector Store (ChromaDB)
*   **Collection**: `langchain` (default).
*   **Embedding**: OpenAI `text-embedding-3-small` or `ada-002`.
*   **Metadata**: stored alongside vectors (`source_file`, `page_number`).

---

## 9. Frontend Details (Streamlit)

### Component Hierarchy
1.  **Sidebar**: Configuration (API Key), File Uploader (`st.file_uploader`), Document List (Toggle/Delete).
2.  **Main Area**:
    *   **Tabs Container**:
        *   **Chat Tab**: `st.chat_message` interface.
        *   **Diagrams Tab**: Prompt input -> Image output.
        *   **Quiz Tab**: `st.expander` elements for generated questions.
        *   **Evaluation Tab**: Metric dashboard (`st.dataframe`, `st.metric`).

### State Management
*   `st.session_state.doc_library`: Instance of `DocumentLibrary`.
*   `st.session_state.rag`: Instance of `RAGEngine` (Lazy loaded).
*   `st.session_state.messages`: List of chat history dicts `{'role':..., 'content':...}`.

---

## 10. Testing Guide

### Unit Testing
*   **Tools**: `pytest` (recommended).
*   **Scope**: Test `RAGEngine` loading logic and `PromptManager` string formatting.
*   **Mocking**: Mock `OpenAI` calls to avoid billing during tests.

### Smoke Testing
Run the provided `smoke_test.py` to verify basic environment health:
```bash
python smoke_test.py
```
*Expected Output*: "Imports successful. Basic PyMuPDF check passed."

### Ragas Evaluation (Integration Test)
Run the internal evaluation tab or the script:
```bash
python quick_evaluation.py
```
*   **Metrics**: Faithfulness, Answer Relevancy.
*   **Flakiness**: High. Ragas generation depends on LLM stability.

---

## 11. Deployment Guide

### Deployment Environments
1.  **Local Machine**: Run via `streamlit run`.
2.  **Streamlit Community Cloud**:
    *   Connect GitHub repo.
    *   add `OPENAI_API_KEY` to "Secrets" management.
    *   **Caveat**: `chroma_db` is local; data is lost on container restart unless using persistent cloud storage (e.g., S3) or a Cloud Vector DB (Pinecone).

### CI/CD (Proposed)
*   **GitHub Actions**:
    *   Linting: `flake8`.
    *   Test: `pytest` on non-LLM components.

---

## 12. Security Considerations

1.  **API Keys**: Never commit `.env`. The app allows runtime key entry in Sidebar, which overrides env vars.
    *   *Risk*: Session state allows key to persist in memory during run.
2.  **File Uploads**:
    *   *Risk*: Malicious PDF payloads.
    *   *Mitigation*: `PyMuPDF` is generally robust, but sandboxing is recommended for public deployment.
3.  **Input Injection**:
    *   *Risk*: Prompt Injection in Chat.
    *   *Mitigation*: LangChain `ChatPromptTemplate` separates System vs User context securely.

---

## 13. Performance Notes

*   **Bottlenecks**:
    *   **Ingestion**: Embedding a 100-page PDF can take 30-60s.
    *   **Evaluation**: Generating synthetic test sets (Ragas) is slow (~1-2 mins for 10 questions).
*   **Scaling**:
    *   Current architecture is **Single-Tenant**. Scaling requires moving `ChromaDB` to a server-based instance and handling concurrent Streamlit users via stateless backend design.
*   **Cost**:
    *   Heavy use of Ragas (Evaluation) burns significant GPT-4 tokens. Use cautiously.

---

## 14. Failure Modes & Recovery

| Failure | Cause | Detection | Recovery |
| :--- | :--- | :--- | :--- |
| **IndexError on Chat** | Empty Vector Store | "Index out of bounds" in logs. | Upload a document first. |
| **OpenAI 401 Error** | Invalid API Key | Red Error Toast in UI. | Re-enter key in Sidebar. |
| **Rate Limit 429** | Too many requests | "RateLimitError" log. | Wait 60s; Implement backoff in `rag_engine.py`. |
| **Stale File Handles** | Windows File Lock | PermissionError on Delete. | Restart Streamlit server (`cntrl+c`). |

---

## 15. Roadmap

*   **Short Term**:
    *   Add "Clear Chat History" button.
    *   Implement "Citation Highlighting" (show source text).
*   **Long Term**:
    *   Replace local ChromaDB with Pinecone/Weaviate for cloud persistence.
    *   Add User Authentication (Login/Signup).

---

## 16. FAQ & Troubleshooting

**Q: Why does the app say "No documents found"?**
A: You must upload a file utilizing the Sidebar. Just placing files in the folder is insufficient; they must be processed.

**Q: Why is the "Generate Diagram" button failing?**
A: Ensure your OpenAI Key has DALL-E 3 permissions. Trial keys may not support it.

**Q: How do I reset the database?**
A: Delete the `chroma_db` folder and `document_library.json` file, then restart the app.

---

## 17. License

**MIT License** - See `LICENSE` file for full text.

---

## 18. Credits

**Developed by**: GenAI Project Team (Special Topics in AI Engineering).
**Acknowledgments**: Built with LangChain, Streamlit, and Ragas.
