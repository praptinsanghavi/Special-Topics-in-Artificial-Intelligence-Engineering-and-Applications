# Video Demonstration Script: Technical Documentation Assistant
**Target Length:** 10 Minutes

---

## 0:00 - 1:30 | Introduction & Hook
*   **Visual:** Show yourself (webcam) or a title slide.
*   **Audio:** "Hi, I'm [Name]. For my GenAI Systems Architecture project, I built the 'Technical Documentation Assistant'. Accessing complex technical manuals in the field is difficult. My system solves this by turning static PDFs into intelligent, multimodal assistants."
*   **Visual:** Briefly flash the "Landing Page" (`index.html`) you created.

## 1:30 - 3:00 | System Architecture
*   **Visual:** Show the Mermaid Architecture diagram in the README or Report.
*   **Audio:** "The system is built on a microservices-style architecture orchestrated by LangChain.
    *   **Data Layer:** We use ChromaDB for vector storage.
    *   **RAG Engine:** GPT-4o processes retrieved chunks.
    *   **Vision:** A dedicated agent connects to DALL-E 3 for generating diagrams."

## 3:00 - 6:00 | Live Demo (The Core)
*   **Visual:** Switch to the Streamlit App.
*   **Action:** Upload a PDF (e.g., a sample manual).
*   **Audio:** "Creating a knowledge base is as simple as uploading a PDF. The system heavily chunks and indexes the data in real-time."
*   **Action:** Go to "Chat & Support". Ask: "How do I perform monthly maintenance?"
*   **Audio:** "Notice the retrieval speed. The answer is generated strictly from the context, minimizing hallucinations. It even cites the steps."
*   **Action:** Go to "Visual Aids". Type: "Exploded view of the primary valve."
*   **Audio:** "This is where it gets interesting. The system retrieves the technical description of the valve from the text, then instructs DALL-E 3 to generate an accurate isometric diagram."

## 6:00 - 8:00 | Evaluation & Testing
*   **Visual:** Switch to "Systematic Evaluation" tab.
*   **Action:** Click "Run Evaluation".
*   **Audio:** "A generative system is only as good as its reliability. I've integrated the **Ragas** framework. It generates synthetic test questions—essentially 'quizzing' itself—to calculate Faithfulness and Recall scores. This ensures the system is enterprise-ready."

## 8:00 - 9:00 | Code Highlight
*   **Visual:** Switch to VS Code. Show `rag_engine.py` or `prompt_manager.py`.
*   **Audio:** "I want to highlight the `PromptManager` class. I used a strict Persona pattern here to ensure the AI behaves like a Senior Support Engineer, prioritizing safety warnings above all else."

## 9:00 - 10:00 | Conclusion
*   **Visual:** Back to Landing Page or slide.
*   **Audio:** "In conclusion, this project demonstrates a full-stack implementation of RAG, Multimodal AI, and Automated Evaluation. It addresses the real-world need for accessible technical knowledge. Thank you."
