import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Import our modules
from rag_engine import RAGEngine
from prompt_manager import PromptManager
from multimodal_agent import MultimodalAgent
from synthetic_data import SyntheticDataGenerator
from evaluation import RagasEvaluator
from document_library import DocumentLibrary
import pandas as pd

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(page_title="AI Study Assistant", layout="wide", page_icon="📚")

def main():
    """
    Main application entry point for the AI Study Assistant.
    
    Architecture:
        - Frontend: Streamlit (Python-based UI)
        - Backend: LangChain for RAG orchestration
        - Database: ChromaDB (Vector Store) for document embeddings
        - Evaluation: Ragas framework for systematic testing
        
    Flow:
        1. User configures API Key.
        2. User uploads documents (PDF/TXT) to library.
        3. User selects which documents to use as knowledge base.
        4. System ingests, chunks, and indexes selected documents into ChromaDB.
        5. User interacts via Chat, Visual Generation, or Evaluation.
    """
    st.title("📚 AI Study Assistant")
    st.caption("RAG-powered document Q&A with multi-document support")
    
    # Initialize document library in session state
    if 'doc_library' not in st.session_state:
        st.session_state.doc_library = DocumentLibrary()
    
    # -------------------------------------------------------------------------
    # Sidebar: Configuration & Document Library
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")
        # Securely handle API Key input
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        st.markdown("---")
        st.header("📚 Document Library")
        st.caption("Upload and manage your knowledge base")
        
        # Upload new document
        uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt"], key="doc_uploader")
        
        if st.button("➕ Add to Library") and uploaded_file:
            if not api_key and not os.getenv("OPENAI_API_KEY"):
                st.error("Please provide an OpenAI API Key first.")
            else:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Save file
                        file_ext = uploaded_file.name.split(".")[-1]
                        temp_file = f"library_{uploaded_file.name}"
                        with open(temp_file, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process document
                        temp_rag = RAGEngine()
                        docs = temp_rag.load_documents(temp_file)
                        
                        # Add to library
                        st.session_state.doc_library.add_document(
                            filename=uploaded_file.name,
                            filepath=temp_file,
                            num_chunks=len(docs)
                        )
                        
                        st.success(f"✅ Added {uploaded_file.name} to library!")
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Display document library
        all_docs = st.session_state.doc_library.get_all_documents()
        
        if all_docs:
            st.markdown("### Your Documents")
            st.caption("Select documents to use as AI's knowledge base:")
            
            for doc in all_docs:
                col1, col2, col3 = st.columns([0.1, 0.7, 0.2])
                
                with col1:
                    # Checkbox to toggle active status
                    is_active = st.checkbox(
                        "",
                        value=doc.get("active", False),
                        key=f"active_{doc['id']}",
                        label_visibility="collapsed"
                    )
                    if is_active != doc.get("active", False):
                        st.session_state.doc_library.toggle_document(doc['id'])
                        st.rerun()
                
                with col2:
                    status_icon = "✅" if doc.get("active") else "⭕"
                    st.caption(f"{status_icon} **{doc['filename']}**")
                    st.caption(f"   {doc['num_chunks']} chunks")
                
                with col3:
                    if st.button("🗑️", key=f"del_{doc['id']}", help="Remove"):
                        st.session_state.doc_library.remove_document(doc['id'])
                        st.rerun()
            
            # Process selected documents button
            active_docs = st.session_state.doc_library.get_active_documents()
            if active_docs:
                st.markdown("---")
                if st.button("🔄 Load Selected Documents", type="primary"):
                    with st.spinner("Indexing selected documents..."):
                        try:
                            # Get active file paths
                            active_paths = st.session_state.doc_library.get_active_filepaths()
                            
                            # Initialize RAG with multiple documents
                            st.session_state.rag = RAGEngine()
                            all_docs = st.session_state.rag.load_multiple_documents(active_paths)
                            st.session_state.rag.process_documents(all_docs)
                            
                            st.success(f"✅ Loaded {len(active_docs)} document(s)!")
                            st.session_state.doc_processed = True
                            st.rerun()
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                # Show active sources
                if st.session_state.get('doc_processed', False):
                    st.info(f"🎯 **Active Sources:** {len(active_docs)} document(s)")
        else:
            st.info("No documents yet. Upload your first document above!")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("[Project Repository](https://github.com/your-username/genai-project)")
        st.caption("AI Study Assistant with RAG")

        # History Display
        if 'query_history' in st.session_state and st.session_state.query_history:
            st.markdown("---")
            st.subheader("📝 Recent Queries")
            for item in reversed(st.session_state.query_history[-3:]):
                with st.expander(f"{item['time']} ({item['conf']})", expanded=False):
                    st.caption(f"Q: {item['q'][:40]}...")
                    st.write(item['a'][:100] + "...")

    # -------------------------------------------------------------------------
    # Main Interface: Feature Tabs
    # -------------------------------------------------------------------------
    if "doc_processed" in st.session_state and st.session_state.doc_processed:
        # Organize features into logical tabs for better UX
        tab1, tab2, tab3, tab4 = st.tabs([
            "💬 Ask Questions",
            "🖼️ Generate Diagrams",
            "📝 Practice Quiz",
            "⚙️ System Metrics"
        ])
        
        # --- Tab 1: Ask Questions (RAG Chat) ---
        with tab1:
            st.subheader("Ask a Question")
            # Initialize chat history in session state
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Handle user input
            if prompt := st.chat_input("How do I fix error X?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    try:
                        import time
                        start_time = time.time()
                        
                        # 1. Retrieval with Scores (Confidence)
                        # We use search_with_score to get metrics
                        results = st.session_state.rag.search_with_score(prompt)
                        
                        if not results:
                            response = "I couldn't find any relevant information in the uploaded manual."
                            st.error(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            # Calculate Confidence (Heuristic: 1 - L2 Distance)
                            # Lower distance = better match. If average distance > 1, confidence is low.
                            scores = [score for _, score in results]
                            avg_distance = sum(scores) / len(scores) if scores else 0
                            confidence = max(0, min(1, 1 - (avg_distance * 0.5))) # normalization heuristic
                            
                            # Prepare Context
                            context_docs = [doc for doc, _ in results]
                            context_text = "\n\n".join([d.page_content for d in context_docs])
                            
                            # 2. Augmentation & Generation
                            system_prompt = PromptManager.get_technical_support_prompt()
                            llm = ChatOpenAI(model="gpt-4o", temperature=0)
                            chain = system_prompt | llm | StrOutputParser()
                            
                            response = chain.invoke({"context": context_text, "input": prompt, "chat_history": []})
                            
                            query_time = time.time() - start_time
                            
                            # Display Result
                            st.markdown(response)
                            
                            # --- METRICS DISPLAY ---
                            st.markdown("---")
                            cols = st.columns(3)
                            cols[0].caption(f"⏱️ **Latency:** {query_time:.2f}s")
                            
                            # Confidence Indicator
                            color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                            cols[1].caption(f"🎯 **Confidence:** :{color}[{confidence*100:.1f}%]")
                            
                            # Source Citations
                            with st.expander("📄 View Sources"):
                                # Group sources by document
                                sources_by_doc = {}
                                for i, doc in enumerate(context_docs, 1):
                                    source_file = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
                                    page = doc.metadata.get('page', 'N/A')
                                    
                                    if source_file not in sources_by_doc:
                                        sources_by_doc[source_file] = []
                                    sources_by_doc[source_file].append((page, doc.page_content))
                                
                                # Display grouped by document
                                for source_file, pages in sources_by_doc.items():
                                    st.markdown(f"**📚 {source_file}**")
                                    for page, content in pages:
                                        st.markdown(f"  • Page {page}")
                                        st.caption(f"    {content[:150]}...")
                                    st.markdown("")  # Spacing

                            # Update History
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                            # Session History Log
                            if 'query_history' not in st.session_state:
                                st.session_state.query_history = []
                            st.session_state.query_history.append({
                                "time": time.strftime("%H:%M:%S"),
                                "q": prompt,
                                "a": response,
                                "latency": f"{query_time:.2f}s",
                                "conf": f"{confidence*100:.1f}%"
                            })

                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        st.exception(e)

        # --- Tab 2: Multimodal Generation (Text-to-Image) ---
        with tab2:
            st.subheader("🖼️ Generate Diagrams")
            st.info("Create visual diagrams based on document content.")
            visual_prompt = st.text_input("Concept description (e.g., 'Flowchart of the process described in chapter 3')")
            
            if st.button("Generate Diagram"):
                if not visual_prompt:
                    st.warning("Please enter a description.")
                else:
                    with st.spinner("Generating visual aid..."):
                        # RAG-Enhanced Image Generation:
                        # We retrieve technical details first to ensure the image prompt is accurate.
                        retriever = st.session_state.rag.get_retriever()
                        context_docs = retriever.invoke(visual_prompt)
                        context_text = "\n\n".join([d.page_content for d in context_docs])
                        
                        # Create a specialized DALL-E prompt
                        dalle_prompt = PromptManager.get_image_generation_prompt(context_text, visual_prompt)
                        
                        agent = MultimodalAgent()
                        image_url = agent.generate_visual_aid(dalle_prompt)
                        
                        if image_url.startswith("http"):
                            st.image(image_url, caption=f"Generated Diagram: {visual_prompt}")
                        else:
                            st.error(image_url)

        # --- Tab 3: Synthetic Data Generation (Interactive) ---
        with tab3:
            st.subheader("📝 Practice Quiz")
            st.write("Generate practice questions based on the document.")
            
            if st.button("Generate Quiz"):
                with st.spinner("Analyzing document and generating quiz..."):
                    # Retrieve broad context for scenario generation
                    retriever = st.session_state.rag.get_retriever()
                    context_docs = retriever.invoke("key concepts main ideas important topics")
                    
                    # Use Ragas TestsetGenerator to create realistic scenarios
                    generator = SyntheticDataGenerator()
                    scenarios = generator.generate_test_scenarios(context_docs)
                    
                    if isinstance(scenarios, list):
                        for i, scenario in enumerate(scenarios):
                            with st.expander(f"Question {i+1}: {scenario.get('scenario', 'Unknown')}", expanded=False):
                                st.write(f"**Expected Answer:** {scenario.get('expected_answer_summary', 'N/A')}")
                    else:
                        st.error("Failed to generate quiz questions.")

        # --- Tab 4: Systematic Evaluation (Ragas) ---
        with tab4:
            st.subheader("Systematic Evaluation (Ragas)")
            st.write("Run a comprehensive evaluation of the RAG system using Ragas metrics.")
            
            if st.button("Run Evaluation"):
                with st.spinner("Generating test data and evaluating..."):
                    # Step 1: Generate Synthetic Test Data (Golden Dataset)
                    # We use the raw document chunks stored in the RAGEngine
                    if hasattr(st.session_state.rag, 'documents') and st.session_state.rag.documents:
                        generator = SyntheticDataGenerator()
                        scenarios = generator.generate_test_scenarios(st.session_state.rag.documents, test_size=3)
                    else:
                        st.error("No documents found. Please upload and process a file first.")
                        scenarios = []
                    
                    if isinstance(scenarios, list) and len(scenarios) > 0:
                        questions = [s.get('scenario') for s in scenarios]
                        ground_truths = [s.get('expected_answer_summary') for s in scenarios]
                        
                        # Step 2: Run RAG Pipeline for each generated question
                        answers = []
                        contexts = []
                        retriever = st.session_state.rag.get_retriever()
                        
                        progress_bar = st.progress(0)
                        for i, q in enumerate(questions):
                            # Retrieve context
                            docs = retriever.invoke(q)
                            ctx = [d.page_content for d in docs]
                            contexts.append(ctx)
                            
                            # Generate answer
                            system_prompt = PromptManager.get_technical_support_prompt()
                            llm = ChatOpenAI(model="gpt-4o", temperature=0)
                            chain = system_prompt | llm | StrOutputParser()
                            ans = chain.invoke({"context": "\n".join(ctx), "input": q, "chat_history": []})
                            answers.append(ans)
                            progress_bar.progress((i + 1) / len(questions))
                        
                        # Step 3: Calculate Metrics using Ragas
                        # Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
                        evaluator = RagasEvaluator()
                        results = evaluator.evaluate_results(questions, answers, contexts, ground_truths)
                        
                        st.success("Evaluation Complete!")
                        
                        # Display Results
                        st.write("### Aggregate Scores")
                        st.json(results)
                        
                        st.write("### Detailed Results")
                        df = results.to_pandas()
                        st.dataframe(df)
                        
                    else:
                        st.error("Could not generate test data for evaluation.")

    else:
        st.info("Please upload a technical manual to get started.")

if __name__ == "__main__":
    main()
