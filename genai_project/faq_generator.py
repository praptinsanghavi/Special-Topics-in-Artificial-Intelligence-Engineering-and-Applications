"""
FAQ Generator Module

Automatically generates frequently asked questions from documents.
Leverages the existing Ragas synthetic data infrastructure.
"""

from synthetic_data import SyntheticDataGenerator
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
from langchain_core.documents import Document


class FAQGenerator:
    """
    Generates FAQ (Frequently Asked Questions) from documents.
    
    Features:
        - Auto-generates realistic questions
        - Provides source-grounded answers
        - Categorizes by topic
        - Ranks by importance
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
        self.synthetic_gen = SyntheticDataGenerator()
    
    def generate_faq(self, documents: List[Document], num_questions: int = 10) -> List[Dict]:
        """
        Generate FAQ from documents.
        
        Args:
            documents: Source documents
            num_questions: Number of Q&A pairs to generate
            
        Returns:
            List of {"question": "...", "answer": "...", "category": "..."}
        """
        # Use Ragas to generate questions
        try:
            scenarios = self.synthetic_gen.generate_test_scenarios(documents, test_size=num_questions)
            
            # Convert to FAQ format
            faq_items = []
            for scenario in scenarios:
                if isinstance(scenario, dict):
                    question = scenario.get('scenario', '')
                    # Generate proper answer using RAG
                    answer = self._generate_answer(question, documents)
                    category = self._categorize_question(question)
                    
                    faq_items.append({
                        "question": question,
                        "answer": answer,
                        "category": category
                    })
            
            return faq_items
        
        except Exception as e:
            print(f"FAQ generation error: {e}")
            # Fallback: generate manually
            return self._generate_manual_faq(documents, num_questions)
    
    def _generate_answer(self, question: str, documents: List[Document]) -> str:
        """Generate answer to question using document context."""
        # Get relevant context
        context = "\n\n".join([doc.page_content for doc in documents[:5]])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are answering a question based on the provided context.
            Provide a clear, concise answer (2-4 sentences).
            If the context doesn't contain the answer, say "This information is not covered in the document."
            Always cite specific details from the context."""),
            ("user", "Context:\n{context}\n\nQuestion: {question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context[:3000], "question": question})
    
    def _categorize_question(self, question: str) -> str:
        """Categorize question into topic area."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Categorize this question into ONE of these categories:
            - Overview
            - Key Concepts
            - Details
            - Application
            - Comparison
            
            Return only the category name."""),
            ("user", question)
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        category = chain.invoke({})
        return category.strip()
    
    def _generate_manual_faq(self, documents: List[Document], num: int) -> List[Dict]:
        """Fallback: manually generate FAQ without Ragas."""
        text = "\n\n".join([doc.page_content for doc in documents[:10]])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Generate {num} frequently asked questions about this content.
            Return as JSON array:
            [
                {{"question": "...", "answer": "...", "category": "Overview|Key Concepts|Details"}},
                ...
            ]
            
            Make questions realistic and answers concise (2-3 sentences)."""),
            ("user", "Content:\n{text}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"text": text[:4000]})
        
        try:
            import json
            faq = json.loads(result)
            return faq if isinstance(faq, list) else []
        except:
            return []
