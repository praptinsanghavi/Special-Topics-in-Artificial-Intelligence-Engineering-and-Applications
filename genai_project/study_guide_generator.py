"""
Study Guide Generator Module

Generates comprehensive study guides from uploaded documents using GPT-4o.
Creates hierarchical outlines with overview, key concepts, and detailed sections.
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
from langchain_core.documents import Document
import json


class StudyGuideGenerator:
    """
    Generates structured study guides from documents.
    
    Features:
        - Overview generation (2-3 paragraph summary)
        - Key concepts extraction (bullet points)
        - Hierarchical section breakdown
        - Source-grounded content
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    def generate_study_guide(self, documents: List[Document]) -> Dict:
        """
        Generate a comprehensive study guide from documents.
        
        Args:
            documents: List of document chunks from RAG engine
            
        Returns:
            Dictionary with:
                - overview: High-level summary
                - key_concepts: List of main ideas
                - sections: Detailed breakdown by topic
                - timeline: Chronological events (if applicable)
        """
        # Combine document content
        full_text = "\n\n".join([doc.page_content for doc in documents[:20]])  # Limit for token efficiency
        
        # Generate overview
        overview = self._generate_overview(full_text)
        
        # Extract key concepts
        key_concepts = self._extract_key_concepts(full_text)
        
        # Create detailed sections
        sections = self._create_sections(full_text)
        
        # Extract timeline (if applicable)
        timeline = self._extract_timeline(full_text)
        
        return {
            "overview": overview,
            "key_concepts": key_concepts,
            "sections": sections,
            "timeline": timeline
        }
    
    def _generate_overview(self, text: str) -> str:
        """Generate 2-3 paragraph overview."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert study guide creator. Generate a clear, concise overview 
            of the provided text in 2-3 paragraphs. Focus on the main purpose, key themes, and 
            overall structure of the content."""),
            ("user", "Text:\n{text}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"text": text[:4000]})  # Limit input
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract 8-12 key concepts as bullet points."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract 8-12 key concepts from this text. 
            Format as a JSON array of strings. Each concept should be concise (1-2 sentences).
            Focus on the most important ideas, principles, or facts."""),
            ("user", "Text:\n{text}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"text": text[:4000]})
        
        try:
            concepts = json.loads(result)
            return concepts if isinstance(concepts, list) else []
        except:
            # Fallback: split by newlines
            return [line.strip("- ").strip() for line in result.split("\n") if line.strip()]
    
    def _create_sections(self, text: str) -> List[Dict]:
        """Create hierarchical sections with titles and summaries."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Organize this text into 4-6 logical sections. 
            For each section, provide a title (clear, descriptive heading) and summary (2-3 sentence explanation).
            
            Return as a valid JSON array where each object has 'title' and 'summary' keys."""),
            ("user", "Text:\n{text}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"text": text[:4000]})
        
        try:
            sections = json.loads(result)
            return sections if isinstance(sections, list) else []
        except:
            return []
    
    def _extract_timeline(self, text: str) -> List[Dict]:
        """Extract chronological events if present."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """If this text contains chronological events, dates, or historical progression, 
            extract them as a timeline. Return as a valid JSON array where each object has 'date' and 'event' keys.
            If no timeline is present, return empty array []."""),
            ("user", "Text:\n{text}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"text": text[:4000]})
        
        try:
            timeline = json.loads(result)
            return timeline if isinstance(timeline, list) else []
        except:
            return []
