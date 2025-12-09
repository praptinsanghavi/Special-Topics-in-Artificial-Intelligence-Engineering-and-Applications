"""
Audio Briefing Generator

Creates podcast-style audio summaries using OpenAI's Text-to-Speech API.
Supports both simple narration and conversational two-host dialogue.
"""

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from langchain_core.documents import Document
import os


class AudioBriefingGenerator:
    """
    Generates audio briefings from documents.
    
    Features:
        - Simple TTS narration
        - Conversational podcast (two-host dialogue)
        - Multiple voice options
        - MP3 output
    """
    
    def __init__(self):
        self.client = OpenAI()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    def generate_simple_briefing(self, documents: List[Document], duration_target: str = "2-3 minutes") -> bytes:
        """
        Generate a simple narrated audio briefing.
        
        Args:
            documents: Source documents
            duration_target: Target length (e.g., "2-3 minutes")
            
        Returns:
            Audio bytes (MP3 format)
        """
        # Generate script
        script = self._create_narration_script(documents, duration_target)
        
        # Convert to speech
        audio = self._text_to_speech(script, voice="alloy")
        
        return audio
    
    def generate_podcast_briefing(self, documents: List[Document]) -> bytes:
        """
        Generate a conversational two-host podcast.
        
        Args:
            documents: Source documents
            
        Returns:
            Audio bytes (MP3 format) with dialogue between two hosts
        """
        # Generate dialogue script
        dialogue = self._create_podcast_script(documents)
        
        # Generate audio for each speaker
        audio_segments = []
        for line in dialogue:
            voice = "alloy" if line["speaker"] == "Host 1" else "nova"
            audio = self._text_to_speech(line["text"], voice=voice)
            audio_segments.append(audio)
        
        # Combine segments (simple concatenation)
        combined_audio = b"".join(audio_segments)
        return combined_audio
    
    def _create_narration_script(self, documents: List[Document], duration: str) -> str:
        """Create a narration script from documents."""
        text = "\n\n".join([doc.page_content for doc in documents[:15]])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are creating an audio briefing script. 
            Write a clear, engaging narration that summarizes the key points.
            Target duration: {duration} (approximately 300-400 words).
            
            Style:
            - Conversational but professional
            - Use transitions ("First...", "Additionally...", "Finally...")
            - Avoid jargon unless necessary
            - End with a brief conclusion"""),
            ("user", "Content to summarize:\n{text}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"text": text[:5000]})
    
    def _create_podcast_script(self, documents: List[Document]) -> List[dict]:
        """Create a conversational podcast script with two hosts."""
        text = "\n\n".join([doc.page_content for doc in documents[:15]])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Create a conversational podcast script between two hosts discussing this content.
            
            Format as JSON array:
            [
                {{"speaker": "Host 1", "text": "Welcome to today's briefing..."}},
                {{"speaker": "Host 2", "text": "Thanks! So what are we covering?"}},
                ...
            ]
            
            Guidelines:
            - 8-12 exchanges total
            - Host 1: Introduces topics, asks questions
            - Host 2: Explains concepts, provides insights
            - Natural conversation flow
            - End with key takeaways"""),
            ("user", "Content:\n{text}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"text": text[:5000]})
        
        try:
            import json
            dialogue = json.loads(result)
            return dialogue if isinstance(dialogue, list) else []
        except:
            # Fallback to simple narration
            return [{"speaker": "Host 1", "text": result}]
    
    def _text_to_speech(self, text: str, voice: str = "alloy") -> bytes:
        """
        Convert text to speech using OpenAI TTS.
        
        Args:
            text: Text to convert
            voice: Voice ID (alloy, echo, fable, onyx, nova, shimmer)
            
        Returns:
            Audio bytes (MP3)
        """
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        
        return response.content
    
    def save_audio(self, audio_bytes: bytes, filename: str = "briefing.mp3"):
        """Save audio to file."""
        with open(filename, "wb") as f:
            f.write(audio_bytes)
        return filename
