from langchain_openai import OpenAI
from openai import OpenAI as OpenAIClient
import os

class MultimodalAgent:
    def __init__(self):
        self.client = OpenAIClient() # Uses OPENAI_API_KEY env var

    def generate_visual_aid(self, prompt: str) -> str:
        """
        Generates an image based on the prompt using DALL-E 3.
        Returns the URL of the generated image.
        """
        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            return response.data[0].url
        except Exception as e:
            return f"Error generating image: {str(e)}"
