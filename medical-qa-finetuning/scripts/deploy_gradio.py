#!/usr/bin/env python
"""Deploy Gradio interface"""

from src.inference.pipeline import InferencePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model = AutoModelForCausalLM.from_pretrained("./models/final")
    tokenizer = AutoTokenizer.from_pretrained("./models/final")
    
    pipeline = InferencePipeline(model, tokenizer)
    interface = pipeline.create_interface()
    interface.launch(share=True)

if __name__ == "__main__":
    main()
