class InferencePipeline:
    """
    Production-ready inference pipeline with Gradio interface
    Implements efficient inference with safety validation
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.safety_validator = MedicalSafetyValidator()
        
    def create_interface(self):
        """
        Create interactive Gradio interface for model inference
        
        Returns:
            Gradio Interface object
        """
        def generate_response(question: str, temperature: float = 0.7, 
                             max_length: int = 100, use_safety: bool = True) -> str:
            """
            Generate medical response with optional safety validation
            
            Args:
                question: Medical question
                temperature: Generation temperature
                max_length: Maximum response length
                use_safety: Whether to apply safety validation
                
            Returns:
                Generated response with safety warnings if applicable
            """
            # Format input
            input_text = f"### Instruction:\nMedical Question: {question}\n\n### Response:\n"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=256
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response
            if "### Response:" in full_response:
                response = full_response.split("### Response:")[1].strip()
            else:
                response = full_response
            
            # Apply safety validation if enabled
            if use_safety:
                response, safety_report = self.safety_validator.validate_medical_response(response)
                
                # Add safety summary
                if any(v.get('risk_level', 0) > 0 for v in safety_report.values()):
                    response += "\n\n---\n**Safety Analysis:**\n"
                    for check, result in safety_report.items():
                        if result.get('risk_level', 0) > 0:
                            response += f"• {check.replace('_', ' ').title()}: Risk Level {result['risk_level']}/5\n"
            
            return response
        
        # Create Gradio interface
        interface = gr.Interface(
            fn=generate_response,
            inputs=[
                gr.Textbox(
                    label="Medical Question",
                    placeholder="Enter your medical question here...",
                    lines=3
                ),
                gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature (lower = more focused)"
                ),
                gr.Slider(
                    minimum=50,
                    maximum=200,
                    value=100,
                    step=10,
                    label="Max Response Length"
                ),
                gr.Checkbox(
                    label="Enable Safety Validation",
                    value=True
                )
            ],
            outputs=gr.Textbox(
                label="Medical Response",
                lines=10
            ),
            title="🏥 Advanced Medical QA System",
            description="""
            Fine-tuned medical question-answering system with comprehensive safety validation.
            Features drug interaction checking, emergency detection, and clinical guideline compliance.
            """,
            examples=[
                ["What are the symptoms of diabetes?", 0.7, 100, True],
                ["How is hypertension treated?", 0.7, 150, True],
                ["What are the drug interactions between warfarin and aspirin?", 0.5, 100, True],
                ["Explain the pathophysiology of heart failure", 0.8, 200, True]
            ],
            theme="default"
        )
        
        return interface
