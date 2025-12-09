from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class PromptManager:
    @staticmethod
    def get_technical_support_prompt():
        """Returns the system prompt for the Technical Support Assistant."""
        system_template = """You are an expert Technical Support Engineer and Documentation Assistant.
        Your goal is to help users understand technical manuals and troubleshoot issues.
        
        Use the following pieces of retrieved context to answer the user's question.
        If the answer is not in the context, say that you don't know based on the provided manual.
        Do not make up information.
        
        When answering:
        1. Be precise and step-by-step.
        2. Cite the section or page number if available in the context.
        3. If the user asks for a visual explanation, describe what the diagram should look like in detail.
        
        Context:
        {context}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        return prompt

    @staticmethod
    def get_image_generation_prompt(context: str, user_query: str) -> str:
        """Generates a prompt for DALL-E based on the technical context."""
        # Truncate context to avoid hitting DALL-E's 4000 char limit
        # Reserve ~500 chars for template and user query
        truncated_context = context[:1000]
        
        return f"""Create a clear, technical diagram or illustration explaining the following concept based on this description:
        
        User Query: {user_query}
        
        Technical Context: {truncated_context}
        
        Style: Clean, isometric technical drawing, white background, labeled parts if applicable. High resolution, professional manual style."""

    @staticmethod
    def get_synthetic_data_prompt() -> ChatPromptTemplate:
        """Returns the prompt for generating synthetic troubleshooting scenarios."""
        system_template = """You are a QA Engineer testing a technical manual.
        Based on the provided technical context, generate 3 realistic "User Troubleshooting Scenarios" or "Common Error" questions that a user might ask.
        
        Format the output as a JSON list of objects with 'scenario' and 'expected_answer_summary' keys.
        
        Context:
        {context}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "Generate scenarios."),
        ])
        return prompt
