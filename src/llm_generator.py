from groq import Groq
import os 
from dotenv import load_dotenv

load_dotenv()

class LLMGenerator:
    def __init__(self):
        self.client = Groq(api_key = os.getenv("GROQ_API_KEY"))
    
    def generate(self, query, context):
        prompt = f"""
            You are a supply chain analyst.

            Use ONLY the data below to answer the question.

            Context:
            {context}

            Question:
            {query}

            Give a clear, concise answer.
            """
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
            )
        
        return response.choices[0].message.content
    

