from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import polars as pl
import os


class ConfigOpenAi:
    def __init__(self) -> None:
        load_dotenv()
        self.model = "Llama-3-8B-Instruct"
        self.BASE_URL = os.getenv("OPENAI_BASE_URL")
        self.API_KEY = os.getenv("API_OPENAI_KEY")
        self.llm = self.chatOpenAi(self.model)
        
        
    def chatOpenAi(self, model_name: str) -> ChatOpenAI:
        return ChatOpenAI(
            temperature=0.0,
            model=model_name,
            api_key=self.API_KEY,
            base_url=self.BASE_URL,
            streaming=True,
            timeout=100
        )
    
    def create_prompt_message(self, table: pl.DataFrame) -> str:
        table_str = table.write_csv(separator="|") 
        prompt = f"""
        Aqui estão os dados extraídos:

        ```
        {table_str}
        ```

        Por favor, melhores essa descricoes.
        """
        return prompt
    
    def run_prompt(self, prompt_message: str) -> str:
        prompt_template = ChatPromptTemplate.from_template("{input_text}")
        chain = prompt_template | self.llm
        response = chain.invoke({"input_text": prompt_message})
        
        return response
