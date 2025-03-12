import polars as pl
import requests


class Llama3:
    def __init__(self):
        self.url_generate = "http://localhost:11434/api/generate"
        self.model = "llama3"

    
    def generate_pyload(self, prompt: str) -> dict:

        return {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
    

    def create_prompt_message(self, table: pl.DataFrame) -> str:
        table_str = table.write_csv(separator="|") 
        prompt = f"""
        Reescreva a seguinte descrição técnica para torná-la mais clara e legível. A descricao precisa ter o nome do material ou do servico. Transforme isso em uma frase descritiva e bem estruturada em portugues.
        Agora, reformule as seguintes descrições:
        ```
        {table_str}
        ```
        """
        return prompt
    

    def generate_response(self, prompt: str) -> str:

        pyload = self.generate_pyload(prompt=prompt)
        response = requests.post(self.url_generate, json=pyload)
        return response.json().get("response")