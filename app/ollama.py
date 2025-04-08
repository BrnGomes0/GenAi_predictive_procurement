import polars as pl
import requests

# @Deprecated
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
        Reescreva as descrições técnicas abaixo para que fiquem mais claras, legíveis e bem estruturadas em português.

        Regras para cada descrição:
            - Incluir obrigatoriamente o nome do material ou serviço;
            - Escrever uma frase completa, técnica e entendível, evitando o uso de termos em inglês;
            - Finalizar a frase com a categoria do produto ou serviço entre parênteses.

        A seguir, reestruture as descrições contidas na tabela fornecida:
        ```
        {table_str}
        ```
        """
        return prompt
    

    def generate_response(self, prompt: str) -> str:
        pyload = self.generate_pyload(prompt=prompt)
        response = requests.post(self.url_generate, json=pyload)
        return response.json().get("response")