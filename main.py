from app.file import Files
from app.ollama import Llama3

files_instance = Files()
model_llama3 = Llama3()

table = files_instance.read_csv()
prompt = model_llama3.create_prompt_message(table=table)
response = model_llama3.generate_response(prompt=prompt)
