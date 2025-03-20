from app.file import Files
from app.ollama import Llama3

files_instance = Files()
model_llama3 = Llama3()

# df_bosch = files_instance.bosch_items()
# df_no_bosch = files_instance.no_bosch_items()

# prompt = model_llama3.create_prompt_message(table=df_no_bosch)
# response = model_llama3.generate_response(prompt=prompt)

# print(response)

table = files_instance.read_csv()
prompt = model_llama3.create_prompt_message(table=table)
response = model_llama3.generate_response(prompt=prompt)

print(response)