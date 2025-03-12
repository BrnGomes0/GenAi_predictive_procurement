from file import Files
from gen_ai import ConfigOpenAi 

files_instance = Files()
openai_instance = ConfigOpenAi()


df_bosch = files_instance.bosch_items()
df_no_bosch = files_instance.no_bosch_items()


prompt_bosch = openai_instance.create_prompt_message(df_bosch)
prompt_no_bosch = openai_instance.create_prompt_message(df_no_bosch)

response_bosch = openai_instance.run_prompt(prompt_bosch)
response_no_bosch = openai_instance.run_prompt(prompt_no_bosch)

print("Insights para Bosch:")
print(response_bosch)

print("\nInsights para Não Bosch:")
print(response_no_bosch)
