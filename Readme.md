# Using GenAI in Predictive Procuemente - Study Case

## Testing some models (LLaMA3, LLaMA2, Mistral)

### Prompt template
~~~py
f"""
Reescreva a seguinte descrição técnica para torná-la mais clara e legível. A descrição precisa conter o nome do material ou do serviço. Transforme-a em uma frase descritiva bem estruturada em português.
Agora, reformule as seguintes descrições:
```
{table_str}
```
"""
~~~

### Using LLaMA3 (Supplier not Bosch)
---

Return:
~~~md
Aquilo que você está procurando é uma lista de materiais e serviços com descrições técnicas. Aqui estão as descrições reformuladas para torná-las mais claras e legíveis:

MFT3-1852 -992/21-71880055JD-15-06 (2PC) - Conjunto de Fixação de Tornos MFT3 com 2 peças.
SP - CONJUNTO MFT5 0280-20 - Conjunto de Fixação de Tornos MFT5 com 20 peças.
MFT3-4529 - LM2024-Gr31 - Material CA 140/1 para Fixação de Tornos MFT, grade 31.
SP - MFT3-3999 - 08040GN2645 (5pos/8pc) - Conjunto de Fixação de Tornos MFT com 8 peças e 5 posições.
MFT3-272 -718800546L-17 - 2pos/3PC - Conjunto de Fixação de Tornos MFT com 3 peças e 2 posições.
MFT3-4003 - 08A4N3126 gr.12(15pos/18pcs) - Material CA 140/1 para Fixação de Tornos MFT, grade 12.
MP MUELLER NF 460297 MO - Moagem de aço para peças específicas.
MFT3-1114:71880057MB 13 (5 POS/ 5PC) - Conjunto de Fixação de Tornos MFT com 5 peças e 5 posições.
MFT3-1116: 7188.005.72X-9 (3 POS/ 4PC) - Conjunto de Fixação de Tornos MFT com 4 peças e 3 posições.
NIPO-TEC - 47.001.00.730 - BROCA - Broca para uso geral.
Material CA 140/1 - MFT - Material para fixação de tornos MFT.
SP - MFT5-0006-21 - Conjunto de Fixação de Tornos MFT com 21 peças.
MFT3-1983 - 718800952E-20 (1pos/3pc) - Conjunto de Fixação de Tornos MFT com 3 peças e 1 posição.
MFT3-1727 - 718800573C Gr65 15/28 - Material CA 140/1 para fixação de tornos MFT, grade 65.
SP - CONJUNTO MFT5 0155-20 - Conjunto de Fixação de Tornos MFT com 20 peças.
MFT3-1182 - 71880055JD-11 - 3POS/3P - Conjunto de Fixação de Tornos MFT com 3 peças e 3 posições.
MFT3-4528 - 71880062FR1 Gr11 02/03 - Material CA 140/1 para Fixação de Tornos MFT, grade 11.
MFT3-1115 - 718800884W-41 - Conjunto de Fixação de Tornos MFT com 41 peças e 2 posições.
MFT3-1117 - 71880055JD-08 - Conjunto de Fixação de Tornos MFT com 4 peças e 3 posições.
MP MUELLER NF 460297 MO - Moagem de aço para peças específicas.
NIPO-TEC - BROCA NTF495-854 - Broca para uso geral.
KIT ARMATURA DELTA 881 - LEXUS NX 300 TSV - Kit para reparo e manutenção de veículos Lexus.
Essas descrições técnicas foram reformuladas para torná-las mais claras e legíveis, destacando os principais elementos das características dos materiais e serviços.
~~~

### Using LLaMA2 (Supplier not Bosch)
---
Return:
~~~md
Here are the descriptions rewritten in Portuguese:

Material ou serviço | Descrição
MFT3-1852 -992/21-71880055JD-15-06 (2PC) | Sold by Luis Fernando Soldati - ME
SP - CONJUNTO MFT5_0280-20 | Sold by RDB Comércio Indústria e Serviço
MFT3-4529 - LM2024-Gr31 | Sold by GDS Usinação Ltda - ME
SP - MFT3-3999 - 0804GW2645 (5pos/8pc) | Sold by GDS Usinação Ltda - ME
MFT3 272 -718800546L-17 - 2pos/3PC | Sold by Luis Fernando Soldati - ME
MFT3-4003 - 0804EN3126 gr.12(15pos/18pcs) | Sold by Passarella Usinação e Ferragem
MP MUELLER NF 460297 MO | Sold by C.O. Mueller Comércio de Motores e Bicicletas
MFT3-1031 71880055JJ-19 (13POS/33PC) | Sold by Luis Fernando Soldati - ME
NIPO-TEC ORÇ13.570 BROCA 4706100085 CRIN | Sold by Nipo-Tec Indústria e Comércio
MFT3-1906 - 71880059GL_20 3pos/4pçs | Sold by Metalurgica VH Ltda
MFT3_1115 - 71880054VY_13 (3POS/4PC) | Sold by Passarella Usinação e Ferragem
KIT ARAMIDA TEC 811 - Sold by Lexus NX 300 TSV | Carbon Blindados Ltda

Note that some of the descriptions have been shortened or modified to better fit the Portuguese language and grammar rules.
~~~

### Using Mistral (Supplier not Bosch)
---
Return:
~~~md
Material ou serviço | Descrição
MFT3-1852 -992/21-71880055JD-15-06 (2PC) | Luis Fernando Soldati (Mecânica de Precisão)
SP - Conjunto MFT5_0280-20       | RDB Comércio, Indústria e Serviços de Equipamentos
MFT3-4529 - LM2024-Gr31           | GDS Usinagem Ltda (Mecânica de Precisão)
SP - MFT3-3999 - 0804GW2645 (5pos/8pc) | GDS Usinagem Ltda (Mecânica de Precisão)
MFT3 272 -718800546L-17 - 2pos/3PC    | Luis Fernando Soldati (Mecânica de Precisão)
MFT3-4003 - 0804EN3126 gr.12(15pos/18pcs| Passarela Usinagem e Ferramentaria
MP Mueller NF 460297 MO         | C.O. Mueller (Comércio de Motores e Bares)
Tec8311-22 - Conjunto Adeq. à NR12 Arburg 4    | K2 Projetos, Equipamentos e Serviços Ltda
MFT3-1717 -71880040HK / GR. 24 1PC      | Luis Fernando Soldati (Mecânica de Precisão)
MFT3-750 - 718800564B Gr14         | Luis Fernando Soldati (Mecânica de Precisão)
MFT3-4756  718800700U-34 4pc/4pos    | NSB - Instalações Comerciais Ltda
MFT3-879 - 718800203Z-18            | Luis Fernando Soldati (Mecânica de Precisão)
MFT3-850 - 71880055WZ-32 (6 POS/ 24PC).     | Metalurgia VH Ltda
MFT3-1144: 7188.001.4VK-17 (8 POS/ 10 PC| Metalurgia VH Ltda
MFT3-1809 - 718800573C Gr50 03/04    | Luis Fernando Soldati (Mecânica de Precisão)
MP Mueller10983NF448235 MO          | C.O. Mueller (Comércio de Motores e Bares)
MFT3-1031 71880055JJ-19 (13POS/33PC)| Luis Fernando Soldati (Mecânica de Precisão)
NIPO-TEC BROCA FNT495-854         | Ferramentas Nipo-Tec Indústria e Serviços Ltda
MFT3-1906 - 71880059GL_20 3pos/4pcs    | Metalurgia VH Ltda
MFT3_1115 - 71880054VY_13 (3POS/4PC)| Passarela Usinagem e Ferramentaria
Kit Aramida Tec 811 - Lexus NX 300 TSV| Carbon Blindados Ltda
Observação: O termo "MFT3" parece ser uma identificação de código para o material ou serviço, enquanto que"SP" aparece como um indicativo de que a descrição se refere a um conjunto. Além disso, as descriçõescontêm referências às indústrias da mecânica de precisão e ferreiro (ou ferramenta), bem como o nome dasempresas relacionadas com estas indústrias ou os nomes dos produtos fabricados por elas.
~~~