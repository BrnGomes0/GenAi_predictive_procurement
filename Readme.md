# Using GenAI in Predictive Procuremente - Study Case

## Testing some models (LLaMA3, LLaMA2 and Mistral) 💻

### Prompt template 📃
~~~py
f"""
Reescreva a seguinte descrição técnica para torná-la mais clara e legível. A descrição precisa conter o nome do material ou do serviço. Transforme-a em uma frase descritiva bem estruturada em português.
Agora, reformule as seguintes descrições:
```
{table_str}
```
"""
~~~


### Using LLaMA3 (Supplier not Bosch) ✅ 
---

#### Example One:
Return:
~~~md
Aquilo que você está procurando é uma lista de materiais e serviços com descrições técnicas. Aqui estão as descrições reformuladas para torná-las mais claras e legíveis:

Material ou serviço | Descrição

MFT3-1852 -992/21-71880055JD-15-06 (2PC) | Conjunto de Fixação de Tornos MFT3 com 2 peças.
SP - CONJUNTO MFT5 0280-20 | Conjunto de Fixação de Tornos MFT5 com 20 peças.
MFT3-4529 - LM2024-Gr31 | Material CA 140/1 para Fixação de Tornos MFT, grade 31.
SP - MFT3-3999 - 08040GN2645 (5pos/8pc) | Conjunto de Fixação de Tornos MFT com 8 peças e 5 posições.
MFT3-272 -718800546L-17 - 2pos/3PC | Conjunto de Fixação de Tornos MFT com 3 peças e 2 posições.
MFT3-4003 - 08A4N3126 gr.12(15pos/18pcs) | Material CA 140/1 para Fixação de Tornos MFT, grade 12.
MP MUELLER NF 460297 MO | Moagem de aço para peças específicas.
MFT3-1114:71880057MB 13 (5 POS/ 5PC) | Conjunto de Fixação de Tornos MFT com 5 peças e 5 posições.
MFT3-1116: 7188.005.72X-9 (3 POS/ 4PC) | Conjunto de Fixação de Tornos MFT com 4 peças e 3 posições.
NIPO-TEC - 47.001.00.730 - BROCA | Broca para uso geral.
Material CA 140/1 - MFT | Material para fixação de tornos MFT.
SP - MFT5-0006-21 | Conjunto de Fixação de Tornos MFT com 21 peças.
MFT3-1983 - 718800952E-20 (1pos/3pc) | Conjunto de Fixação de Tornos MFT com 3 peças e 1 posição.
MFT3-1727 - 718800573C Gr65 15/28 | Material CA 140/1 para fixação de tornos MFT, grade 65.
SP - CONJUNTO MFT5 0155-20 | Conjunto de Fixação de Tornos MFT com 20 peças.
MFT3-1182 - 71880055JD-11 - 3POS/3P | Conjunto de Fixação de Tornos MFT com 3 peças e 3 posições.
MFT3-4528 - 71880062FR1 Gr11 02/03 | Material CA 140/1 para Fixação de Tornos MFT, grade 11.
MFT3-1115 - 718800884W-41 | Conjunto de Fixação de Tornos MFT com 41 peças e 2 posições.
MFT3-1117 - 71880055JD-08 | Conjunto de Fixação de Tornos MFT com 4 peças e 3 posições.
MP MUELLER NF 460297 MO | Moagem de aço para peças específicas.
NIPO-TEC - BROCA NTF495-854 | Broca para uso geral.
KIT ARMATURA DELTA 881 | LEXUS NX 300 TSV - Kit para reparo e manutenção de veículos Lexus.
Essas descrições técnicas foram reformuladas para torná-las mais claras e legíveis, destacando os principais elementos das características dos materiais e serviços.
~~~

#### Example Two:
Return:
~~~md
Aqui estão as descrições reformuladas em português:

**Material e Serviço**

* Alicate SK42 BZI Ø40,63: É um alicate de alta qualidade com um diâmetro de 40,63 milímetros.
* Chapa N610014970AE: Trata-se de uma chapa resistente e duradoura com características específicas.
* Cesto em Aço Inoxidável N210088015AA: Um cesto robusto feito de aço inoxidável para uso em ambientes industriais.
* Tubo PCTFE N510069068AA: Um tubo flexível e resistente feito de poliéster fluorado, ideal para aplicações específicas.
* Guia N610048963AB: Uma guia precisa e confiável com características técnicas específicas.
* Porta-Bico N610009409AA: Uma porta-bico segura e prática para uso em operações de manuseio de materiais.
* Cilindro Elástico KXF0DKAAA00: Um cilindro elástico resistente e flexível com características específicas.
* Braço de Alicate KXFB00S6A03: Um braço de alicate robusto e confiável para uso em operações de manuseio de materiais.
* Cilindro de Aço N210007425AA: Um cilindro de aço resistente e duradouro com características específicas.
* Colar de Presa 969.483 MGN6: Um colar de presa preciso e confiável para uso em aplicações industriais.
* Colar de Presa 969.488 MGN20: Um colar de presa robusto e resistente com características específicas.
* Nut de Segurança 969.873 MPS10-0708: Uma nut de segurança segura e confiável para uso em aplicações industriais.    
* Nut de Segurança 969.884 MPS13-1112: Uma nut de segurança robusta e resistente com características específicas.     
* Nut de Segurança 969.909 MPS20-1314: Uma nut de segurança precisa e confiável para uso em aplicações industriais.   
* Nut de Segurança 969.915 MPS20-1920: Uma nut de segurança segura e confiável para uso em aplicações industriais.    
* Chave de Colar 805.256: Uma chave de colar precisa e confiável para uso em operações de manuseio de materiais.      
* Cola 961.522 NBC6-6AA: Uma cola resistente e flexível com características específicas.
* Cola 961.557 NBC10-5AA: Uma cola robusta e resistente com características específicas.
* Cola 961.561 NBC10-7AA: Uma cola precisa e confiável para uso em operações de manuseio de materiais.

Lembre-se de que essas descrições devem ser feitas de forma clara e objetiva, evitando linguagem técnica exagerada ou ambíguas.
~~~


#### Example Three:
Prompt:
~~~py
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
~~~

Return:
~~~md
Aqui estão as descrições técnicas reescritas para serem mais claras, legíveis e bem estruturadas em português:

1. Clamp SK42 BZI Ø40,63 (Materiais de Conexão)
O clamp SK42 BZI Ø40,63 é um dispositivo utilizado para fixar materiais com precisão, oferecendo uma compressão segura e estabilizadora.

2. PLATE N610014970AE (Peças de Metal)
A placa N610014970AE é uma peça metalúrgica utilizada em aplicativos industriais, caracterizada por sua resistência e durabilidade.

3. PALLET,STAINLESS STEEL N210088015AA (Equipamentos de Logística)
O pallet N210088015AA é um equipamento de logística feito de aço inoxidável, projetado para transportar materiais pesados e delicados com segurança.

4. TUBE,PCTFE N510069068AA (Componentes Plásticos)
A tubulação PCTFE N510069068AA é um componente plástico utilizado em aplicativos industriais, conhecido por sua resistência à temperatura e ao impacto.

5. GUIDE N610048963AB (Peças de Guiamento)
O guia N610048963AB é uma peça utilizada para guiagem precisa de materiais, oferecendo estabilidade e previsibilidade em aplicativos industriais.

6. NOZZLE HOLDER N610009409AA (Equipamentos de Processamento)
O holder de bico N610009409AA é um equipamento utilizado em processos de fabricação, projetado para manter a precisão e segurança durante a manipulação de materiais.

7. SPRING KXF0DKAAA00 (Componentes Elásticos)
A mola KXF0DKAAA00 é um componente elástico utilizado em aplicativos industriais, conhecido por sua flexibilidade e resistência à compressão.

8. CLAMP ARM KXFB00S6A03 (Materiais de Conexão)
O braço de clamp KXFB00S6A03 é um dispositivo utilizado para fixar materiais com precisão, oferecendo uma compressão segura e estabilizadora.

9. SPRING,STEEL N210007425AA (Componentes Elásticos)
A mola de aço N210007425AA é um componente elástico utilizado em aplicativos industriais, conhecido por sua resistência à compressão e flexibilidade.

10. Collet Nut 969.483 MGN6 (Componentes de Fixação)
A coluna 969.483 MGN6 é uma peça utilizada para fixar materiais com precisão, oferecendo estabilidade e segurança em aplicativos industriais.

11. Collet Nut 969.488 MGN20 (Componentes de Fixação)
A coluna 969.488 MGN20 é uma peça utilizada para fixar materiais com precisão, oferecendo estabilidade e segurança em aplicativos industriais.

12. Sealed Nut 969.873 MPS10-0708 (Componentes de Fixação)
A porca selada 969.873 MPS10-0708 é uma peça utilizada para fixar materiais com precisão, oferecendo estabilidade e segurança em aplicativos industriais.

13. Sealed Nut 969.884 MPS13-1112 (Componentes de Fixação)
A porca selada 969.884 MPS13-1112 é uma peça utilizada para fixar materiais com precisão, oferecendo estabilidade e segurança em aplicativos industriais.

14. Sealed Nut 969.909 MPS20-1314 (Componentes de Fixação)
A porca selada 969.909 MPS20-1314 é uma peça utilizada para fixar materiais com precisão, oferecendo estabilidade e segurança em aplicativos industriais.

15. Sealed Nut 969.915 MPS20-1920 (Componentes de Fixação)
A porca selada 969.915 MPS20-1920 é uma peça utilizada para fixar materiais com precisão, oferecendo estabilidade e segurança em aplicativos industriais.

16. Collet Chuck 805.256 (Componentes de Fixação)
O aperto 805.256 é um componente utilizado para fixar materiais com precisão, oferecendo estabilidade e segurança em aplicativos industriais.

17. Collet 961.522 NBC6-6AA (Componentes de Fixação)
A coluna 961.522 NBC6-6AA é uma peça utilizada para fixar materiais com precisão, oferecendo estabilidade e segurança em aplicativos industriais.

18. Collet 961.557 NBC10-5AA (Componentes de Fixação)
A coluna 961.557 NBC10-5AA é uma peça utilizada para fixar materiais com precisão, oferecendo estabilidade e segurança em aplicativos industriais.

19. Collet 961.561 NBC10-7AA (Componentes de Fixação)
A coluna 961.561 NBC10-7AA é uma peça utilizada para fixar materiais com precisão, oferecendo estabilidade e segurança em aplicativos industriais.
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

### Doing the clustes with Llama3
Prompt:
~~~md
prompt = f"""
            Você receberá dados contendo:
                item_d: descrição do item ou serviço.

            Sua tarefa é:
                Ler e interpretar o conteúdo de item_d.
                Criar apenas uma categoria que agrupe o item/serviço, baseada na função principal, finalidade ou tipo de atividade descrita.
                A categoria deve ser concisa, específica e usar termos técnicos e comerciais adequados.
                A categoria deve ser criada em no máximo 3 palavras.

            Regras:
                Não explique, não justifique e não adicione comentários.
                Não reescreva a descrição.
                Não altere o conteúdo original.
                Não adicione observações.

            Apenas responda no seguinte formato:
                item_d: [descrição original], categoria: [categoria gerada]
            Exemplo:
                item_d: Instalação de rede elétrica residencial, categoria: Instalações Elétricas
                item_d: Serviço de consultoria jurídica empresarial, categoria: Consultoria Jurídica
                item_d: Venda de computadores e periféricos, categoria: Equipamentos de Informática

            Importante:
                A resposta deve ser uma lista de descrições e categorias no formato acima, uma por linha.
                Se a descrição estiver vaga, use a categoria mais próxima possível, mas nunca invente funções.
        ```
        {table_str}
        ```
        """
~~~~

Return One:
~~~md
Here is the output:

4103064514: G.S.O. VEDAcoES INDS. LTDA, categoria: Serviços Gerais
4103033547: TEF3-P2H400290, categoria: Equipamentos Industriais
4103094843: N°. 73/24- EKPT - Mão de obra mecânica, categoria: Serviços Mecânicos
4103076951: Calibração Zoller SN 405 - COT. 379/22, categoria: Serviços de Calibração
4103095700: CABO AK9/20M, categoria: Componentes Elétricos
4103019267: SP - FILTRO PM-CG-E 495X495X50MM, categoria: Filtros Industriais
4103027968: MODULO TAMPA P/ CONECTOR, categoria: Componentes Eletrônicos
4103021438: 450.20.00 - Válvula Dosadora 0.003-0.2ml, categoria: Componentes Químicos
4103108980: CAFE SOLUVEL, categoria: Produtos de Consumo
4103108533: TEF3|SERVIÇO SERRALHERIA SERRANORTE, categoria: Serviços de Serralheria
~~~~

Return Two:
~~~md
Here are the categorized items:

4103064514|3|SP - ANEL VITON 2137|0|G.S.O.  VEDAcoES INDS. LTDA, categoria: Materiais
4103033547|14|TEF3-P2H400290|0|ASSAHI MAQUINAS E EQUIPAMENTOS LTDA, categoria: Maquinaria
4103094843|1|N°. 73/24- EKPT - Mão de obra mecânica|1|LUIS FERNANDO SOLDATI - ME, categoria: Serviços
4103076951|1|Calibração Zoller SN 405 - COT. 379/22|1|ZOLLER DO BRASIL SOLUCOES PARA A, categoria: Consultoria        
4103095700|4|CABO AK9/20M|0|Mettler - Toledo Industria, categoria: Materiais
4103019267|1|SP - FILTRO PM-CG-E 495X495X50MM|0|LINTER FILTROS INDUSTRIAIS LTDA, categoria: Materiais
4103027968|6|MODULO TAMPA P/ CONECTOR|0|DIGITALDOOR COMERCIO DE EQUIPAMENTO, categoria: Equipamentos
4103021438|1|450.20.00 - Válvula Dosadora 0.003-0.2ml|0|B SORDI COMERCIO DE MAQUINAS E, categoria: Maquinaria
4103108980|2|CAFE SOLUVEL|0|PRACAFE COMERCIAL LTDA  EPP, categoria: Produtos
4103108533|1|"TEF3|SERVIÇO SERRALHERIA SERRANORTE"|1|Dimas Reger de Godoi Oliveira - ME, categoria: Serviços
~~~~

Return Three:
~~~md
Here is the list of descriptions and categories:

4103064514|SP - ANEL VITON 2137, categoria: Anéis
4103033547|TEF3-P2H400290, categoria: Componentes Eletrônicos
4103094843|Mão de obra mecânica, categoria: Serviços Manuais
4103076951|Calibração Zoller SN 405 - COT. 379/22, categoria: Calibrações
4103095700|CABO AK9/20M, categoria: Cabos e Fios
4103019267|FILTRO PM-CG-E 495X495X50MM, categoria: Filtragem
4103027968|MODULO TAMPA P/ CONECTOR, categoria: Módulos Eletrostáticos
4103021438|Válvula Dosadora 0.003-0.2ml, categoria: Componentes Químicos
4103108980|CAFE SOLUVEL, categoria: Produtos Alimentares
4103108533|TEF3|SERVIÇO SERRALHERIA SERRANORTE, categoria: Serviços de Serralheria
~~~~

Return Four:
~~~md
Here is the response:

4103064514|item_d: SP - ANEL VITON 2137, categoria: Engenharia Mecânica
4103033547|item_d: TEF3-P2H400290, categoria: Manutenção Industrial
4103094843|item_d: N°. 73/24- EKPT - Mão de obra mecânica, categoria: Serviços Mecânicos
4103076951|item_d: Calibração Zoller SN 405 - COT. 379/22, categoria: Calibração e Testagem
4103095700|item_d: CABO AK9/20M, categoria: Cabos Elétricos
4103019267|item_d: SP - FILTRO PM-CG-E 495X495X50MM, categoria: Filtração Industrial
4103027968|item_d: MODULO TAMPA P/ CONECTOR, categoria: Conectores Eletrônicos
4103021438|item_d: 450.20.00 - Válvula Dosadora 0.003-0.2ml, categoria: Componentes Químicos
4103108980|item_d: CAFE SOLUVEL, categoria: Produtos Químicos
4103108533|item_d: TEF3 - SERVIÇO SERRALHERIA SERRANORTE, category: Serviços de Serralheria
~~~~

## Getting Started 📍

### As a dev 💻

#### First Step - Clone the repo:
~~~git
git clone --branch main https://github.com/BrnGomes0/GenAi_predictive_procurement.git
~~~

#### Second Step - Create a container based in docker-compose.yml:
~~~docker
docker compose up
~~~

#### Fifth Step - Run the command python for start the project
~~~py
python .\run.py
~~~

### As a Client 😀

*URL*: `http://localhost:5001/ollama`

**ENDPOINTS API:**

🌐 TEST THE APPLICATION:

- Endpoint: `/test`
- Method: `GET`

✅ Expected Response:
~~~json
{
    "message": "The application OLLAMA it's working..."
}
~~~
Status Code: `200`

🌐 SEND PROMPT:

- Endpoint: `/create`
- Method: `POST`

Required Attributes:
~~~json
{
    "prompt": "string"
}
~~~

✅ Expected Response:
~~~json
{
    "message": "string"
}
~~~
Status Code: `200`
