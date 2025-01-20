# -*- coding: utf-8 -*-
import pandas as pd
import ollama
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
import os
import time

# Define a seed para o sorteio
seed = 144
quantidade_artigos = 1500

random.seed(seed)

# Intervalo de números e lista de proibidos
total_numbers = 21163

proibidos = [531, 15812, 7943, 16172, 6142, 4681, 2872, 1243, 6473, 9371, 111, 3827, 8472, 678, 44, 900, 871, 442, 369, 10, 
             13816, 15455, 16517, 2911, 2302, 19154, 960, 14137, 11724, 2053]

# Conjunto de números válidos
valid_numbers = set(range(1, total_numbers + 1)) - set(proibidos)

# Converte para lista para o random.sample
artigos = random.sample(sorted(valid_numbers), quantidade_artigos)

# Exibindo os números sorteados
print()
print("Artigos Sorteados:", artigos)

modelo = 'mistral-large:123b'
dfs = []

print(f'Modelo utilizado: {modelo}')
print()

for artigo in artigos:
    df = pd.read_csv(f'../compilador/sentencas/ABS{artigo}.csv') 
    df["artigo_id"] = artigo  # Adiciona uma coluna para o ID do artigo
    dfs.append(df)
    print("Titulo:", df["Titulo"][0])

c = 0
abstracts = []
for i in dfs:
    a = ''
    for j in i["Sentencas"]:
        c = c+1
        a = a + j + " "
    abstracts.append(a)

print("Quantidade de sentencas (maximo ~1600): ", c)

def modelo_extrator(sentenca, temp, tp, tk):
    exemplos = [
        {"role": "system", "content": (
            "You are a material extractor for scientific abstracts about Solide Oxide Fuel Cells (SOFC). "
            "Your task is to identify and extract all material names specifically used in solid oxide fuel cells that are either cathode, electrolyte, or anode materials. "
            "Respond only with a Python list format, e.g., ['Material1', 'Material2']. "
            "If no specific materials are explicitly labeled as cathode, electrolyte, or anode in the sentence, respond only with 'None'. Avoid giving any explanations or additional information."
            "\n\n"
            "Examples:\n"
            "User: Extract the fuel cell materials in this sentence: "
            "'BaIn0.3Ti0.7O2.85 (BIT07) can be used as electrolyte material with LSCF as cathode and BIT07-Ni cermet as anode.'\n"
            "Assistant: ['BaIn0.3Ti0.7O2.85', 'LSCF', 'BIT07-Ni cermet']\n\n"
            "User: Extract the fuel cell materials in this sentence: "
            "'Button cells have been designed based on those materials.'\n"
            "Assistant: None\n\n"
            "User: Extract the fuel cell materials in this sentence: "
            "'The study discusses general electrodes and conductive materials.'\n"
            "Assistant: None"
        )}
    ]

    response = ollama.chat(
        model=modelo,
        messages=exemplos + [
            {"role": "user", "content": (
                f"Extract only the fuel cell materials that are explicitly identified as cathode, electrolyte, or anode in this sentence: {sentenca}. "
                "Provide them in a Python list format like this: ['Material1', 'Material2'] or respond 'None' if no materials of these types are mentioned."
            )}
        ],
        options={
            'temperature': temp,
            'top_p': tp,
            'top_k': tk,
        }
    )
    
    resposta = response['message']['content'].strip()
    print("Materiais OK")
    return "None" if resposta.lower() == "none" else resposta

def modelo_temperatura(sentenca, temp, tp, tk):
    exemplos = [
        {"role": "system", "content": (
            "You are an operation temperature extractor for scientific abstracts. "
            "Your task is to identify and extract the operating temperatures specifically mentioned in sentences about fuel cells or their components. "
            "Respond only with the temperature in the same format as mentioned in the text, e.g., '600 degrees C' or '1000 K'. "
            "If no specific temperature is mentioned in the sentence, respond only with 'None'. Avoid giving any explanations or additional information."
            "\n\n"
            "Examples:\n"
            "User: Extract the operating temperature from this sentence: "
            "'Experiments were also carried out at 600 degrees C using direct methanol fuels at the anode; the maximum power output was approximately half of that obtained with hydrogen.'\n"
            "Assistant: '600 degrees C'\n\n"
            "User: Extract the operating temperature from this sentence: "
            "'The performance was tested under various conditions but no temperature was explicitly provided.'\n"
            "Assistant: None\n\n"
            "User: Extract the operating temperature from this sentence: "
            "'This system operates effectively at temperatures as high as 1000 K.'\n"
            "Assistant: '1000 K'"
        )}
    ]

    response = ollama.chat(
        model=modelo,
        messages=exemplos + [
            {"role": "user", "content": (
                f"Extract the operating temperature specifically mentioned in this sentence: {sentenca}. "
                "Provide it in the same format as found in the text or respond 'None' if no temperature is mentioned."
            )}
        ],
        options={
            'temperature': temp,
            'top_p': tp,
            'top_k': tk,
        }
    )

    resposta = response['message']['content'].strip()
    print("Temperatura OK")
    return "None" if resposta.lower() == "none" else resposta

def modelo_power(sentenca, temp, tp, tk):
    exemplos = [
        {"role": "system", "content": (
            "You are a power density extractor for scientific abstracts. "
            "Your task is to identify and extract the power density or output power values specifically mentioned in sentences about fuel cells or their components. "
            "These values are often given in formats like 'mW/cm(2)', 'mW cm(-2)', or similar variations. "
            "If no specific power density or output power value is mentioned in the sentence, respond only with 'None'. "
            "Avoid giving any explanations or additional information."
            "\n\n"
            "Examples:\n"
            "User: Extract the power density from this sentence: "
            "'The maximum power density achieved was 500 mW/cm(2) at 800 degrees C.'\n"
            "Assistant: '500 mW/cm(2)'\n\n"
            "User: Extract the power density from this sentence: "
            "'This design delivered a power output of 200 mW cm(-2).'\n"
            "Assistant: '200 mW cm(-2)'\n\n"
            "User: Extract the power density from this sentence: "
            "'The study reports performance without specifying power density or output.'\n"
            "Assistant: None\n\n"
            "User: Extract the power density from this sentence: "
            "'It achieved 350 mW/cm(2) under standard conditions.'\n"
            "Assistant: '350 mW/cm(2)'"
        )}
    ]

    response = ollama.chat(
        model=modelo,
        messages=exemplos + [
            {"role": "user", "content": (
                f"Extract the power density or output power value specifically mentioned in this sentence: {sentenca}. "
                "Provide it in the same format as found in the text or respond 'None' if no power density is mentioned."
            )}
        ],
        options={
            'temperature': temp,
            'top_p': tp,
            'top_k': tk,
        }
    )

    resposta = response['message']['content'].strip()
    print("Densidade de Potencia OK")
    return "None" if resposta.lower() == "none" else resposta

def modelo_combustivel(sentenca, temp, tp, tk):
    exemplos = [
        {"role": "system", "content": (
            "You are a fuel type extractor for scientific abstracts. "
            "Your task is to identify and extract the type of fuel explicitly mentioned in sentences about solid oxide fuel cells (SOFCs). "
            "Look for phrases like 'as fuel' or similar contexts to determine the specific fuel being used, e.g., '3% humidified hydrogen', 'methanol', or 'natural gas'. "
            "If no specific fuel is mentioned in the sentence, respond only with 'None'. "
            "Avoid giving any explanations or additional information."
            "\n\n"
            "Examples:\n"
            "User: Extract the type of fuel from this sentence: "
            "'The performance of the cell with homogeneous microstructure reaches 389 mW cm(-2) at 800 degrees C in condition of 3% humidified hydrogen as fuel and normal air as oxidant.'\n"
            "Assistant: '3% humidified hydrogen'\n\n"
            "User: Extract the type of fuel from this sentence: "
            "'Experiments were performed using direct methanol fuel.'\n"
            "Assistant: 'methanol'\n\n"
            "User: Extract the type of fuel from this sentence: "
            "'The study focuses on cell performance but does not specify any particular fuel.'\n"
            "Assistant: None\n\n"
            "User: Extract the type of fuel from this sentence: "
            "'Natural gas was used as fuel in this experiment.'\n"
            "Assistant: 'Natural gas'"
        )}    
    ]

    response = ollama.chat(
        model=modelo,
        messages=exemplos + [
            {"role": "user", "content": (
                f"Extract the type of fuel explicitly mentioned in this sentence: {sentenca}. "
                "Provide it in the same format as found in the text or respond 'None' if no fuel is mentioned."
            )}
        ],
        options={
            'temperature': temp,
            'top_p': tp,
            'top_k': tk,
        }
    )

    resposta = response['message']['content'].strip()
    print("Combustivel OK")
    return "None" if resposta.lower() == "none" else resposta


def modelo_classificador(resposta, sentenca, temp, tp, tk):
    exemplos = [
        {"role": "system", "content": (
            "You are a material role classifier for scientific abstracts. "
            "Your task is to classify materials as either cathode, anode, or electrolyte based on their explicit association in the sentence provided. "
            "Respond only in Python dictionary format, e.g., {'cathode': ['Material1'], 'anode': ['Material2'], 'electrolyte': ['Material3']}. "
            "If no materials are found for a category, the category should have an empty list. If no roles can be assigned to any material, "
            "respond with {'cathode': [], 'anode': [], 'electrolyte': []}. "
            "Avoid giving any explanations or additional information."
            "\n\n"
            "Examples:\n"
            "User: Classify the materials in this sentence: "
            "'BaIn0.3Ti0.7O2.85 (BIT07) can be used as electrolyte material with LSCF as cathode and BIT07-Ni cermet as anode.'\n"
            "Assistant: {'cathode': ['LSCF'], 'anode': ['BIT07-Ni cermet'], 'electrolyte': ['BaIn0.3Ti0.7O2.85']}\n\n"
            "User: Classify the materials in this sentence: "
            "'The study mentions general electrode materials but does not specify their roles.'\n"
            "Assistant: {'cathode': [], 'anode': [], 'electrolyte': []}"
        )}
    ]

    # Chamada ao modelo
    response = ollama.chat(
        model=modelo,
        messages=exemplos + [
            {"role": "user", "content": (
                f"Classify these materials based on the sentence: {sentenca}. "
                f"Materials to classify: {resposta}. "
                "Respond only in Python dictionary format."
            )}
        ],
        options={
            'temperature': temp,
            'top_p': tp,
            'top_k': tk,
        }
    )
    
    # Processar resposta e substituir listas vazias por None
    classificacao = eval(response['message']['content'].strip())
    for key in ['cathode', 'anode', 'electrolyte']:
        if not classificacao.get(key):  # Se a chave não existe ou está vazia
            classificacao[key] = None

    print("Classificacao OK")
    return classificacao



# Listas de parâmetros
temperaturas = [0]
top_ps = [0.25]
top_ks = [1]

# Lista para armazenar os resultados
resultados = []

artigos_ok = 0
sentencas_ok = 0

try:
    for abstract_idx, l in enumerate(dfs):
        artigos_ok = artigos_ok + 1
        porcentagem = artigos_ok/quantidade_artigos
        print(f'Progresso {artigos_ok}/{quantidade_artigos} ({porcentagem:.2f})')
        for idx, sentenca in enumerate(l["Sentencas"]):
            for temp in temperaturas:
                for tp in top_ps:
                    for tk in top_ks:
                        try:
                            sentencas_ok = sentencas_ok + 1
                            sent_porc = sentencas_ok/c
                            print(f'Progresso: {artigos_ok}/{quantidade_artigos} - Sentenca {sentencas_ok}/{c} ({sent_porc:.2f})')
                            resposta = modelo_extrator(sentenca, temp, tp, tk)
                            #time.sleep(0.5)
                            resposta_temperatura = modelo_temperatura(sentenca, temp, tp, tk)
                            #time.sleep(0.5)
                            resposta_power = modelo_power(sentenca, temp, tp, tk)
                            #time.sleep(0.5)
                            resposta_combustivel = modelo_combustivel(sentenca, temp, tp, tk)
                            #time.sleep(0.5)

                            # Inicializa os campos padrão
                            classificacao = {'cathode': None, 'anode': None, 'electrolyte': None}

                            if resposta != 'None':
                                classificacao = modelo_classificador(resposta, sentenca, temp, tp, tk)
                            else:
                                classificacao = {'cathode': 'None', 'anode': 'None', 'electrolyte': 'None'}

                            # Adicionar ao DataFrame
                            resultados.append({
                                "artigo_id": l["artigo_id"].iloc[0],
                                "modelo": modelo,
                                "abstract_id": abstract_idx + 1,
                                "sentenca_id": idx + 1,
                                "sentenca": sentenca,
                                "temperatura": temp,
                                "top_p": tp,
                                "top_k": tk,
                                "resposta": resposta,
                                "catodo": classificacao['cathode'],
                                "anodo": classificacao['anode'],
                                "eletrolito": classificacao['electrolyte'],
                                "temp_op": resposta_temperatura,
                                "dens_pot": resposta_power,
                                "combustivel": resposta_combustivel,
                            })
                            #time.sleep(1)  # Pausa de 1 segundo
                            print()

                        except Exception as e:
                            print(f"Ocorreu um erro durante a execução do programa. Progresso salvo!")
                            df_resultados = pd.DataFrame(resultados)
                            df_resultados.to_excel('resultados1500.xlsx', index=False)
                            print("Processo Finalizado COM ERRO!")

                            

except Exception as e:
    print()
    print(f"Ocorreu um erro durante a execução do programa. Progresso salvo!")
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_excel('resultados1500.xlsx', index=False)
    print("Processo Finalizado COM ERRO!")

print()
print(f"Nenhum erro ocorreu!")
df_resultados = pd.DataFrame(resultados)
df_resultados.to_excel('resultados1500.xlsx', index=False)
print("PROCESSO FINALIZADO COM SUCESSO!")
