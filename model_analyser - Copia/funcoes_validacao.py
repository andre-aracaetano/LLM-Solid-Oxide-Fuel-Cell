import pandas as pd 
import numpy as np 
import re
import matplotlib.pyplot as plt
import pandas as pd
import re
from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Função principal para avaliar várias configurações de modelo
def avaliar_modelos(df_respostas_certas, df_modelo, temperaturas, top_ps, top_ks, limite_similaridade=0.7, vetorizador = TfidfVectorizer(), mostrar = True):
    
    vectorizer = vetorizador
    # Função auxiliar para extrair materiais de uma resposta formatada como lista de strings - Tipo 1
    def extrair_materiais(resposta):
        if not isinstance(resposta, str) or resposta == "None":
            return set()
        
        materiais = re.findall(r"\[([^\]]*)\]", resposta)
        materiais_extraidos = set()
        for material in materiais:
            materiais_extraidos.update(re.findall(r"['\"]([^'\"]*)['\"]", material))
        return materiais_extraidos
    
    # # Função para extrair materiais de uma resposta formatada como lista de strings - Tipo 2
    # def extrair_materiais(resposta):
    #     if not isinstance(resposta, str) or resposta == "None":
    #         return set()
        
    #     # Extrai elementos contidos entre aspas simples
    #     return set(re.findall(r"'(.*?)'", resposta))

    # Função auxiliar para comparar respostas do modelo com as respostas corretas
    def contar_acertos(df_respostas_certas, df_modelo, config):
        temp, top_p, top_k = config
        subset = df_modelo[(df_modelo['temperatura'] == temp) & 
                           (df_modelo['top_p'] == top_p) & 
                           (df_modelo['top_k'] == top_k)]
        
        acertos = 0
        
        for i, row in subset.iterrows():
            # Identifica o abstract e a sentença correspondente
            abstract_id = row['abstract_id']
            sentenca_id = row['sentenca_id']
            
            # Seleciona a resposta certa
            resposta_certa_row = df_respostas_certas[(df_respostas_certas['abstract_id'] == abstract_id) &
                                                     (df_respostas_certas['sentenca_id'] == sentenca_id)]
            
            if resposta_certa_row.empty:
                continue
            
            resposta_certa = resposta_certa_row['resposta_certa'].values[0]
            materiais_certos = extrair_materiais(resposta_certa)
            materiais_preditos = extrair_materiais(row['resposta'])

            text_1 = f"{materiais_preditos}"
            text_2 = f"{materiais_certos}"

            # Usar TF-IDF para representar as listas como vetores
            tfidf_matrix = vectorizer.fit_transform([text_1, text_2])

            #print("TDIDF:", tfidf_matrix)

            # Calcular a similaridade do cosseno entre as duas representações vetoriais
            cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

            if mostrar == True:

                print(f"Configuração {config} - Abstract ID: {abstract_id}, Sentença ID: {sentenca_id}")
                print("Sentença do Modelo:", row['sentenca'])
                print("Resposta do Modelo:", row['resposta'])
                print("Materiais corretos:", materiais_certos)
                print("Materiais preditos pelo modelo:", materiais_preditos)
                print("Similaridade do cosseno:", cos_sim[0][0])

            if cos_sim >= limite_similaridade:
                acertos += 1

                if mostrar == True:
                    print("Acertou")
            else:
                if mostrar == True:
                  print("Errou")
            
            if mostrar == True:
               print()

        return acertos

    # Avalia cada configuração
    configuracoes = list(product(temperaturas, top_ps, top_ks))
    resultados = []

    for config in configuracoes:

        if mostrar == True:
            print("Configuração concluída:", config)
            print()
        acertos = contar_acertos(df_respostas_certas, df_modelo, config)
        resultados.append((config, acertos))

        if mostrar == True:
            print("Número de acertos:", acertos)
            print()

    # Ordena os resultados pela quantidade de acertos, em ordem decrescente
    resultados_ordenados = sorted(resultados, key=lambda x: x[1], reverse=True)

    print("Melhor modelo:", resultados_ordenados[0])
    return resultados_ordenados

def plotar_resultados(resultados_ordenados, df_respostas_certas, df_modelo):
    # Calcula a precisão normalizada para cada configuração
    acertos_normalizados = [acertos / len(df_respostas_certas) for _, acertos in resultados_ordenados]
    configuracoes = [config for config, _ in resultados_ordenados]
    
    # Define as cores de acordo com a temperatura
    cores = ['skyblue' if config[0] == 0 else 'salmon' if config[0] == 0.5 else 'lightgreen' for config in configuracoes]
    
    # Cria o gráfico
    plt.figure(figsize=(12, 5))
    bars = plt.bar(
        range(len(configuracoes)), 
        acertos_normalizados, 
        color=cores, 
        tick_label=[f"T: {cfg[0]}, P: {cfg[1]}, K: {cfg[2]}" for cfg in configuracoes], 
    )
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Configurações (Temperatura, Top_p, Top_k)")
    plt.ylabel("Precisão (Acertos/Total)")

    # Ajusta o título com o nome do modelo e o número total de sentenças
    modelo = df_modelo["modelo"].iloc[0]  # Assume que o modelo é o mesmo para todo o df_modelo
    plt.title(f'Precisão Modelo {modelo} dado {len(df_respostas_certas)} sentenças')
    plt.grid(axis='y')

    # Adiciona rótulos aos pontos das barras
    for bar, precisao in zip(bars, acertos_normalizados):
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            bar.get_height() + 0.02,
            f"{precisao:.2f}", 
            ha='center', 
            va='bottom'
        )

    normalizador = 1.0 - max(acertos_normalizados)

    plt.ylim(0, max(acertos_normalizados) + normalizador + 0.1)

    # Cria a legenda para as cores de temperatura
    handles = [
        Patch(color='skyblue', label='Temperatura: 0'),
        Patch(color='salmon', label='Temperatura: 0.5'),
        Patch(color='lightgreen', label='Temperatura: 1')
    ]
    plt.legend(handles=handles, loc='upper right')

    plt.tight_layout()
    plt.show()
