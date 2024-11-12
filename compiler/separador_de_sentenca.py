import pandas as pd
import numpy as np
import os
import spacy

# Carrega o modelo de linguagem do spaCy
nlp = spacy.load('en_core_web_lg')

# Caminhos para as pastas de entrada e saída
pasta_entrada = '../concatened_excel'
pasta_saida = '../abstracts_sentences'

# Carrega o arquivo CSV de entrada
caminho_entrada = os.path.join(pasta_entrada, 'wos_dataset.csv')
df = pd.read_csv(caminho_entrada, low_memory=False)

# Remove linhas com valores NaN na coluna 'Abstract'
df_corrigido = df.dropna(subset=['Abstract'])
numero_de_abstracts = len(df_corrigido)

print("Numero de Abstracts com valores NaN:", len(df))
print("Numero de Abstracts sem valores NaN:", numero_de_abstracts)

# Gera um array com números de 1 até o número de abstracts sem NaN
numero_de_abstracts = np.arange(1, numero_de_abstracts + 1, 1)

# Cria a pasta de saída, caso ela não exista
if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

# Processa cada abstract e salva as sentenças em arquivos CSV individuais
for i, j in zip(df_corrigido['Abstract'], numero_de_abstracts):
    dados = {}
    csv = f'ABS{j}.csv'
    doc = nlp(i)
    sentencas = []
    titulo = []
    DOI = []
    n_artigo = []
    ID = []
    
    print("Abstract numero:", j)
    
    # Extrai as sentenças do abstract
    for sentenca in doc.sents:
        sentencas.append(sentenca.text)
    
    # Extrai título e DOI do artigo
    titulo_do_artigo = df_corrigido.iloc[j-1]["Article Title"]
    DOI_do_artigo = df_corrigido.iloc[j-1]['DOI']
    
    contador = 1
    
    # Prepara os dados para o DataFrame
    for k in sentencas:
        ID.append(contador)
        n_artigo.append(j)
        titulo.append(titulo_do_artigo)
        DOI.append(DOI_do_artigo)
        
        contador += 1

    # Cria um DataFrame com as informações extraídas
    dados = {
        'ID': ID,
        'Artigo': n_artigo,
        'Sentencas': sentencas,
        'Titulo': titulo,
        'DOI': DOI,
    }
    
    df_csv = pd.DataFrame(dados)

    # Caminho para salvar o arquivo CSV individual
    caminho_csv = os.path.join(pasta_saida, csv)

    # Salva o DataFrame em um arquivo CSV
    df_csv.to_csv(caminho_csv, index=False)
