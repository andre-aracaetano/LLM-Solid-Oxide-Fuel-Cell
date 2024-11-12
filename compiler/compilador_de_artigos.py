import pandas as pd
import os

# Caminho para a pasta contendo os arquivos de entrada
pasta_entrada = '../wos_excels'
# Caminho para a pasta onde o arquivo de saída será salvo
pasta_saida = '../concatened_excel'

# Lista para armazenar os dataframes lidos
dataframes = []

# Itera sobre os arquivos na pasta de entrada
for arquivo in os.listdir(pasta_entrada):
    if arquivo.endswith('.xls'):
        caminho_arquivo = os.path.join(pasta_entrada, arquivo)
        df = pd.read_excel(caminho_arquivo)
        dataframes.append(df)

# Concatena todos os dataframes
df_concatenado = pd.concat(dataframes, ignore_index=True)

# Garante que a pasta de saída exista
os.makedirs(pasta_saida, exist_ok=True)

# Caminho do arquivo de saída
caminho_saida = os.path.join(pasta_saida, 'wos_dataset.csv')

# Salva o dataframe concatenado em um arquivo CSV
df_concatenado.to_csv(caminho_saida, sep=',', index=False, na_rep='NaN')
