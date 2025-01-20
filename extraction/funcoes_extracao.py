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
import ast

def extrair_materiais(valor):

    if pd.isna(valor):
        return []
    try:
        # Avalia a string de forma segura como uma lista
        return ast.literal_eval(valor)
    except (ValueError, SyntaxError):
        return []

def extrair_valores_simples(valor):

    if pd.isna(valor):
        return []
    try:
        return [valor.strip("'")]
    except AttributeError:
        return []

def concatenador(df):

    df['resposta'] = df['resposta'].apply(extrair_materiais)
    df['catodo'] = df['catodo'].apply(extrair_materiais)
    df['anodo'] = df['anodo'].apply(extrair_materiais)
    df['eletrolito'] = df['eletrolito'].apply(extrair_materiais)
    df['temp_op'] = df['temp_op'].apply(extrair_valores_simples)
    df['dens_pot'] = df['dens_pot'].apply(extrair_valores_simples)
    df['combustivel'] = df['combustivel'].apply(extrair_valores_simples)

    df_agregado = (
        df.groupby(['artigo_id', 'abstract_id'])
        .agg({
            'sentenca_id': 'max', 
            'sentenca': ' '.join, 
            'modelo': 'first', 
            'temperatura': 'first', 
            'top_p': 'first',
            'top_k': 'first',
            'resposta': lambda x: [item for sublist in x for item in sublist],  
            'catodo': lambda x: [item for sublist in x for item in sublist],
            'anodo': lambda x: [item for sublist in x for item in sublist],
            'eletrolito': lambda x: [item for sublist in x for item in sublist],
            'temp_op': lambda x: [item for sublist in x for item in sublist],
            'dens_pot': lambda x: [item for sublist in x for item in sublist],
            'combustivel': lambda x: [item for sublist in x for item in sublist],
        })
        .reset_index()
    )

    return df_agregado

def extrair_valor(texto):

    if not isinstance(texto, str):
        return 0.0
    match = re.match(r'^(\d+(\.\d+)?)', texto)
    if match:
        return float(match.group(1))
    return 0.0

def processar_densidade(cell):
    if isinstance(cell, list) and cell:
        # Unir todos os itens da lista em uma única string
        combined = ' '.join(cell)
        # Procurar por números válidos (evitando notações como cm(-2))
        numbers = re.findall(r"\b\d+\.?\d*\b(?=\s*(?:W|MW|mW))", combined)
        # Converter os números para float
        numbers = list(map(float, numbers))
        # Retornar a média dos números, se houver algum
        return sum(numbers) / len(numbers) if numbers else None
    return None

def processar_temperatura(temps):
        # Regex para encontrar números
        matches = re.findall(r"(\d+(?:[.,]\d+)?(?:-\d+(?:[.,]\d+)?)?)", str(temps))
        extracted = []
        
        for match in matches:
            if '-' in match:
                start, end = map(lambda x: float(x.replace(',', '')), match.split('-'))
                extracted.extend([start, end])
            else:
                extracted.append(float(match.replace(',', '')))
        
        # Converter para Celcius
        if "K" in str(temps):
            extracted = [round(temp - 273.15, 2) if temp > 273 else temp for temp in extracted]

        return np.mean(extracted) if extracted else None