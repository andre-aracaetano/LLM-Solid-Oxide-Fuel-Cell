# -*- coding: utf-8 -*-
 
import pandas as pd
import ollama
import numpy as np
import random
from pathlib import Path
import os
import time
import unicodedata
import json
import re
import ast
 
print()
 
df = pd.read_excel('dataset/df_original_wos.xlsx')

model = 'mistral-large:123b-instruct-2407-q2_K'
 
def model_extrator(abstract, titulo):
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": (
                "You are a material extractor for scientific abstracts about Solid Oxide Fuel Cells (SOFC)."
                " Your task is to extract all mentioned materials for cathode, anode, and electrolyte,"
                " as well as determine the best material if explicitly stated."
                " If a material is given with its abbreviation (e.g., Yttrium-Stabilized Zirconia (YSZ)),"
                " extract both the full name and abbreviation."
                " Ensure to extract known abbreviations even if the full name isn't present but indicated (e.g., used LSM as cathode)."
                "\n\nExtraction Criteria:\n"
                "- Identify all materials used for cathode, anode, and electrolyte."
                "- If the best-performing material is explicitly stated, extract only that one."
                "- Extract the operation temperature (the temperature that is used in experiment to take the power densisty) at which the highest performance was reported."
                "- Extract power density values in mW/cm(2) or W/cm(2) or W cm(-2)."
                "- Ignore performance metrics other than power density."
                "- If no relevant data is found, return ('None', 'None', 'None', 'None', 'None')."
                "\n\nOutput Format:\n"
                "- Return only the extracted data as a tuple in the format: (cathode, anode, electrolyte, power_density, temperature)."
                "- Do not add any explanation, label, or additional text only return the tuple."
                "\n\nExample 1:\n"
                "Title: Advanced SOFC cathode materials\n"
                "Abstract: The study analyzed LSM, LSCF, and BSCF as cathode materials, concluding that BSCF exhibited the highest power density of 1200 mW/cm2 at 750 degrees C."
                "Extracted Data: ('BSCF', 'None', 'None', '1200 mW/cm2', '750 degrees C')\n\n"
                "Example 2:\n"
                "Title: Novel electrolyte for SOFC applications\n"
                "Abstract: The electrolytes tested included Yttrium-Stabilized Zirconia (YSZ) and SDC. The highest performance was observed for SDC at 800 degrees C with 1000 mW/cm2."
                "Extracted Data: ('None', 'None', 'SDC', '1000 mW/cm2', '800 degrees C')\n\n"
                "Example 3:\n"
                "Title: Novel SOFC materials\n"
                "Abstract: A solid oxide fuel cell using La0.6Sr0.4Co0.2Fe0.8O3 (LSCF) as a cathode, Ni as an anode, and Yttrium-Stabilized Zirconia (YSZ) as an electrolyte achieved a peak power density of 900 mW/cm(2) at 750 degrees C.\n"
                "Extracted Data: ('La0.6Sr0.4Co0.2Fe0.8O3 (LSCF)', 'Ni', 'Yttrium-Stabilized Zirconia (YSZ)', '900 mW/cm(2)', '750 degrees C')\n\n"
            )},
            {"role": "user", "content": f"Title: {titulo}\nAbstract: {abstract}\nExtract all materials for cathode, anode, and electrolyte, identifying the best-performing one, along with power density and temperature."}
        ],
        options={"temperature": temp, "top_p": tp, "top_k": tk, 'num_ctx': 8000,}
    )
   
    conteudo_bruto = response.message.content
    match = re.search(r"\(.*\)", conteudo_bruto, re.DOTALL)
   
    if match:
        return match.group(0)
    else:
        return "('None', 'None', 'None', 'None', 'None')"
 
temp = 0
tp = 0.25
tk = 1
results = []
 
print("~~~~~~~~~~~~~~~~~~ EXTRACTION ~~~~~~~~~~~~~~~~~~")
pro = 0

len_articles = len(df)

try:
    for x_abstract, x_title, x_doi, x_year, x_wos in zip(df["abstract"], df["title"], df["DOI"], df["year"], df["wos_id"]):
 
        title_article = x_title
        pro += 1
        p = pro/len_articles
        print(f"Progress: {pro}/{len_articles} ({p:.3%})")
        pre_abstract = []
 
        abstract = x_abstract
 
        print("Title:", title_article)
        print("Abstract:", abstract)

        extraction = model_extrator(abstract, title_article)
        print("Extraction:", extraction)

        results.append({
        "title": title_article,
        "article_id": pro,
        "model": model,
        "abstract": abstract,
        "extraction": extraction,
        "DOI":x_doi,
        "year":x_year,
        "wos_id":x_wos,
        })
        print()
                           
except Exception as e:
    df_extraction = pd.DataFrame(results)
    df_extraction.to_excel('df_extration_preprocessed.xlsx', index=False)
    print("Erro", e)
 
df_extraction = pd.DataFrame(results)
df_extraction.to_excel('df_extration_preprocessed.xlsx', index=False)

print("~~~~~~~~~~~~~~~~~~ COMPLET ~~~~~~~~~~~~~~~~~~")