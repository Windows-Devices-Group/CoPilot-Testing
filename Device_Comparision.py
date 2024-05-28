#!/usr/bin/env python
# coding: utf-8

# In[168]:


import streamlit as st
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
import io
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
import openai
import pyodbc
import urllib
from sqlalchemy import create_engine
import pandas as pd
from azure.identity import InteractiveBrowserCredential
from pandasai import SmartDataframe
import pandas as pd
from pandasai.llm import AzureOpenAI
import matplotlib.pyplot as plt
import os
import time
from PIL import Image
import base64
import pandasql as ps
from openai import AzureOpenAI
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["AZURE_OPENAI_API_KEY"] = "3a3850af863b4dddbc2d3834f0ff097b"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://fordmustang.openai.azure.com/"

Sentiment_Data  = pd.read_csv("Windows_Data_116K.csv")

def Sentiment_Score_Derivation(value):
    try:
        if value == "positive":
            return 1
        elif value == "negative":
            return -1
        else:
            return 0
    except Exception as e:
        err = f"An error occurred while deriving Sentiment Score: {e}"
        return err    

#Deriving Sentiment Score and Review Count columns into the dataset
Sentiment_Data["Sentiment_Score"] = Sentiment_Data["Sentiment"].apply(Sentiment_Score_Derivation)
Sentiment_Data["Review_Count"] = 1.0


# In[ ]:


os.environ["AZURE_OPENAI_API_KEY"] = "3a3850af863b4dddbc2d3834f0ff097b"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://fordmustang.openai.azure.com/"
client = AzureOpenAI(
api_key=os.getenv("3a3850af863b4dddbc2d3834f0ff097b"),  
api_version="2024-02-01",
azure_endpoint = os.getenv("https://fordmustang.openai.azure.com/")
)

deployment_name='Surface_Analytics'

start_phrase_verbatim = """

    Your task is to generate SQL queries for analyzing user reviews stored in a table named Sentiment_Data in Microsoft SQL Server Management Studio (SSMS). 
    Each row in the table represents a user review and contains the following columns:

    Review: The text of the user review.
    Data_Source: The source from which the review was collected.
    Geography: The country or region from which the review was given.
    Title: The title of the review.
    Review_Date: The date when the review was posted.
    Product: The corresponding product for the review, which can be "Windows 11 (Preinstall)" or "Windows 10".
    Product_Family: The specific model or type of the product like Asus Rog Zephyrus G16 16,Hp Omen Transcend 14,Lenovo Legion Pro 5I 16,Asus Rog Strix Scar 18,Asus Vivobook Pro 16,Asus Rog Zephyrus G14 14,Lenovo Slim 7I 16,Asus Zenbook 14,Dell G16 16,Asus Rog Strix 18,Razer Blade 16,Msi Prestige 16,Hp Laptop 14,Dell G15 15,Hp Spectre X360 16,Hp Omen 14,Razer Blade 14,Hp Spectre X360 14,Hp Pavilion 14,Lenovo V15 15,Hp G10 15,Lenovo Thinkpad E16 16,Hp G9 15,Lenovo V14 14,Lenovo Ideapad 1 14,Lenovo Thinkpad P17 17,Hp Laptop 17,Samsung Galaxy Book3 360 15,Lenovo Ideapad Slim 3 15,Acer Aspire 5 15,Dell Alienware X16 16,Asus Rog Strix G17 17,Asus Tuf Dash 15,Acer Aspire Vero 14,Microsoft Surface Laptop 4 15,Lenovo Ideapad 1 15,Hp Laptop 15,Lenovo Ideapad 3 15,Microsoft Sorface Laptop Go 3 12,Asus Zenbook Duo 14,Hp Stream 14,Asus Laptop 15,Hp Pavilion 15,Dell Inspiron 3000 15,Lenovo Slim 7 Pro X 14,Lg Gram 17,Acer Swift Go 14,Msi Stealth 16,Msi Titan 18,Asus Tuf Gaming F15 15,Dell Alienware M18 18,Asus Vivobook 15,Dell Precision 3000 15,Hp Envy X360 15,Asus Zenbook 14X 14,Hp Omen Transcend 16,Hp Envy 17T 17,Asus Rog Strix G16 16,Hp Envy 16,Hp Victus 15,Lenovo Ideapad 14,Microsoft Surface Laptop 15,Lenovo Thinkpad Yoga L13 13,Asus Rog Strix G15 15,Asus Zenbook Pro 14X 14,Razer Blade Gaming 15,Hp Pavilion 17,Asus Rog Strix G17  17,Asus Tuf Gaming A15 15,Msi Gaming Cyborg 15,Acer Predator Helios 16,Asus Tuf Gaming A17 17,Acer Nitro 5 15,Asus Vivobook Flip 16,Lenovo Thinkpad P14S 14,Dell Xps 13,Asus Rog Flow Z13 13,Msi Thin Gaming Gf63 15,Dell Inspiron 15,Microsoft Surface Laptop Go 12,Lenovo Ideapad 3I 15,Lenovo Laptop 15,Lenovo Legion Pro 7 16,Asus Rog Zephyrus M16 16,Lenovo Yoga 7I 14,Samsung Galaxy Book4 Pro 14,Hp Envy X360 16,Asus Zenbook Pro 14,Lenovo Thinkbook G6 16,Lenovo Thinkpad X1 Carbon 14,Samsung Galaxy Book2 Pro 360 13,Microsoft Surface Laptop Studio 14,Lenovo Thinkpad Yoga X1 14,Samsung Galaxy Book4 Pro 16,Microsoft Sorface Laptop Go 2 12,Dell Inspiron 14,Hp Envy X360 14,Microsoft Surface Go 13,Hp Omen  15,Dell Inspiron 3000  15,Lenovo Ideapad  15,Lenovo Slimpro 9I 14,Dell Inspiron 16,Hp Omen Gaming Laptop,Asus Vivobook 16,Hp Envy 2,Hp Envy  17,Asus Rog Flow 16,Asus Tuf Gaming A16 16,Asus Rog Gaming 14,Lenovo Loq 15,Lenovo Yoga 7 16,Lenovo Slim Pro 14,Lenovo Yoga 6 13,Lenovo Yoga 7I 16,Lenovo Yoga Book 9,Lenovo Ideapad 15,Samsung Galaxy Book 3 360 15,Microsoft Surface Laptop 5 13,Microsoft Surface Pro 9 13,Microsoft Surface Laptop 5 15,Samsung Galaxy Book 2 Pro 360 15,Microsoft Surface Pro 7 Plus 12,Microsoft Surface Laptop 4 13,Lenovo Ideapad 1  14,Alienware M16 16,Dell Xps 15,Hp Omen 16,Asus Vivobook 14,Lenovo Yoga 9I 14,Samsung Galaxy Book3 Pro 16,Samsung Galaxy Book 3 Pro 360 16,Acer Aspire 3 15,Hp Pavilion X360 14,Microsoft Surface Studio 2 14,Hp Stream 11,Asus Vivobook S 15,Lenovo Ideapad 16,Asus Vivobook 17,Hp Laptop  17,Asus Rog Zephyrus 16,Samsung Galaxy Book3 Ultra 16,Dell Inspiron 3000 14,Acer Predator Helios 18,Lenovo Thinkpad X1 14,Acer Nitro 17,Microsoft Surface Go Laptop 3 12,Msi Stealth 14,Asus Proart Studiobook 16,Asus Rog Zephyrus 14,Asus Rog 16,Lenovo Gaming Laptop,Samsung Galaxy Book 2 Pro 13,Microsoft Surface Laptop Go 2 12,Asus Vivobook  16,Hp Notebook 15,Hp Laptop  15,Dell Inspiron 5000  16,Microsoft Surface Go 2 10,Dell Inspiron 7000 14 Plus,Asus Tuf A16 16,Lenovo Legion 5  14,Asus Laptop 11,Lenovo Legion Slim 5 16,Msi Bravo 15,Acer Predator Helios Neo 16,Lenovo Legion 16,Samsung Galaxy Tab S7 Fe 12,Acer Aspire 1 15,Asus Tuf Gaming A 17,Dell Alienware  16,Samsung Galaxy Book 3 360 13,Lenovo Ideapad 3I 14,Asus Br1100 11,Lenovo Ideapad 3 14,Acer Aspire Vero 15,Lenovo Yoga 9I 13,Microsoft Surface Laptop Studio2 13,Asus Zenbook Pro Duo 15,Lenovo Wei 9 Pro  16,Hp Envy X360  15,Asus Rog Flow X13,Lenovo Wei 7 Pro  14,Asus Tuf Gaming 15,Hp Spectre X360 13,Microsoft Surface Go 3 10,Samsung Galaxy Ultra 16,Asus Vivobook Pro 15,Asus Vivobook Pro 16X  16,Acer Predator Helios 300 15,Asus Rog Strix G 17,Lenovo Ideapad Flex 5 14,Asus L210 11,Lenovo Ideapad Gaming 3 15,Asus Rog Strix G 16,Lenovo Ideapad 1I 14,Hp Envy 17,Lenovo Ideapad 3 17,Asus Rog Zephyrus G14  14,Lenovo Flex 5 14,Microsoft Surface Pro 8 13,Acer Swift Edge 16,Asus Rog Flow X13 13,Alienware M18 18,Asus Tuf Gaming A15  15,Lenovo Legion Pro 7I 16,Asus Vivobook Go 15,Acer Swift 3 14,Asus Rog Strix 17,Msi Cyborg 15,Hp Stream  14,Asus Vivobook 11,Dell G15 5000 15,Samusng Galaxy Book 3 Pro 14,Dell Inspiron 7000 2-In-1 16,Msi Gaming Laptop 16,Msi Stealth 17,Msi Gaming 15,Dell Alienware M16 16,Lenovo Ideapad Flex 5  14,Asus Tuf Gaming F15,Lenovo Ideapad Flex 5 16,Lenovo Laptop 14,Msi Gamingthin 15,Asus Rog Zephyrus Duo 16,Acer Swift X 14,Msi Gf63 Thin 15,Microsoft Surface Pro 7 12,Lenovo Flex 11,Lenovo Wei 5 Pro  15,Asus Vivobook Go 14,Asus Vivobook  14,Msi Katana 15,Hp Laptop 17T 17,Asus Vivobook Go 12,Asus L510 15,Microsoft Surface Go 10,Dell Insprion 3000 11,Asus Zenbook Pro 15,Dell Alienware M15 R7 15,Hp Envy Desktop,Asus Zenbook Duo 15,Lg Gram 16,Asus Rog Strix Scar 15,Asus Tuf Gaming F17 17,Msi Sword 15,Acer Nitro 5 17,Dell Xps 17,Lenovo Legion 5 15,Razer Gaming Laptop 18,Asus Vivobook Pro  16,Dell Inspiron 5000 14,Samsung Galaxy Book3 15,Hp Eliteboook G7 14,Lenovo Yoga C740 15,Lenovo Yoga Slim 7 16,Lenovo Slim 7 Pro 14,Lg Gram 14,Hp Spectre 17,Asus Zenbook Pro Duo 14,Asus Rog Zephyrus G16  16,Hp Victus 16,Msi Summit Flip 14,Asus Zenbook Pro 17,Acer Nitro 16,Lenovo Thinkpad T16 16,Asus Zenbook S 13,Msi Raider 17,Asus Rog Strix Scar 16,Msi Vector 16,Razer Blade 15,Dell Alienware M17 R5 17,Dell Xps Plus 13,Lenovo Gaming Desktop,Asus Rog Flow X16 16,Hp Stream  17,Asus Zenbook Flip 14,Razer Blade 18,Msi Thin Gf63 15,Asus Vivobook Pro 16X 16,Asus Vivobook Flip 14,Asus Vivobook Pro X 16,Dell Inspiron 5620 Desktop,Lenovo 300W 3 11,Asus Zenbook 15,Acer Aspire 3 14,Samsung Galaxy Book2 Pro 15,Asus Vivobook Pro 14,Asus Vivobook Pro 15X,Samsung Galaxy Book 2 Pro 15,Msi Gv15 15,Asus Rog Strix G15  15,Lenovo Ideapad 3  15,Lg Gram 15,Samsung Galaxy Book3 16,Lg Gram 17Z95P,Lenovo Ideapad Flex 5 15,Asus Rog Strix G16   16,Lenovo Thinkpad Yoga 11E,Msi Creator M 16,Lenovo Ideapad 5I Pro 16,Asus E410 14,Dell Inspiron 15 3000,Samsung Galaxy Book3 Pro 15,Dell Inspiron 3501 15,Asus Vivobook 13,Microsoft Surface Laptop 3 13,Lenovo Ideapad 5I 15,Dell Alienware X15 R2 15,Asus Vivobook Pro 15X 15,Samsung Galaxy Book Go 14.
    Sentiment: The sentiment of the review, which can be 'Positive', 'Neutral', or 'Negative'.
    Aspect: The aspect or feature of the product discussed in the review like "Audio-Microphone","Software","Performance","Storage/Memory","Keyboard","Browser","Connectivity","Hardware","Display","Graphics","Battery","Gaming","Design","Ports","Price","Camera","Customer-Service","Touchpad","Account","Generic".
    Keywords: Keywords mentioned in the review.
    Review_Count: Always 1 for each review.
    Sentiment_Score: A score based on the sentiment, which can be 1, 0, or -1.
    Your queries should allow analysis based on various criteria, including product family, sentiment, aspect, geography, etc. Always use the LIKE keyword for pattern matching in SQL queries.

    For example, to retrieve all rows and columns for a specific product family (e.g., "Asus Rog Zephyrus G16 16"), your query should look like:

        SELECT *
        FROM Sentiment_Data
        WHERE Product_Family LIKE '%Asus Rog Zephyrus G16 16%'
    Ensure that your queries follow SSMS syntax and can be executed directly in Microsoft SQL Server Management Studio.
    IMPORTANT : Give only the SQL Query if i pass the Query directly it should excute without any error.
    User Question :
    
    """
def query_verbatims(review):
    SQL_Query_Temp = client.completions.create(model=deployment_name, prompt=start_phrase_verbatim+review, max_tokens=1000,temperature=0.2)
    SQL_Query = SQL_Query_Temp.choices[0].text
    data_verbatims = ps.sqldf(SQL_Query,globals())
    return data_verbatims


# In[ ]:


def convert_top_to_limit(sql):
    try:
        tokens = sql.upper().split()
        is_top_used = False

        for i, token in enumerate(tokens):
            if token == 'TOP':
                is_top_used = True
                if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    limit_value = tokens[i + 1]
                    # Remove TOP and insert LIMIT and value at the end
                    del tokens[i:i + 2]
                    tokens.insert(len(tokens), 'LIMIT')
                    tokens.insert(len(tokens), limit_value)
                    break  # Exit loop after successful conversion
                else:
                    raise ValueError("TOP operator should be followed by a number")

        return ' '.join(tokens) if is_top_used else sql
    except Exception as e:
        err = f"An error occurred while converting Top to Limit in SQL Query: {e}"
        return err


# In[ ]:


def process_tablename(sql, table_name):
    try:
        x = sql.upper()
        query = x.replace(table_name.upper(), table_name)
        return query
    except Exception as e:
        err = f"An error occurred while processing table name in SQL query: {e}"
        return err


# In[160]:


def get_conversational_chain_quant():
        prompt_template = """
        
            Your task is to generate SQL queries for analyzing user reviews stored in a table named Sentiment_Data in Microsoft SQL Server Management Studio (SSMS). 
    Each row in the table represents a user review and contains the following columns:

    Review: The text of the user review.
    Data_Source: The source from which the review was collected.
    Geography: The country or region from which the review was given.
    Title: The title of the review.
    Review_Date: The date when the review was posted.
    Product: The corresponding product for the review, which can be "Windows 11 (Preinstall)" or "Windows 10".
    Product_Family: The specific model or type of the product like Asus Rog Zephyrus G16 16,Hp Omen Transcend 14,Lenovo Legion Pro 5I 16,Asus Rog Strix Scar 18,Asus Vivobook Pro 16,Asus Rog Zephyrus G14 14,Lenovo Slim 7I 16,Asus Zenbook 14,Dell G16 16,Asus Rog Strix 18,Razer Blade 16,Msi Prestige 16,Hp Laptop 14,Dell G15 15,Hp Spectre X360 16,Hp Omen 14,Razer Blade 14,Hp Spectre X360 14,Hp Pavilion 14,Lenovo V15 15,Hp G10 15,Lenovo Thinkpad E16 16,Hp G9 15,Lenovo V14 14,Lenovo Ideapad 1 14,Lenovo Thinkpad P17 17,Hp Laptop 17,Samsung Galaxy Book3 360 15,Lenovo Ideapad Slim 3 15,Acer Aspire 5 15,Dell Alienware X16 16,Asus Rog Strix G17 17,Asus Tuf Dash 15,Acer Aspire Vero 14,Microsoft Surface Laptop 4 15,Lenovo Ideapad 1 15,Hp Laptop 15,Lenovo Ideapad 3 15,Microsoft Sorface Laptop Go 3 12,Asus Zenbook Duo 14,Hp Stream 14,Asus Laptop 15,Hp Pavilion 15,Dell Inspiron 3000 15,Lenovo Slim 7 Pro X 14,Lg Gram 17,Acer Swift Go 14,Msi Stealth 16,Msi Titan 18,Asus Tuf Gaming F15 15,Dell Alienware M18 18,Asus Vivobook 15,Dell Precision 3000 15,Hp Envy X360 15,Asus Zenbook 14X 14,Hp Omen Transcend 16,Hp Envy 17T 17,Asus Rog Strix G16 16,Hp Envy 16,Hp Victus 15,Lenovo Ideapad 14,Microsoft Surface Laptop 15,Lenovo Thinkpad Yoga L13 13,Asus Rog Strix G15 15,Asus Zenbook Pro 14X 14,Razer Blade Gaming 15,Hp Pavilion 17,Asus Rog Strix G17  17,Asus Tuf Gaming A15 15,Msi Gaming Cyborg 15,Acer Predator Helios 16,Asus Tuf Gaming A17 17,Acer Nitro 5 15,Asus Vivobook Flip 16,Lenovo Thinkpad P14S 14,Dell Xps 13,Asus Rog Flow Z13 13,Msi Thin Gaming Gf63 15,Dell Inspiron 15,Microsoft Surface Laptop Go 12,Lenovo Ideapad 3I 15,Lenovo Laptop 15,Lenovo Legion Pro 7 16,Asus Rog Zephyrus M16 16,Lenovo Yoga 7I 14,Samsung Galaxy Book4 Pro 14,Hp Envy X360 16,Asus Zenbook Pro 14,Lenovo Thinkbook G6 16,Lenovo Thinkpad X1 Carbon 14,Samsung Galaxy Book2 Pro 360 13,Microsoft Surface Laptop Studio 14,Lenovo Thinkpad Yoga X1 14,Samsung Galaxy Book4 Pro 16,Microsoft Sorface Laptop Go 2 12,Dell Inspiron 14,Hp Envy X360 14,Microsoft Surface Go 13,Hp Omen  15,Dell Inspiron 3000  15,Lenovo Ideapad  15,Lenovo Slimpro 9I 14,Dell Inspiron 16,Hp Omen Gaming Laptop,Asus Vivobook 16,Hp Envy 2,Hp Envy  17,Asus Rog Flow 16,Asus Tuf Gaming A16 16,Asus Rog Gaming 14,Lenovo Loq 15,Lenovo Yoga 7 16,Lenovo Slim Pro 14,Lenovo Yoga 6 13,Lenovo Yoga 7I 16,Lenovo Yoga Book 9,Lenovo Ideapad 15,Samsung Galaxy Book 3 360 15,Microsoft Surface Laptop 5 13,Microsoft Surface Pro 9 13,Microsoft Surface Laptop 5 15,Samsung Galaxy Book 2 Pro 360 15,Microsoft Surface Pro 7 Plus 12,Microsoft Surface Laptop 4 13,Lenovo Ideapad 1  14,Alienware M16 16,Dell Xps 15,Hp Omen 16,Asus Vivobook 14,Lenovo Yoga 9I 14,Samsung Galaxy Book3 Pro 16,Samsung Galaxy Book 3 Pro 360 16,Acer Aspire 3 15,Hp Pavilion X360 14,Microsoft Surface Studio 2 14,Hp Stream 11,Asus Vivobook S 15,Lenovo Ideapad 16,Asus Vivobook 17,Hp Laptop  17,Asus Rog Zephyrus 16,Samsung Galaxy Book3 Ultra 16,Dell Inspiron 3000 14,Acer Predator Helios 18,Lenovo Thinkpad X1 14,Acer Nitro 17,Microsoft Surface Go Laptop 3 12,Msi Stealth 14,Asus Proart Studiobook 16,Asus Rog Zephyrus 14,Asus Rog 16,Lenovo Gaming Laptop,Samsung Galaxy Book 2 Pro 13,Microsoft Surface Laptop Go 2 12,Asus Vivobook  16,Hp Notebook 15,Hp Laptop  15,Dell Inspiron 5000  16,Microsoft Surface Go 2 10,Dell Inspiron 7000 14 Plus,Asus Tuf A16 16,Lenovo Legion 5  14,Asus Laptop 11,Lenovo Legion Slim 5 16,Msi Bravo 15,Acer Predator Helios Neo 16,Lenovo Legion 16,Samsung Galaxy Tab S7 Fe 12,Acer Aspire 1 15,Asus Tuf Gaming A 17,Dell Alienware  16,Samsung Galaxy Book 3 360 13,Lenovo Ideapad 3I 14,Asus Br1100 11,Lenovo Ideapad 3 14,Acer Aspire Vero 15,Lenovo Yoga 9I 13,Microsoft Surface Laptop Studio2 13,Asus Zenbook Pro Duo 15,Lenovo Wei 9 Pro  16,Hp Envy X360  15,Asus Rog Flow X13,Lenovo Wei 7 Pro  14,Asus Tuf Gaming 15,Hp Spectre X360 13,Microsoft Surface Go 3 10,Samsung Galaxy Ultra 16,Asus Vivobook Pro 15,Asus Vivobook Pro 16X  16,Acer Predator Helios 300 15,Asus Rog Strix G 17,Lenovo Ideapad Flex 5 14,Asus L210 11,Lenovo Ideapad Gaming 3 15,Asus Rog Strix G 16,Lenovo Ideapad 1I 14,Hp Envy 17,Lenovo Ideapad 3 17,Asus Rog Zephyrus G14  14,Lenovo Flex 5 14,Microsoft Surface Pro 8 13,Acer Swift Edge 16,Asus Rog Flow X13 13,Alienware M18 18,Asus Tuf Gaming A15  15,Lenovo Legion Pro 7I 16,Asus Vivobook Go 15,Acer Swift 3 14,Asus Rog Strix 17,Msi Cyborg 15,Hp Stream  14,Asus Vivobook 11,Dell G15 5000 15,Samusng Galaxy Book 3 Pro 14,Dell Inspiron 7000 2-In-1 16,Msi Gaming Laptop 16,Msi Stealth 17,Msi Gaming 15,Dell Alienware M16 16,Lenovo Ideapad Flex 5  14,Asus Tuf Gaming F15,Lenovo Ideapad Flex 5 16,Lenovo Laptop 14,Msi Gamingthin 15,Asus Rog Zephyrus Duo 16,Acer Swift X 14,Msi Gf63 Thin 15,Microsoft Surface Pro 7 12,Lenovo Flex 11,Lenovo Wei 5 Pro  15,Asus Vivobook Go 14,Asus Vivobook  14,Msi Katana 15,Hp Laptop 17T 17,Asus Vivobook Go 12,Asus L510 15,Microsoft Surface Go 10,Dell Insprion 3000 11,Asus Zenbook Pro 15,Dell Alienware M15 R7 15,Hp Envy Desktop,Asus Zenbook Duo 15,Lg Gram 16,Asus Rog Strix Scar 15,Asus Tuf Gaming F17 17,Msi Sword 15,Acer Nitro 5 17,Dell Xps 17,Lenovo Legion 5 15,Razer Gaming Laptop 18,Asus Vivobook Pro  16,Dell Inspiron 5000 14,Samsung Galaxy Book3 15,Hp Eliteboook G7 14,Lenovo Yoga C740 15,Lenovo Yoga Slim 7 16,Lenovo Slim 7 Pro 14,Lg Gram 14,Hp Spectre 17,Asus Zenbook Pro Duo 14,Asus Rog Zephyrus G16  16,Hp Victus 16,Msi Summit Flip 14,Asus Zenbook Pro 17,Acer Nitro 16,Lenovo Thinkpad T16 16,Asus Zenbook S 13,Msi Raider 17,Asus Rog Strix Scar 16,Msi Vector 16,Razer Blade 15,Dell Alienware M17 R5 17,Dell Xps Plus 13,Lenovo Gaming Desktop,Asus Rog Flow X16 16,Hp Stream  17,Asus Zenbook Flip 14,Razer Blade 18,Msi Thin Gf63 15,Asus Vivobook Pro 16X 16,Asus Vivobook Flip 14,Asus Vivobook Pro X 16,Dell Inspiron 5620 Desktop,Lenovo 300W 3 11,Asus Zenbook 15,Acer Aspire 3 14,Samsung Galaxy Book2 Pro 15,Asus Vivobook Pro 14,Asus Vivobook Pro 15X,Samsung Galaxy Book 2 Pro 15,Msi Gv15 15,Asus Rog Strix G15  15,Lenovo Ideapad 3  15,Lg Gram 15,Samsung Galaxy Book3 16,Lg Gram 17Z95P,Lenovo Ideapad Flex 5 15,Asus Rog Strix G16   16,Lenovo Thinkpad Yoga 11E,Msi Creator M 16,Lenovo Ideapad 5I Pro 16,Asus E410 14,Dell Inspiron 15 3000,Samsung Galaxy Book3 Pro 15,Dell Inspiron 3501 15,Asus Vivobook 13,Microsoft Surface Laptop 3 13,Lenovo Ideapad 5I 15,Dell Alienware X15 R2 15,Asus Vivobook Pro 15X 15,Samsung Galaxy Book Go 14.
    Sentiment: The sentiment of the review, which can be 'Positive', 'Neutral', or 'Negative'.
    Aspect: The aspect or feature of the product discussed in the review like "Audio-Microphone","Software","Performance","Storage/Memory","Keyboard","Browser","Connectivity","Hardware","Display","Graphics","Battery","Gaming","Design","Ports","Price","Camera","Customer-Service","Touchpad","Account","Generic".
    Keywords: Keywords mentioned in the review.
    Review_Count: Always 1 for each review.
    Sentiment_Score: A score based on the sentiment, which can be 1, 0, or -1.
    Your queries should allow analysis based on various criteria, including product family, sentiment, aspect, geography, etc. Always use the LIKE keyword for pattern matching in SQL queries.

    For example, to retrieve all rows and columns for a specific product family (e.g., "Asus Rog Zephyrus G16 16"), your query should look like:

        SELECT *
        FROM Sentiment_Data
        WHERE Product_Family LIKE '%Asus Rog Zephyrus G16 16%'
    Ensure that your queries follow SSMS syntax and can be executed directly in Microsoft SQL Server Management Studio.
    IMPORTANT : Give only the SQL Query if i pass the Query directly it should excute without any error.        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment="Thruxton_R",
            api_version='2023-12-01-preview',
            temperature = 0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

def query_reviews(user_question, vector_store_path="faiss_index_CopilotSample"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Mv_Agusta")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_quant()
        docs = []
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        SQL_Query = response["output_text"]
        SQL_Query = convert_top_to_limit(SQL_Query)
        SQL_Query = process_tablename(SQL_Query,"Sentiment_Data")
        data = ps.sqldf(SQL_Query, globals())
        data_1 = data
        html_table = data.to_html(index=False)
        return data_1
    except Exception as e:
        err = f"An error occurred while generating response for quantitative review summarization: {e}"
        return err


# In[161]:


def get_txt_text(txt_file_path):
    with io.open(txt_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


# In[162]:


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
    chunks = text_splitter.split_text(text)
    return chunks


# In[163]:


def get_vector_store(chunks):
    embeddings = AzureOpenAIEmbeddings(azure_deployment="Mv_Agusta")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss-index")


# In[164]:


def get_conversational_chain_summary():
    prompt_template = """
    Your task is to analyze the reviews of Windows products and generate a summary of the pros and cons for each product based on the provided dataset.Provide an overall summary. focus only on listing the pros and cons. 
    Use the format below for your response:

    Pros and Cons of [Product Name]:

    Pros:

    [Aspect]: [Brief summary of positive feedback regarding this aspect. Include specific examples if available.]
    [Aspect]: [Brief summary of positive feedback regarding this aspect. Include specific examples if available.]
    [Aspect]: [Brief summary of positive feedback regarding this aspect. Include specific examples if available.]
    [Aspect]: [Brief summary of positive feedback regarding this aspect. Include specific examples if available.]
    [Aspect]: [Brief summary of positive feedback regarding this aspect. Include specific examples if available.]
    Cons:

    [Aspect]: [Brief summary of negative feedback regarding this aspect. Include specific examples if available.]
    [Aspect]: [Brief summary of negative feedback regarding this aspect. Include specific examples if available.]
    [Aspect]: [Brief summary of negative feedback regarding this aspect. Include specific examples if available.]
    [Aspect]: [Brief summary of negative feedback regarding this aspect. Include specific examples if available.]
    [Aspect]: [Brief summary of negative feedback regarding this aspect. Include specific examples if available.]
    
    [Overall Summary]: [Brief summary of overall feedback regarding all aspect.]
    The dataset includes the following columns:

    Review: Review of the Windows product.
    Data_Source: Source of the review, containing different retailers.
    Geography: Country or region of the review.
    Title: Title of the review.
    Review_Date: Date the review was posted.
    Product: Product the review corresponds to, with values: "Windows 11 (Preinstall)", "Windows 10".
    Product_Family: Version or type of the corresponding product.
    Sentiment: Sentiment of the review, with values: 'Positive', 'Neutral', 'Negative'.
    Aspect: Aspect or feature of the product discussed in the review, with values: "Audio-Microphone", "Software", "Performance", "Storage/Memory", "Keyboard", "Browser", "Connectivity", "Hardware", "Display", "Graphics", "Battery", "Gaming", "Design", "Ports", "Price", "Camera", "Customer-Service", "Touchpad", "Account", "Generic".
    Keywords: Keywords mentioned in the review.
    Review_Count: Will be 1 for each review or row.
    Sentiment_Score: Will be 1, 0, or -1 based on the sentiment.
    Please ensure that the response is based on the analysis of the provided dataset, summarizing both positive and negative aspects of each product. 
     
        
    Context:\n {context}?\n
    Question: \n{question}\n
 
    Answer:
    """
    model = AzureChatOpenAI(
    azure_deployment="Thruxton_R",
    api_version='2023-12-01-preview',temperature = 0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def query_to_embedding_summarize(user_question, txt_file_path):
    text = get_txt_text(txt_file_path)
    chunks = get_text_chunks(text)
    get_vector_store(chunks)
    embeddings = AzureOpenAIEmbeddings(azure_deployment="Mv_Agusta")
    
    # Load the vector store with the embeddings model
    new_db = FAISS.load_local("faiss-index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain_summary()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response['output_text']


# In[165]:


def get_product_image(device_name):
    device_details = pd.read_csv('Device_Details.csv')
    image_url = device_details.loc[device_details['Device_Name'] == device_name, 'Image_URL'].values
    if len(image_url) > 0:
        return image_url[0]
    else:
        return None


# In[166]:


from PIL import Image
import requests
from io import BytesIO
def load_and_resize_image(url, new_height):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        aspect_ratio = img.width / img.height
        new_width = int(aspect_ratio * new_height)
        resized_img = img.resize((new_width, new_height))
        image = st.image(resized_img)
        return image
    except:
        st.write("Image not available for this product.")


# In[167]:


st.title("Device Comparison")

device_name_1 = st.text_input("Enter the first device name : ")
device_name_2 = st.text_input("Enter the second device name : ")

col1, col2 = st.columns(2)

# Container for the first device
with col1:
    with st.container():
        if device_name_1:
            image_url_1 = get_product_image(device_name_1)
            if image_url_1:
                image1 = load_and_resize_image(image_url_1, 150)
            else:
                st.write("Image not available for this product.")
            data_1 = query_reviews("Give me all the reviews of " + device_name_1)
            data_1.to_csv(device_name_1 + "_Reviews.txt", sep='\t')
            a = device_name_1 + "_Reviews.txt"
            summary_1 = query_to_embedding_summarize("Give me the pros and cons of " + device_name_1, a)
            st.subheader(device_name_1)
            st.write(summary_1)

# Container for the second device
with col2:
    with st.container():
        if device_name_2:
            image_url_2 = get_product_image(device_name_2)
            if image_url_2:
                image2 = load_and_resize_image(image_url_2, 150)
            else:
                st.write("Image not available for this product.")
            data_2 = query_reviews("Give me all the reviews of " + device_name_2)
            data_2.to_csv(device_name_2 + "_Reviews.txt", sep='\t')
            a = device_name_2 + "_Reviews.txt"
            summary_2 = query_to_embedding_summarize("Give me the pros and cons of " + device_name_2, a)
            st.subheader(device_name_2)
            st.write(summary_2)


# In[120]:





# In[121]:





# In[ ]:




