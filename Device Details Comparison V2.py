#This notebook generates summary for Surface devices along with images, specs and sales info. This notebook also has the functionality to move to dummy comparison view from current device specs and summarization.

#Import Required Libraries
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
# from langchain.llms import AzureOpenAI
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
# from pandasai.llm import AzureOpenAI
import matplotlib.pyplot as plt
import os
import time
from PIL import Image
import requests
from io import BytesIO
import base64
import pandasql as ps
from openai import AzureOpenAI
from QuantitativeSummaryFinal import Sentiment_Score_Derivation, get_final_df,custom_color_gradient, query_detailed, get_conversational_chain_detailed, query_detailed_summary, get_conversational_chain_detailed_summary, query_quant, get_conversational_chain_quant, process_tablename
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["AZURE_OPENAI_API_KEY"] = "672370cd6ca440f2a0327351d4f4d2bf"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://hulk-openai.openai.azure.com/"


client = AzureOpenAI(
    api_key=os.getenv("672370cd6ca440f2a0327351d4f4d2bf"),  
    api_version="2024-02-01",
    azure_endpoint = os.getenv("https://hulk-openai.openai.azure.com/")
    )
    
deployment_name='SurfaceGenAI'

Sentiment_Data  = pd.read_csv("Windows_Data_116K.csv")

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
    IMPORTANT : Give only the SQL Query if i pass the Query directly it should excute without error.
    User Question :
    
    """
def query_verbatims(review):
    SQL_Query_Temp = client.completions.create(model=deployment_name, prompt=start_phrase_verbatim+review, max_tokens=1000,temperature=0)
    SQL_Query = SQL_Query_Temp.choices[0].text
    data_verbatims = ps.sqldf(SQL_Query,globals())
    return data_verbatims

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

def get_txt_text(txt_file_path):
    with io.open(txt_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = AzureOpenAIEmbeddings(azure_deployment="Mv_Agusta")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss-index")
    



context = """
    1. Your Job is to convert the user question to SQL Query (Follow Microsoft SQL server SSMS syntax.). You have to give the query so that it can be used on Microsoft SQL server SSMS.You have to only return query as a result.
    2. There is only one table with table name RCR_Sales_Data where each row has. The table has 20 columns, they are:
        Month: Contains dates for the records
        Country: From where the sales has happened. It contains following values: 'Turkey','India','Brazil','Germany','Philippines','France','Netherlands','Spain','United Arab Emirates','Czech Republic','Norway','Belgium','Finland','Canada','Mexico','Russia','Austria','Poland','United States','Switzerland','Italy','Colombia','Japan','Chile','Sweden','Vietnam','Saudi Arabia','South Africa','Peru','Indonesia','Taiwan','Thailand','Ireland','Korea','Hong Kong SAR','Malaysia','Denmark','New Zealand','China' and 'Australia'.
        Geography: From which Country or Region the review was given. It contains following values: 'Unknown', 'Brazil', 'Australia', 'Canada', 'China', 'Germany','France'.
        OEMGROUP: OEM or Manufacturer of the Device. It contains following values: 'Lenovo','Acer','Asus','HP','All Other OEMs', 'Microsoft' and 'Samsung'
        SUBFORMFACTOR: Formfactor of the device. It contains following values: 'Ultraslim Notebook'.
        GAMINGPRODUCTS: Flag whether Device is a gaming device or not. It contains following values: 'GAMING', 'NO GAMING' and 'N.A.'.
        SCREEN_SIZE_INCHES: Screen Size of the Device.
        PRICE_BRAND_USD_3: Band of the price at which the device is selling. It contains following values: '0-300', '300-500', '500-800' and '800+.
        OS_VERSION: Operating System version intall on the device. It contains following values: 'Windows 11', 'Chrome', 'Mac OS'.
        Operating_System_Summary: Operating System installed on the device. This is at uber level. It contains following values: 'Windows', 'Google OS', 'Apple OS'.
        Sales_Units: Number of Devices sold for that device in a prticular month and country.
        Sales_Value: Revenue Generated by the devices sold.
        Series: Family of the device such as IdeaPad 1, HP Laptop 15 etc.
        Specs_Combination: Its contains the combination of Series, Processor, RAM , Storage and Screen Size. For Example: SURFACE LAPTOP GO | Ci5 | 8 GB | 256.0 SSD | 12" .
    3.  When Asked for Price Range you have to use ASP Column to get minimum and Maxium value. Do not consider Negative Values. Also Consider Sales Units it shouldn't be 0.
        Exaple Query:
            SELECT MIN(ASP) AS Lowest_Value, MAX(ASP) AS Highest_Value
            FROM RCR_Sales_Data
            WHERE
            Series = 'Device Name'
            AND ASP >= 0
            AND Sales_Units <> 0;
    4. Total Sales_Units Should Always be in Thousands. 
        Example Query:
            SELECT (SUM(Sales_Units) / 1000) AS "TOTAL SALES UNITS"
            FROM RCR_Sales_Data
            WHERE
            SERIES LIKE '%SURFACE LAPTOP GO%';
    5. Average Selling Price (ASP): It is calculated by sum of SUM(Sales_Value)/SUM(Sales_Units)
    6. Total Sales Units across countries or across regions is sum of sales_units for those country. It should be in thousand of million hence add "K" or "M" after the number.
        Example to calculate sales units across country:
            SELECT Country, (SUM(Sales_Units) / 1000) AS "Sales_Units(In Thousands)"
            FROM RCR_Sales_Data
            GROUP BY Country
            ORDER BY Sales_Units DESC
    7. Total Sales Units across column "X" or across regions is sum of sales_units for those country. It should be in thousand of million hence add "K" or "M" after the number.
        Example to calculate sales units across country:
            SELECT "X", (SUM(Sales_Units) / 1000) AS "Sales_Units(In Thousands)"
            FROM RCR_Sales_Data
            GROUP BY "X"
            ORDER BY Sales_Units DESC
    8. If asked about the highest selling Specs Combination. 
        Example Query:
            SELECT Specs_Combination, (SUM(Sales_Units) / 1000) AS "TOTAL SALES UNITS"
            FROM RCR_Sales_Data
            WHERE SERIES LIKE '%Macbook AIR%'
            AND SALES_UNITS <> 0
            GROUP BY Specs_Combination
            ORDER BY "TOTAL SALES UNITS" DESC
            LIMIT 1;
    9. If asked about similar compete devices.
    Example Query:
            WITH SurfaceLaptopGoASP AS (
                SELECT
                    'Surface Laptop Go' AS Series,
                    SUM(Sales_Value) / SUM(Sales_Units) AS ASP
                FROM
                    RCR_Sales_Data
                WHERE
                    Series LIKE '%Surface Laptop Go%'
            ),
            CompetitorASP AS (
                SELECT
                    Series,
                    SUM(Sales_Value) / SUM(Sales_Units) AS ASP
                FROM
                    RCR_Sales_Data
                WHERE
                    Operating_System_Summary IN ('Apple OS', 'Google OS')
                GROUP BY
                    Series
            )
            SELECT
                C.Series,
                C.ASP AS CompetitorASP
            FROM
                CompetitorASP C
            JOIN
                SurfaceLaptopGoASP S
            ON
                ABS(C.ASP - S.ASP) <= 200;
    10. If asked about dates or year SUBSTR() function instead of Year() or Month()
    11. Convert numerical outputs to float upto 2 decimal point.
    12. Always include ORDER BY clause to sort the table based on the aggregate value calculated in the query.
    13. Always use 'LIKE' operator whenever they mention about any Country, Series. Use 'LIMIT' operator instead of TOP operator.Do not use TOP OPERATOR. Follow syntax that can be used with pandasql.
    14. If you are using any field in the aggregate function in select statement, make sure you add them in GROUP BY Clause.
    15. Make sure to Give the result as the query so that it can be used on Microsoft SQL server SSMS.
    16. Always use LIKE function instead of = Symbol while generating SQL Query
    17. Important: User can ask question about any categories including Country, OEMGROUP,OS_VERSION etc etc. Hence, include the in SQL Query if someone ask it.
    18. Important: Use the correct column names listed above. There should not be Case Sensitivity issue. 
    19. Important: The values in OPERATING_SYSTEM_SUMMARY are ('Apple OS', 'Google OS') not ('APPLE OS', 'GOOGLE OS'). So use exact values. Not everything should be capital letters.
    20. Important: You Response should directly starts from SQL query nothing else."""
# Initialize an empty context
detail_summary_template_prompt = """Provide a detailed consumer review summary for the [device_name] with aspect-wise net sentiment. Please mention what consumers like and dislike about the device, focusing on Performance, Design, and Display.
Overall, the [device_name] is well-received by consumers, with its performance, design, and display being the standout features. However, there are some minor criticisms regarding performance issues and display brightness.
Aspect: Performance
Net Sentiment: +70%
Likes: Users appreciate the smooth performance of the device, noting its fast processing speed and ability to handle multitasking with ease.
Dislikes: Some users have reported occasional lag or slowdowns, especially when running demanding applications or games.

Aspect: Design
Net Sentiment: +85%
Likes: Consumers love the sleek and compact design of the device, praising its lightweight build and premium look and feel.
Dislikes: A few users find the design too simplistic and wish for more color options or customizable features.

Aspect: Display
Net Sentiment: +75%
Likes: Users are impressed with the vibrant and sharp display of the device, noting its accurate colors and wide viewing angles.
Dislikes: Some users feel that the display could be brighter, especially when using the device outdoors or in brightly lit environments.

Mention the need for Improvement as well in detail"""
def generate_SQL_Query(user_question):
    global context
    # Append the new question to the context
    full_prompt = context + "\nQuestion:\n" + user_question + "\nAnswer:"
    
    # Send the query to Azure OpenAI
    response = client.completions.create(
        model=deployment_name,
        prompt=full_prompt,
        max_tokens=500,
        temperature=0
    )
    
    # Extract the generated SQL query
    sql_query = response.choices[0].text.strip()
    
    # Update context with the latest interaction
    context += "\nQuestion:\n" + user_question + "\nAnswer:\n" + sql_query
    
    return sql_query

#Converting Top Operator to Limit Operator as pandasql doesn't support Top
def convert_top_to_limit(sql):
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


def process_tablename(sql, table_name):
    x = sql.upper()
    query = x.replace(table_name.upper(), table_name)
    return query

RCR_Sales_Data = pd.read_csv('RCR Sales Data Sample V3.csv')


def get_sales_units(device_name):
    question = "Totals Sales Units for " + device_name
    a = generate_SQL_Query(question)
    SQL_Query = convert_top_to_limit(a)
    SQL_Query = process_tablename(SQL_Query,"RCR_Sales_Data")
    data = ps.sqldf(SQL_Query, globals())
    col_name = data.columns[0]
    total_sales = data[col_name][0]
    if not total_sales:
        total_sales = 0
    total_sales = str(round(total_sales,2)) + "K"
    return total_sales


def get_ASP(device_name):
    question = "What's ASP for " + device_name
    a = generate_SQL_Query(question)
    SQL_Query = convert_top_to_limit(a)
    SQL_Query = process_tablename(SQL_Query,"RCR_Sales_Data")
    data = ps.sqldf(SQL_Query, globals())
    col_name = data.columns[0]
    asp = data[col_name][0]
    if not asp:
        asp = 0
    asp = "$" + str(int(round(asp,0)))
    return asp

def get_highest_selling_specs(device_name):
    question = "What's highest selling Specs Combination for " + device_name
    a = generate_SQL_Query(question)
    SQL_Query = convert_top_to_limit(a)
    SQL_Query = process_tablename(SQL_Query,"RCR_Sales_Data")
    data = ps.sqldf(SQL_Query, globals())
    col_name1 = data.columns[0]
    col_name2 = data.columns[1]
    specs = data[col_name1][0]
    sales_unit = data[col_name2][0]
    if not sales_unit:
        sales_unit = 0
    sales_unit = str(round(sales_unit,2)) + "K"
    return specs,sales_unit

def compete_device(device_name):
    question = "What are the compete device for " + device_name
    a = generate_SQL_Query(question)
    SQL_Query = convert_top_to_limit(a)
    SQL_Query = process_tablename(SQL_Query,"RCR_Sales_Data")
    SQL_Query = SQL_Query.replace('APPLE','Apple')
    SQL_Query = SQL_Query.replace('GOOGLE','Google')
    data = ps.sqldf(SQL_Query, globals())
    col_name1 = data.columns[0]
    devices = list(data[col_name1])
    return devices
    
def get_detailed_summary(user_imput):
    response = client.completions.create(
        model=deployment_name,
        prompt=detail_summary_template_prompt+user_imput,
        max_tokens=1000,
        temperature=0.2
    )
    output = response.choices[0].text
    return output

def get_device_image(user_input):
    df = pd.read_csv('Device Images.csv')
    for i in df['Device Name']:
        if str.lower(i) in str.lower(user_input):
            dev = i
    link = df[df['Device Name']==dev]['Link'].values[0]
    return (dev, link)

def get_comp_device_image(user_input):
    df = pd.read_csv('Device Images.csv')
    dev = None
    for i in df['Device Name']:
        if str.lower(i) in str.lower(user_input):
            dev = i
            break  # Exit the loop once a match is found
    if dev is None:
        return None, None  # Return None if no matching device is found
    link = df[df['Device Name']==dev]['Link'].values[0]  # Using .values[0] to get the link
    return dev, link
    
def get_detailed_summary(device_name):
    if device_name:
        data = query_quant("Summarize the reviews of "+ device_name, [])
        total_reviews = data.loc[data['ASPECT'] == 'TOTAL', 'REVIEW_COUNT'].iloc[0]
        data['REVIEW_PERCENTAGE'] = data['REVIEW_COUNT'] / total_reviews * 100
        dataframe_as_dict = data.to_dict(orient='records')
        data_new = data
        data_new = data_new.dropna(subset=['ASPECT_SENTIMENT'])
        data_new = data_new[~data_new["ASPECT"].isin(["Generic", "Account", "Customer-Service", "Browser"])]
        vmin = data_new['ASPECT_SENTIMENT'].min()
        vmax = data_new['ASPECT_SENTIMENT'].max()
        styled_df = data_new.style.applymap(lambda x: custom_color_gradient(x, vmin, vmax), subset=['ASPECT_SENTIMENT'])
        data_filtered = data_new[data_new['ASPECT'] != 'TOTAL']
        data_sorted = data_filtered.sort_values(by='REVIEW_COUNT', ascending=False)
        top_four_aspects = data_sorted.head(4)
        aspects_list = top_four_aspects['ASPECT'].to_list()
        formatted_aspects = ', '.join(f"'{aspect}'" for aspect in aspects_list)
        key_df = get_final_df(aspects_list, device_name)
        b =  key_df.to_dict(orient='records')
        su = query_detailed_summary("Summarize reviews of" + device_name + "for " +  formatted_aspects +  "Aspects which have following "+str(dataframe_as_dict)+ str(b) + "Reviews: ",[])
    return su

def generate_device_details(device_input):
    device_name, img_link = get_device_image(device_input)
    total_sales = get_sales_units(device_name)
    asp = get_ASP(device_name)
    high_specs, sale = get_highest_selling_specs(device_name)
    summ = get_detailed_summary(device_input)
    return device_name, img_link, total_sales, asp, high_specs, sale, summ

def load_and_resize_image(url, new_height):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    aspect_ratio = img.width / img.height
    new_width = int(aspect_ratio * new_height)
    resized_img = img.resize((new_width, new_height))
    image = st.image(resized_img)
    return image
    
def main():
    try:
    # Chat history state management
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        if 'selected_device_comparison' not in st.session_state:
            st.session_state['selected_device_comparison'] = None
        # Create a container for logos and title with horizontal layout
        col1, col2, col3 = st.columns([1, 2, 1])
      
        # Display logo on the left
        with col1:
            st.image("microsoft_logo.png", width=50)  # Adjust width as needed

        # Display title in the center
        with col2:
            st.header("Consumer Reviews Synthesizer")

        # Display logo on the right
        with col3:
            st.image("copilot_logo.svg", width=50)  # Align the logo to the right
      
        # User input section
        user_input = st.text_input("Enter your text:", placeholder="What would you like to process?")
        if user_input:
            inp = user_input
            if not st.session_state['chat_history']:
                device_name, img_link, total_sales, asp, high_specs, sale, summ = generate_device_details(inp)
                st.session_state['chat_history'].append((inp,device_name,img_link,total_sales,asp,high_specs,sale,summ))
            elif inp != st.session_state['chat_history'][-1][0]:
                device_name, img_link, total_sales, asp, high_specs, sale, summ = generate_device_details(inp)
                st.session_state['chat_history'].append((inp,device_name,img_link,total_sales,asp,high_specs,sale,summ))
            else:
                device_name = st.session_state['chat_history'][-1][1]
                img_link = st.session_state['chat_history'][-1][2]
                total_sales = st.session_state['chat_history'][-1][3]
                asp = st.session_state['chat_history'][-1][4]
                high_specs = st.session_state['chat_history'][-1][5]
                sale = st.session_state['chat_history'][-1][6]
                summ = st.session_state['chat_history'][-1][7]
            col4, col5 = st.columns([1, 3])
            with col4:
                st.image(img_link,width = 150)
                st.write(device_name)
            with col5:
                st.write(f"Total Devices Sold: **{total_sales}**" )
                st.write(f"Average Selling Price: **{asp}**")
                st.write(f"Highest Selling Specs: **{high_specs}** - **{sale}**")
            st.write(r"$\textsf{\Large Detailed Summary}$")
            st.write(summ)
            inp = False
            
            ######### Adding checkbox to trigger comparison view

            # Define checkbox labels and initial states
            device1_label = "Surface Pro"
            device2_label = "Surface Go"
            device3_label = "Surface Studio"
            device4_label = "Inspiron 15"
            checkbox1_state = False
            checkbox2_state = False
            checkbox3_state = False
            checkbox4_state = False
            # Create checkboxes
            checkbox1_state = st.checkbox(device1_label)
            checkbox2_state = st.checkbox(device2_label)
            checkbox3_state = st.checkbox(device3_label)
            checkbox4_state = st.checkbox(device4_label)
            
            # Display text based on checkbox selection
            if checkbox1_state:
                st.session_state['selected_device_comparison'] = device1_label
                st.write(f"You selected {device1_label}.")
            elif checkbox2_state:
                st.session_state['selected_device_comparison'] = device2_label
                st.write(f"You selected {device2_label}.")
            elif checkbox3_state:
                st.session_state['selected_device_comparison'] = device3_label
                st.write(f"You selected {device3_label}.")
            elif checkbox4_state:
                st.session_state['selected_device_comparison'] = device4_label
                st.write(f"You selected {device4_label}.")
            else:
                st.session_state['selected_device_comparison'] = None
            
            if st.session_state['selected_device_comparison']:
                st.write(f"Inside Comparison View for {st.session_state['selected_device_comparison']}")
                device2_name, img2_link, total2_sales, asp2, high2_specs, sale2, summ2 = generate_device_details(st.session_state['selected_device_comparison'])
                st.write(f"Device Name: {device_name} vs {device2_name}")
                st.write(f"Total Devices Solf: {total_sales} vs {total2_sales}")
                st.write(f"Avg Selling Price: {asp} vs {asp2}")
                st.write(f"Highest Selling Specs: {high_specs} vs {high2_specs}")
                st.write(f"\n\n{device_name} Summary:\n{summ}")
                st.write(f"\n\n{device2_name} Summary:\n{summ2}")
                
       
                                     
    except Exception as e:
        err = f"An error occurred while calling the final function: {e}"
        print(err)
        return err

if __name__ == "__main__":
    main()