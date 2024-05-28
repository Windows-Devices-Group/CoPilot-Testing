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
import base64
import pandasql as ps
from openai import AzureOpenAI
from PIL import Image
import requests
from io import BytesIO
import io
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
        Chassis Segment: It contains following values: 'SMB_Upper','Mainstream_Lower','SMB_Lower','Enterprise Fleet_Lower','Entry','Mainstream_Upper','Premium Mobility_Upper','Enterprise Fleet_Upper','Premium Mobility_Lower','Creation_Lower','UNDEFINED','Premium_Mobility_Upper','Enterprise Work Station','Unknown','Gaming_Musclebook','Entry_Gaming','Creation_Upper','Mainstrean_Lower'
        
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
            SQL = WITH DeviceNameASP AS (
                    SELECT
                        'Device Name' AS Series,
                        SUM(Sales_Value) / SUM(Sales_Units) AS ASP,
                        Chassis_Segment,
                        SUM(Sales_Units) AS Sales_Units
                    FROM
                        RCR_Sales_Data
                    WHERE
                        Series LIKE '%Device Name%'
                    GROUP BY
                        Chassis_Segment
                ),
                CompetitorASP AS (
                    SELECT
                        Series,
                        SUM(Sales_Value) / SUM(Sales_Units) AS ASP,
                        Chassis_Segment,
                        SUM(Sales_Units) AS Sales_Units
                    FROM
                        RCR_Sales_Data
                    WHERE
                        Operating_System_Summary IN ('Apple OS', 'Google OS','Windows OS')
                        AND SERIES NOT LIKE '%Device Name%'
                    GROUP BY
                        Series, Chassis_Segment
                ),
                RankedCompetitors AS (
                    SELECT
                        C.Series,
                        C.ASP,
                        C.Chassis_Segment,
                        C.Sales_Units,
                        ROW_NUMBER() OVER (PARTITION BY C.Chassis_Segment ORDER BY C.Sales_Units DESC) AS rank
                    FROM
                        CompetitorASP C
                    JOIN
                        DeviceNameASP S
                    ON
                        ABS(C.ASP - S.ASP) <= 100
                        AND C.Chassis_Segment = S.Chassis_Segment
                )
                SELECT
                    Series,
                    ASP AS CompetitorASP,
                    Sales_Units
                FROM
                    RankedCompetitors
                WHERE
                    rank <= 4;

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

interaction = ""

# Initialize an empty context

def generate_SQL_Query(user_question):
    global context, interaction
    # Append the new question to the context
    full_prompt = context + interaction + "\nQuestion:\n" + user_question + "\nAnswer:"
    
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
    interaction += "\nQuestion:\n" + user_question + "\nAnswer:\n" + sql_query
    
    return sql_query

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
    embeddings = AzureOpenAIEmbeddings(azure_deployment="Mv_Agusts")
    
    # Load the vector store with the embeddings model
    new_db = FAISS.load_local("faiss-index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain_summary()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response['output_text']

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
    question = None
    a = None
    question = "Totals Sales Units for " + device_name
    a = generate_SQL_Query(question)
    SQL_Query = convert_top_to_limit(a)
    SQL_Query = process_tablename(SQL_Query,"RCR_Sales_Data")
    data = ps.sqldf(SQL_Query, globals())
    col_name = data.columns[0]
    total_sales = data[col_name][0]
    total_sales = str(round(total_sales,2)) + "K"
    return total_sales


def get_ASP(device_name):
    question = None
    a = None
    question = "What's ASP for " + device_name
    a = generate_SQL_Query(question)
    SQL_Query = convert_top_to_limit(a)
    SQL_Query = process_tablename(SQL_Query,"RCR_Sales_Data")
    data = ps.sqldf(SQL_Query, globals())
    col_name = data.columns[0]
    asp = data[col_name][0]
    asp = "$" + str(int(round(asp,0)))
    return asp

def get_highest_selling_specs(device_name):
    question = None
    a = None
    question = "What's highest selling Specs Combination for " + device_name
    a = generate_SQL_Query(question)
    SQL_Query = convert_top_to_limit(a)
    SQL_Query = process_tablename(SQL_Query,"RCR_Sales_Data")
    data = ps.sqldf(SQL_Query, globals())
    col_name1 = data.columns[0]
    col_name2 = data.columns[1]
    specs = data[col_name1][0]
    sales_unit = data[col_name2][0]
    sales_unit = str(round(sales_unit,2)) + "K"
    return specs,sales_unit

def compete_device(device_name):
    question = None
    a = None
    question = "What are the compete device for " + device_name
    a = generate_SQL_Query(question)
    SQL_Query = convert_top_to_limit(a)
    SQL_Query = process_tablename(SQL_Query,"RCR_Sales_Data")
    SQL_Query = SQL_Query.replace('APPLE','Apple')
    SQL_Query = SQL_Query.replace('GOOGLE','Google')
    SQL_Query = SQL_Query.replace('WINDOWS','Windows')
    data = ps.sqldf(SQL_Query, globals())
    return data
    
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
    
def get_net_sentiment(device_name):
    a = query_quant(device_name,[])
    try:
        Net_Sentiment = float(a[a['ASPECT']=='TOTAL']['ASPECT_SENTIMENT'].values[0])
        aspects = a["ASPECT"].unique()
        if "Performance" in aspects:
            Performance_Sentiment = float(a[a['ASPECT']=='Performance']['ASPECT_SENTIMENT'].values[0])
        else:
            Performance_Sentiment = 0
        
        if "Design" in aspects:
            Design_Sentiment = float(a[a['ASPECT']=='Design']['ASPECT_SENTIMENT'].values[0])
        else:
            Design_Sentiment = 0
        
        if "Display" in aspects:
            Display_Sentiment = float(a[a['ASPECT']=='Display']['ASPECT_SENTIMENT'].values[0])
        else:
            Display_Sentiment = 0
        
        if "Battery" in aspects:
            Battery_Sentiment = float(a[a['ASPECT']=='Battery']['ASPECT_SENTIMENT'].values[0])
        else:
            Battery_Sentiment = 0
        
        if "Price" in aspects:
            Price_Sentiment = float(a[a['ASPECT']=='Price']['ASPECT_SENTIMENT'].values[0])
        else:
            Price_Sentiment = 0
        
        if "Software" in aspects:
            Software_Sentiment = float(a[a['ASPECT']=='Software']['ASPECT_SENTIMENT'].values[0])
        else:
            Software_Sentiment = 0
            
        aspect_sentiment = list((Performance_Sentiment, Design_Sentiment, Display_Sentiment, Battery_Sentiment, Price_Sentiment, Software_Sentiment))
                                 
    except:
        Net_Sentiment = None
        aspect_sentiment = None                     
    return Net_Sentiment, aspect_sentiment



def get_comp_device_details(user_input, df1):
    df = pd.read_csv('Device Images.csv')
    dev = None
    for i in df['Device Name']:
        if str.lower(i) in str.lower(user_input):
            dev = i
            break  # Exit the loop once a match is found
    
    if dev is None:
        return None, None, None, None, None  # Return None if no matching device is found
    
    link = df[df['Device Name'] == dev]['Link'].values[0]  # Using .values[0] to get the link
    df1['SERIES'] = df1['SERIES'].str.upper()
    dev = dev.upper()
    sales_data = df1[df1['SERIES'] == dev]
    if sales_data.empty:
        return dev, link, None, None, None  # Return dev and link, but None for sales and ASP if no matching SERIES is found
    
    try:
        sales = str(round(float(sales_data['SALES_UNITS'].values[0]) / 1000, 2)) + "K"
    except:
        sales = "NA"
    try:
        ASP = "$" + str(int(sales_data['COMPETITORASP'].values[0]))
    except:
        ASP = "NA"
    net_sentiment,aspect_sentiment = get_net_sentiment(dev)
    return dev, link, sales, ASP, net_sentiment
    
def get_star_rating_html(net_sentiment):
    try:
    # Normalize net sentiment from -100 to 100 to 0 to 10 for star ratings
        normalized_rating = (net_sentiment + 100) / 40
    
        # Determine the number of full and half stars
        full_stars = int(normalized_rating)
        half_star = 1 if normalized_rating - full_stars >= 0.5 else 0
    
        # Generate the HTML for the stars
        star_html = '<span style="color: gold;">'
        star_html += '★' * full_stars
        star_html += '½' * half_star
        star_html += '☆' * (5 - full_stars - half_star)
        star_html += '</span>'
        return star_html
    except:
        return "NA"
        
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
    global interaction
    device_name, img_link = get_device_image(device_input)
    net_Sentiment,aspect_sentiment = get_net_sentiment(device_name)
    total_sales = get_sales_units(device_name)
    asp = get_ASP(device_name)
    high_specs, sale = get_highest_selling_specs(device_name)
    star_rating_html = get_star_rating_html(net_Sentiment)
    comp_devices = compete_device(device_name)
    interaction = ""
    return device_name, img_link, net_Sentiment, aspect_sentiment, total_sales, asp, high_specs, sale, star_rating_html, comp_devices

def load_and_resize_image(url, new_height):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        aspect_ratio = img.width / img.height
        new_width = int(aspect_ratio * new_height)
        resized_img = img.resize((new_width, new_height))
        return resized_img  # Return the resized PIL image object
    except Exception as e:
        st.write("Image not available for this product.")
        st.write(f"Error: {e}")
        return None
    
def get_txt_text(txt_file_path):
    with io.open(txt_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = AzureOpenAIEmbeddings(azure_deployment="Mv_Agusts")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss-index")
    
def device_details(device):
    device_name, img_link, net_Sentiment, aspect_sentiment, total_sales, asp, high_specs, sale, star_rating_html, comp_devices = generate_device_details(device)
    aspects = ['Performance', 'Design', 'Display', 'Battery', 'Price', 'Software']
    with st.container():
        if device_name:
            if img_link:
                image1 = load_and_resize_image(img_link, 150)
                st.image(image1)
            else:
                st.write("Image not available for this product.")
            st.header(device_name)
            st.markdown(star_rating_html, unsafe_allow_html=True)
            st.write(f"Total Devices Sold: {total_sales}")
            st.write(f"Average Selling Price: {asp}")
            st.write(f"Highest Selling Specs: {high_specs} - {sale}")
            st.subheader('Aspect Ratings')
            asp_rating = []
            for i in aspect_sentiment:
                asp_rating.append(get_star_rating_html(i))
            for aspect, stars in zip(aspects, asp_rating):
                st.markdown(f"{aspect}: {stars}",unsafe_allow_html=True)
            data_1 = query_quant("Give me all the reviews of " + device_name,[])
            a = device_name + "_Reviews.txt"
            data_1.to_csv(a, sep='\t')
            summary_1 = query_to_embedding_summarize("Give me the pros and cons of " + device_name, a)
#             summary_1 = "Placeholder Summary"
            st.subheader(device_name)
            st.write(summary_1)

def comparison_view(device1, device2):
    st.write(r"$\textsf{\Large Device Comparison}$")
    col1, col2 = st.columns(2)
    with col1:
        device_details(device1)
    with col2:
        device_details(device2)

    
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
                device_name, img_link, net_Sentiment, aspect_sentiment, total_sales, asp, high_specs, sale, star_rating_html, comp_devices = generate_device_details(inp)
                summ = get_detailed_summary(inp)
#                 summ = "Summary Placeholder"
                st.session_state['chat_history'].append((inp, device_name, img_link, net_Sentiment, total_sales, asp, high_specs, sale, star_rating_html, comp_devices, summ, aspect_sentiment))
            elif inp != st.session_state['chat_history'][-1][0]:
                device_name, img_link, net_Sentiment, aspect_sentiment, total_sales, asp, high_specs, sale, star_rating_html, comp_devices = generate_device_details(inp)
#                 summ = get_detailed_summary(inp)
                summ = "Summary Placeholder"
                st.session_state['chat_history'].append((inp, device_name, img_link, net_Sentiment, total_sales, asp, high_specs, sale, star_rating_html, comp_devices, summ, aspect_sentiment))
            else:
                device_name = st.session_state['chat_history'][-1][1]
                img_link = st.session_state['chat_history'][-1][2]
                net_Sentiment = st.session_state['chat_history'][-1][3]
                total_sales = st.session_state['chat_history'][-1][4]
                asp = st.session_state['chat_history'][-1][5]
                high_specs = st.session_state['chat_history'][-1][6]
                sale = st.session_state['chat_history'][-1][7]
                star_rating_html = st.session_state['chat_history'][-1][8]
                comp_devices = st.session_state['chat_history'][-1][9]
                summ = st.session_state['chat_history'][-1][10]
                aspect_sentiment = st.session_state['chat_history'][-1][11]
            html_code = f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); display: flex; align-items: center;">
                <div style="flex: 1; text-align: center;">
                    <img src="{img_link}" style="width: 150px; display: block; margin: 0 auto;">
                    <p style="color: black; font-size: 18px;">{device_name}</p>
                    <p>{star_rating_html}</p>
                </div>
                <div style="width: 2px; height: 150px; border-left: 2px dotted #ccc; margin: 0 20px;"></div>
                <div style="flex: 2; color: black; font-size: 18px;">
                    <p>Total Devices Sold: <strong>{total_sales}</strong></p>
                    <p>Average Selling Price: <strong>{asp}</strong></p>
                    <p>Highest Selling Specs: <strong>{high_specs}</strong> - <strong>{sale}</strong></p>
                </div>
            </div>
            """
            st.markdown(html_code, unsafe_allow_html=True)
            st.write(r"$\textsf{\Large Detailed Summary}$")
            st.write(summ)
            st.write(r"$\textsf{\Large Compare with Similar Devices}$")
            inp = False
            
            checkbox_state = []
            
            html_content = ""
            #From this point
            for device in comp_devices['SERIES']:
                com_device_name, link, com_sales, ASP, net_sentiment = get_comp_device_details(device, comp_devices)
                com_star_rating_html = get_star_rating_html(net_sentiment)
                
                html_content += f"""
                    <div style="text-align: center; padding: 10px; border: 1px solid #ccc; display: inline-block; border-radius: 5px; margin: 10px;">
                        <img src="{link}" width="150" style="margin-bottom: 10px;">
                        <div style="font-size: 16px; color: #333;">{com_device_name}</div>
                        <div style="font-size: 14px; color: #666;">Sales: {com_sales}</div>
                        <div style="font-size: 14px; color: #666;">Average Selling Price: {ASP}</div>
                        <p>{com_star_rating_html}</p>
                    </div>
                """
                checkbox_state.append(False)
            #To this point can be moved to a function to reduce time. To be tested later
                
            st.markdown(html_content, unsafe_allow_html=True)
            comp_devices_list = comp_devices['SERIES'].tolist()
            for i in range(len(comp_devices_list)):
                checkbox_state[i] = st.checkbox(comp_devices_list[i])
            
            for i in range(len(checkbox_state)):
                if checkbox_state[i]:
                    st.session_state['selected_device_comparison'] = comp_devices_list[i]
                    st.write(f"You have selected device {comp_devices_list[i]}")
                    break
                st.session_state['selected_device_comparison'] = None
            
            if st.session_state['selected_device_comparison']:
                comparison_view(device_name,st.session_state['selected_device_comparison'])
            
            
                                        
                            
                                     
    except Exception as e:
        err = f"An error occurred while calling the final function: {e}"
        print(err)
        return err

if __name__ == "__main__":
    main()