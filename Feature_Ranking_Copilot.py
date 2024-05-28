#!/usr/bin/env python
# coding: utf-8

# In[31]:


#Import Required Libraries
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
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
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#Initializing API Keys to use LLM
os.environ["AZURE_OPENAI_API_KEY"] = "a22e367d483f4718b9e96b1f52ce6d53"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://fordmustang.openai.azure.com/"

#Reading the dataset
Sentiment_Data  = pd.read_csv("Sampled_Copilot_Reviews_Final.csv")


# In[32]:


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


# In[33]:


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


# In[34]:


def process_tablename(sql, table_name):
    try:
        x = sql.upper()
        query = x.replace(table_name.upper(), table_name)
        return query
    except Exception as e:
        err = f"An error occurred while processing table name in SQL query: {e}"
        return err


# In[35]:


def get_conversational_chain_quant():
        prompt_template = """
        
        If an user is asking for Summarize reviews of any product. Note that user is not seeking for reviews, user is seeking for all the Quantitative things of the product(Net Sentiment & Review Count) and also (Aspect wise sentiment and Aspect wise review count)
        So choose to Provide Net Sentiment and Review Count and Aspect wise sentiment and their respective review count and Union them in single table
        
        Example : If the user Quesiton is "Summarize reviews of CoPilot Produt"
        
        User seeks for net sentiment and aspect wise net sentiment of "CoPilot" Product and their respective review count in a single table
        
        Your response should be : Overall Sentiment is nothing but the net sentiment and overall review count of the product
        
                        Aspect Aspect_SENTIMENT REVIEW_COUNT
                    0 TOTAL 40 15000.0
                    1 Generic 31.8 2302.0
                    2 Microsoft Product 20.2 570.0
                    3 Productivity 58.9 397.0
                    4 Code Generation -1.2 345.0
                    5 Ease of Use 20.1 288.0
                    6 Interface -22.9 271.0
                    7 Connectivity -43.7 247.0
                    8 Compatibility -28.6 185.0
                    9 Innovation 52.9 170.0
                    10 Text Summarization/Generation 19.1 157.0
                    11 Reliability -44.7 152.0
                    12 Price 29.5 95.0
                    13 Customization/Personalization 18.9 90.0
                    14 Security/Privacy -41.3 75.0
                    15 Accessibility 16.7 6.0
                    
                    The Query has to be like this 
                    
                SELECT 'TOTAL' AS Aspect, 
                ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                SUM(Review_Count) AS Review_Count
                FROM Sentiment_Data
                WHERE Product_Family LIKE '%Copilot%'

                UNION

                SELECT Aspect, 
                ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                SUM(Review_Count) AS Review_Count
                FROM Sentiment_Data
                WHERE Product_Family LIKE '%Copilot%'
                GROUP BY Aspect

                ORDER BY Review_Count DESC

                    
                    
                IMPORTANT : if any particular Aspect "Code Generation" in user prompt:
                    

                        SELECT 'TOTAL' AS Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Sentiment_Data
                        WHERE Product_Family LIKE '%Copilot%'

                        UNION

                        SELECT Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Sentiment_Data
                        WHERE Product_Family LIKE '%Copilot%'
                        GROUP BY Aspect
                        HAVING Aspect LIKE %'Code Generation'%

                        ORDER BY Review_Count DESC


        
        IMPORTANT : IT has to be Net sentiment and Aspect Sentiment. Create 2 SQL Query and UNION them
        
        1. Your Job is to convert the user question to SQL Query (Follow Microsoft SQL server SSMS syntax.). You have to give the query so that it can be used on Microsoft SQL server SSMS.You have to only return query as a result.
            2. There is only one table with table name Sentiment_Data where each row is a user review. The table has 10 columns, they are:
                Review: Review of the Copilot Product
                Data_Source: From where is the review taken. It contains following values: 'LaptopMag', 'PCMag', 'Verge', 'ZDNET', 'PlayStore', 'App Store','AppStore', 'Reddit', 'YouTube'.
                Geography: From which Country or Region the review was given. It contains following values: 'Unknown', 'Brazil', 'Australia', 'Canada', 'China', 'Germany','France'.
                Title: What is the title of the review
                Review_Date: The date on which the review was posted
                Product: Corresponding product for the review. It contains following values: 'COPILOT'.
                Product_Family: Which version or type of the corresponding Product was the review posted for. It contains following values: 'Copilot in Windows 11', 'Copilot for Microsoft 365','Microsoft Copilot', 'Copilot for Security', 'Copilot Pro','Github Copilot', 'Copilot for Mobile'.
                Sentiment: What is the sentiment of the review. It contains following values: 'positive', 'neutral', 'negative'.
                Aspect: The review is talking about which aspect or feature of the product. It contains following values: 'Microsoft Product', 'Interface', 'Connectivity', 'Privacy','Compatibility', 'Generic', 'Innovation', 'Reliability','Productivity', 'Price', 'Text Summarization/Generation','Code Generation', 'Ease of Use', 'Performance','Personalization/Customization'.
                Keyword: What are the keywords mentioned in the product
                Review_Count - It will be 1 for each review or each row
                Sentiment_Score - It will be 1, 0 or -1 based on the Sentiment.
            3. Sentiment mark is calculated by sum of Sentiment_Score.
            4. Net sentiment is calculcated by sum of Sentiment_Score divided by sum of Review_Count. It should be in percentage. Example:
                    SELECT ((SUM(Sentiment_Score)*1.0)/(SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment 
                    FROM Sentiment_Data
                    ORDER BY Net_Sentiment DESC
            5. Net sentiment across country or across region is sentiment mark of a country divided by total reviews of that country. It should be in percentage.
                Example to calculate net sentiment across country:
                    SELECT Geography, ((SUM(Sentiment_Score)*1.0) / (SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment
                    FROM Sentiment_Data
                    GROUP BY Geography
                    ORDER BY Net_Sentiment DESC
            6. Net Sentiment across a column "X" is calculcated by Sentiment Mark for each "X" divided by Total Reviews for each "X".
                Example to calculate net sentiment across a column "X":
                    SELECT X, ((SUM(Sentiment_Score)*1.0) / (SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment
                    FROM Sentiment_Data
                    GROUP BY X
                    ORDER BY Net_Sentiment DESC
            7. Distribution of sentiment is calculated by sum of Review_Count for each Sentiment divided by overall sum of Review_Count
                Example: 
                    SELECT Sentiment, SUM(ReviewCount)*100/(SELECT SUM(Review_Count) AS Reviews FROM Sentiment_Data) AS Total_Reviews 
                    FROM Sentiment_Data 
                    GROUP BY Sentiment
                    ORDER BY Total_Reviews DESC
            8. Convert numerical outputs to float upto 1 decimal point.
            9. Always include ORDER BY clause to sort the table based on the aggregate value calculated in the query.
            10. Top Country is based on Sentiment_Score i.e., the Country which have highest sum(Sentiment_Score)
            11. Always use 'LIKE' operator whenever they mention about any Country. Use 'LIMIT' operator instead of TOP operator.Do not use TOP OPERATOR. Follow syntax that can be used with pandasql.
            12. If you are using any field in the aggregate function in select statement, make sure you add them in GROUP BY Clause.
            13. Make sure to Give the result as the query so that it can be used on Microsoft SQL server SSMS.
            14. Important: Always show Net_Sentiment in Percentage upto 1 decimal point. Hence always make use of ROUND function while giving out Net Sentiment and Add % Symbol after it.
            15. Important: User can ask question about any categories including Aspects, Geograpgy, Sentiment etc etc. Hence, include the in SQL Query if someone ask it.
            16. Important: You Response should directly starts from SQL query nothing else.
            17. Important: Always use LIKE keyword instead of = symbol while generating SQL query.
            18. Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.
            19. Sort all Quantifiable outcomes based on review count
        Context:\n {context}?\n
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

#Function to convert user prompt to quantitative outputs for Copilot Review Summarization
def query_quant(user_question, vector_store_path="faiss_index_CopilotSample"):
    try:
        # Initialize the embeddings model
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Mv_Agusta")
        
        # Load the vector store with the embeddings model
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        
        # Rest of the function remains unchanged
        chain = get_conversational_chain_quant()
        docs = []
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        SQL_Query = response["output_text"]
        SQL_Query = convert_top_to_limit(SQL_Query)
#         print(SQL_Query)
        SQL_Query = process_tablename(SQL_Query,"Sentiment_Data")
    #     print(SQL_Query)
        data = ps.sqldf(SQL_Query, globals())
        data_1 = data
        html_table = data.to_html(index=False)
    #     return html_table
        return data_1
    except Exception as e:
        err = f"An error occurred while generating response for quantitative review summarization: {e}"
        return err


# In[36]:


def make_desired_df(data):
    try:
        # Create DataFrame from the dictionary
        df = pd.DataFrame(data)
        
        # Ensure the necessary columns are present
        if 'ASPECT_SENTIMENT' not in df.columns or 'REVIEW_COUNT' not in df.columns:
            raise ValueError("Input data must contain 'ASPECT_SENTIMENT' and 'REVIEW_COUNT' columns")
        
        df = df[df['ASPECT_SENTIMENT'] != 0]
#         df = df[(df['ASPECT_SENTIMENT'] != 0) & (df['ASPECT'] != 'TOTAL') & (df['ASPECT'] != 'Generic')]

        # Compute min and max values for normalization
        min_sentiment = df['ASPECT_SENTIMENT'].min(skipna=True)
        max_sentiment = df['ASPECT_SENTIMENT'].max(skipna=True)
        min_review_count = df['REVIEW_COUNT'].min(skipna=True)
        max_review_count = df['REVIEW_COUNT'].max(skipna=True)

        # Apply min-max normalization for ASPECT_SENTIMENT
        df['NORMALIZED_SENTIMENT'] = df.apply(
            lambda row: (row['ASPECT_SENTIMENT'] - min_sentiment) / (max_sentiment - min_sentiment)
            if pd.notnull(row['ASPECT_SENTIMENT'])
            else None,
            axis=1
        )

        # Apply min-max normalization for REVIEW_COUNT
        df['NORMALIZED_REVIEW_COUNT'] = df.apply(
            lambda row: (row['REVIEW_COUNT'] - min_review_count) / (max_review_count - min_review_count)
            if pd.notnull(row['REVIEW_COUNT'])
            else None,
            axis=1
        )

        # Calculate the aspect ranking based on normalized values
        weight_for_sentiment = 3
        weight_for_review_count = 2 
        
        df['ASPECT_RANKING'] = df.apply(
            lambda row: (weight_for_review_count * row['NORMALIZED_REVIEW_COUNT'] * (1 - weight_for_review_count*row['NORMALIZED_SENTIMENT'])
            if pd.notnull(row['NORMALIZED_SENTIMENT']) and pd.notnull(row['NORMALIZED_REVIEW_COUNT'])
            else None),
            axis=1
        )

        # Assign integer rankings based on the 'Aspect_Ranking' score
        df['ASPECT_RANKING'] = df['ASPECT_RANKING'].rank(method='max', ascending=False, na_option='bottom').astype('Int64')

        # Sort the DataFrame based on 'Aspect_Ranking' to get the final ranking
        df_sorted = df.sort_values(by='ASPECT_RANKING')
        
        # Extract and display the net sentiment and overall review count
        try:
            total_row = df[df['ASPECT'] == 'TOTAL'].iloc[0]
            net_sentiment = str(int(total_row["ASPECT_SENTIMENT"])) + '%'
            overall_review_count = int(total_row["REVIEW_COUNT"])
        except (ValueError, TypeError, IndexError):
            net_sentiment = total_row["ASPECT_SENTIMENT"]
            overall_review_count = total_row["REVIEW_COUNT"]

        st.write(f"Net Sentiment: {net_sentiment}")
        st.write(f"Overall Review Count: {overall_review_count}")

        return df_sorted
    except Exception as e:
        st.error(f"Error in make_desired_df: {str(e)}")
        return pd.DataFrame()


# In[37]:


import numpy as np

def custom_color_gradient(val, vmin, vmax):
    green_hex = '#347c47'
    middle_hex = '#dcdcdc'
    lower_hex = '#b0343c'
    
    # Normalize the value based on the new vmin and vmax
    normalized_val = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
    
    if normalized_val <= 0.5:
        # Interpolate between lower_hex and middle_hex for values <= 0.5
        r = int(np.interp(normalized_val, [0, 0.5], [int(lower_hex[1:3], 16), int(middle_hex[1:3], 16)]))
        g = int(np.interp(normalized_val, [0, 0.5], [int(lower_hex[3:5], 16), int(middle_hex[3:5], 16)]))
        b = int(np.interp(normalized_val, [0, 0.5], [int(lower_hex[5:7], 16), int(middle_hex[5:7], 16)]))
    else:
        # Interpolate between middle_hex and green_hex for values > 0.5
        r = int(np.interp(normalized_val, [0.5, 1], [int(middle_hex[1:3], 16), int(green_hex[1:3], 16)]))
        g = int(np.interp(normalized_val, [0.5, 1], [int(middle_hex[3:5], 16), int(green_hex[3:5], 16)]))
        b = int(np.interp(normalized_val, [0.5, 1], [int(middle_hex[5:7], 16), int(green_hex[5:7], 16)]))
    
    # Convert interpolated RGB values to hex format for CSS color styling
    hex_color = f'#{r:02x}{g:02x}{b:02x}'
    
    return f'background-color: {hex_color}; color: black;'


# In[38]:


import os
from openai import AzureOpenAI
os.environ["AZURE_OPENAI_API_KEY"] = "3a3850af863b4dddbc2d3834f0ff097b"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://fordmustang.openai.azure.com/"
client = AzureOpenAI(
api_key=os.getenv("3a3850af863b4dddbc2d3834f0ff097b"),  
api_version="2024-02-01",
azure_endpoint = os.getenv("https://fordmustang.openai.azure.com/")
)

deployment_name='Surface_Analytics'

start_phrase_verbatim = """

    Your job is to convert the user question to an SQL query (following Microsoft SQL server SSMS syntax). You have to give the query so that it can be used on Microsoft SQL server SSMS. You have to only return the query as a result.
    There is only one table with the table name Sentiment_Data where each row is a user review. The table has the following columns:
    Review: Review of the Copilot Product
    Data_Source: From where the review is taken. It contains the following values: ‘LaptopMag’, ‘PCMag’, ‘Verge’, ‘ZDNET’, ‘PlayStore’, ‘App Store’, ‘AppStore’, ‘Reddit’, ‘YouTube’.
    Geography: From which Country or Region the review was given. It contains the following values: ‘Unknown’, ‘Brazil’, ‘Australia’, ‘Canada’, ‘China’, ‘Germany’, ‘France’.
    Title: The title of the review
    Review_Date: The date on which the review was posted
    Product: Corresponding product for the review. It contains the following value: ‘COPILOT’.
    Product_Family: Which version or type of the corresponding Product the review was posted for. It contains the following values: ‘Copilot in Windows 11’, ‘Copilot for Microsoft 365’, ‘Microsoft Copilot’, ‘Copilot for Security’, ‘Copilot Pro’, ‘Github Copilot’, ‘Copilot for Mobile’.
    Sentiment: The sentiment of the review. It contains the following values: ‘positive’, ‘neutral’, ‘negative’.
    Aspect: The review is talking about which aspect or feature of the product. It contains the following values:‘Interface’, ‘Connectivity’, ‘Privacy’, ‘Compatibility’, ‘Innovation’, ‘Reliability’, ‘Productivity’, ‘Price’, ‘Text Summarization/Generation’, ‘Code Generation’, ‘Ease of Use’, ‘Performance’, ‘Personalization/Customization’.
    Keywords: The keywords mentioned in the product
    Review_Count: It will be 1 for each review or each row
    Sentiment_Score: It will be 1, 0, or -1 based on the Sentiment.
    Sentiment mark is calculated by the sum of Sentiment_Score.
    Net sentiment is calculated by the sum of Sentiment_Score divided by the sum of Review_Count. It should be in percentage.
    Net sentiment across the country or region is the sentiment mark of a country divided by the total reviews of that country. It should be in percentage.
    Net Sentiment across a column “X” is calculated by the Sentiment Mark for each “X” divided by the Total Reviews for each “X”.
    Distribution of sentiment is calculated by the sum of Review_Count for each Sentiment divided by the overall sum of Review_Count.
    Convert numerical outputs to float up to 1 decimal point.
    Always include the ORDER BY clause to sort the table based on the aggregate value calculated in the query.
    The top country is based on Sentiment_Score, i.e., the country which has the highest sum(Sentiment_Score).
    Always use the ‘LIKE’ operator whenever they mention any country. Use the ‘LIMIT’ operator instead of the ‘TOP’ operator. Do not use the ‘TOP’ OPERATOR. Follow the syntax that can be used with pandasql.
    If you are using any field in the aggregate function in the select statement, make sure you add them in the GROUP BY clause.
    Make sure to give the result as the query so that it can be used on Microsoft SQL server SSMS.
    Important: Always show Net_Sentiment in Percentage up to 1 decimal point. Hence always make use of the ROUND function while giving out Net Sentiment and add a % symbol after it.
    Important: The user can ask questions about any categories including Aspects, Geography, Sentiment, etc. Hence, include them in the SQL Query if someone asks it.
    Important: Your response should directly start from the SQL query, nothing else.
    Important: Always use the ‘LIKE’ keyword instead of the ‘=’ symbol while generating SQL queries.
    Important: Generate outputs using the provided dataset only, don’t use pre-trained information to generate outputs.    
    User Question :
    
    """
def query_verbatims(review):
    SQL_Query_Temp = client.completions.create(model=deployment_name, prompt=start_phrase_verbatim+review, max_tokens=1000,temperature=0)
    SQL_Query = SQL_Query_Temp.choices[0].text
    data_verbatims = ps.sqldf(SQL_Query,globals())
    return data_verbatims


# In[39]:


def show_aspect_info(aspect, df):
    row = df[df['ASPECT'] == aspect].iloc[0]
    output = {'Aspect': aspect}
    if not pd.isnull(row['ASPECT_SENTIMENT']) and not pd.isnull(row['REVIEW_COUNT']):
        output['Aspect Sentiment'] = row['ASPECT_SENTIMENT']
        output['Aspect Review Count'] = row['REVIEW_COUNT']
        output['Net Sentiment'] = row.get('NORMALIZED_SENTIMENT', 'No data')
    else:
        output['Net Sentiment'] = 'No data found for aspect'
    
    return output


# In[40]:


from langchain_openai import AzureChatOpenAI
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import AzureOpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage


# In[41]:


def get_txt_text(txt_file_path):
    with io.open(txt_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


# In[42]:


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
    chunks = text_splitter.split_text(text)
    return chunks


# In[43]:


def get_vector_store(chunks):
    embeddings = AzureOpenAIEmbeddings(azure_deployment="Mv_Agusta")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss-index")


# In[44]:


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.environ["AZURE_OPENAI_API_KEY"] = "3a3850af863b4dddbc2d3834f0ff097b"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://fordmustang.openai.azure.com/"


def get_conversational_chain_summary():
    prompt_template = """
    1. You will receive customer feedback about devices, focusing on specific aspects as user input.
    2. These are the potential aspects for improvement or new feature suggestions: ‘Interface’, ‘Connectivity’, ‘Privacy’, ‘Compatibility’, ‘Innovation’, ‘Reliability’, ‘Productivity’, ‘Price’, ‘Text Summarization/Generation’, ‘Code Generation’, ‘Ease of Use’, ‘Performance’, ‘Personalization/Customization’.
    3. Your job is to analyze the feedback and identify areas for improvement or new features that could enhance the user experience. Summarize these insights into a concise paragraph of 4 to 5 lines.
    4. The response should always be a summary of potential improvements or feature suggestions in a 10 to 15 line paragraph. Focus only on the aspect the user asked about.
    5. If the user asks about the Performance aspect, concentrate solely on performance-related feedback. Your suggestions should only address performance improvements or new features related to performance.
    
    Condition 1: If the net sentiment is less than the aspect sentiment, it indicates that this aspect has room for improvement, which could raise the overall sentiment for that device. Provide suggestions on how to improve this aspect.     
    Condition 2: If the net sentiment is higher than the aspect sentiment, it suggests that this aspect is satisfactory but could be enhanced further. Offer ideas for additional features that could elevate this aspect even more.

    Your summary should justify the above conditions and tie in with the net sentiment and aspect sentiment. Mention the difference between Net Sentiment and Aspect Sentiment (e.g., -2% or +2% higher than net sentiment) in your summary and provide justification.

    Additionally, create a list of potential improvements and new feature suggestions for that aspect of the device. 
    
    IMPORTANT: Your response should include a Summary of potential improvements or feature suggestions, Pros: List down max 5 points, Cons: List down max 5 points in the form of a table.
    IMPORTANT: Use only the data provided to you and do not rely on pre-trained documents.
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
    # Rest of the function remains unchanged
    chain = get_conversational_chain_summary()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response['output_text']


# In[45]:


st.title("Reviews Analyser")
device_name = st.text_input("Enter the device name : ")
if device_name:
    st.subheader("Net Sentiment and Aspect Sentiment : ")
    data = query_quant("Give me Net Sentiment and review count and different Aspect Sentiment" + device_name + "?")
    data_new = make_desired_df(data)
    top_aspects = data_new.sort_values('ASPECT_RANKING', ascending=True).head(3)['ASPECT'].tolist()
    vmin = data_new['ASPECT_SENTIMENT'].min()
    vmax = data_new['ASPECT_SENTIMENT'].max()
    styled_df = data_new.style.applymap(lambda x: custom_color_gradient(x, vmin, vmax), subset=['ASPECT_SENTIMENT'])
    st.subheader("Heat-Map")
    st.dataframe(styled_df, hide_index=True)   
    st.subheader("Feature Suggestion")
    for Aspect in top_aspects:
        print(f"Processing Aspect: {Aspect}") 
        if Aspect in data_new['ASPECT'].values:
            print(f"Found Aspect in data: {Aspect}")            
            aspect_data = data_new[data_new['ASPECT'] == Aspect]
            show_aspect_info(Aspect, aspect_data)
            data_verbatims = query_verbatims("Give me reviews of " + device_name + " for " + Aspect + " Aspect")
            if Aspect =="Security/Privacy":
                Aspect = "Security"
            data_verbatims.to_csv(Aspect + "_Verbatim.txt", sep='\t')
            a = Aspect + "_Verbatim.txt"
            summary = query_to_embedding_summarize("Summarize the reviews of " + Aspect + " Aspect " +"Which have " + str(aspect_data['ASPECT_SENTIMENT'].iloc[0]) + "% Aspect Sentiment and " +str(aspect_data['NORMALIZED_SENTIMENT'].iloc[0]) + "% normalized sentiment", a)
            st.subheader("Summary for " + Aspect)
            st.write(summary)
        else:
            print(f"Aspect '{Aspect}' not found in data.")
            st.write(f"Aspect '{Aspect}' not found in data columns.")


# In[ ]:




