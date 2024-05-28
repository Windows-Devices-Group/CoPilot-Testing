#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from openai import AzureOpenAI
import pyodbc
import urllib
from sqlalchemy import create_engine
import pandas as pd
import keyring
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
from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import re
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#Initializing API Keys to use LLM
os.environ["AZURE_OPENAI_API_KEY"] = "3a3850af863b4dddbc2d3834f0ff097b"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://fordmustang.openai.azure.com/"
os.environ['AZURE OPENAI API VERSION'] = '2024-03-01-preview'


#Reading the dataset
# Sentiment_Data  = pd.read_excel("New_Consolidated.xlsx")
Sentiment_Data= pd.read_csv("Windows_Data_116K.csv")

#Function to derive Sentiment Score based on Sentiment
def Sentiment_Score_Derivation(value):
    try:
        if value.lower() == "positive":
            return 1
        elif value.lower() == "negative":
            return -1
        else:
            return 0
    except Exception as e:
        err = f"An error occurred while deriving Sentiment Score: {e}"
        return err    

# #Deriving Sentiment Score and Review Count columns into the dataset
Sentiment_Data["Sentiment_Score"] = Sentiment_Data["Sentiment"].apply(Sentiment_Score_Derivation)
Sentiment_Data["Review_Count"] = 1.0


################################# Definiting Functions #################################

#Review Summarization (Detailed) + Feature Comparison and Suggestion

#Function to extract text from file
def get_text_from_file(txt_file):
    try:
        with open(txt_file, 'r',encoding='latin') as file:
            text = file.read()
        return text
    except Exception as e:
        err = f"An error occurred while getting text from file: {e}"
        return err

# Function to split text into chunks
def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        err = f"An error occurred while getting text chunks: {e}"
        return err

# Function to create and store embeddings
def get_vector_store(text_chunks):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index_Windows_116k")
        return vector_store
    except Exception as e:
        err = f"An error occurred while getting vectos: {e}"
        return err

# Function to setup the vector store (to be run once or upon text update)
def setup(txt_file_path):
    try:
        raw_text = get_text_from_file(txt_file_path)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        print("Setup completed. Vector store is ready for queries.")
    except Exception as e:
        err = f"An error occurred while setting up vector store: {e}"
        return err


## Review Summarization (Quantifiable)

#Converting Top Operator to Limit Operator as pandasql doesn't support Top
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

#Function to add Table Name into the SQL Query as it is, as the Table Name is Case Sensitive here
def process_tablename(sql, table_name):
    try:
        x = sql.upper()
        query = x.replace('=', ' LIKE ')
        query = x.replace('! LIKE', ' != ')
        if 'LIKE' in query and '%' not in query:
            pattern = r"(LIKE\s+')(.*?')"
            def replacer(match):
                # Extract the matched groups
                prefix = match.group(1)
                value = match.group(2)   

                # Add % around the value
                new_value = f"%{value[0:-1]}%"+"'"

                # Return the modified string
                return prefix + new_value
            query = re.sub(pattern, replacer, query)
        query = query.replace(table_name.upper(),table_name) 
        return query
    except Exception as e:
        err = f"An error occurred while processing table name in SQL query: {e}"
        return err

## Generating Response by Identifying Prompt Nature

#Function to get conversation chain for quantitative outputs and also add context from historical conversation as well
def get_conversational_chain_quant():
    try:
#         hist = """"""
#         for i in history:
#             hist = hist+"\nUser: "+i[0]
#             if isinstance(i[1],pd.DataFrame):
#                 x = i[1].to_string()
#             else:
#                 x = i[1]
#             hist = hist+"\nResponse: "+x

#################################################################################################################################################################################################################################################
        prompt_template = """
        
        Your Job is to convert the user question to SQL Query (Follow Microsoft SQL server SSMS syntax.). You have to give the query so that it can be used on Microsoft SQL server SSMS.You have to only return query as a result.
        There is only one table with table name Sentiment_Data where each row is a user review. The table has 10 columns, they are:
                Review: Review of the Copilot Product
                Data_Source: From where is the review taken. It contains different retailers
                Geography: From which Country or Region the review was given. It contains different Grography.
                Title: What is the title of the review
                Review_Date: The date on which the review was posted
                Product: Corresponding product for the review. It contains following values: "Windows 11 (Preinstall)", "Windows 10"
                Product_Family: Which version or type of the corresponding Product was the review posted for. Different Device Names
                Sentiment: What is the sentiment of the review. It contains following values: 'Positive', 'Neutral', 'Negative'.
                Aspect: The review is talking about which aspect or feature of the product. It contains following values: "Audio-Microphone","Software","Performance","Storage/Memory","Keyboard","Browser","Connectivity","Hardware","Display","Graphics","Battery","Gaming","Design","Ports","Price","Camera","Customer-Service","Touchpad","Account","Generic"
                Keyword: What are the keywords mentioned in the product
                Review_Count - It will be 1 for each review or each row
                Sentiment_Score - It will be 1, 0 or -1 based on the Sentiment.
                
            1. If the user asks for count of column 'X', the query should be like this:
                    SELECT COUNT(DISTINCT ('X')) 
                    FROM Sentiment_Data
            2. If the user asks for count of column 'X' for different values of column 'Y', the query should be like this:
                    SELECT 'Y', COUNT(DISTINCT('X')) AS Total_Count
                    FROM Sentiment_Data 
                    GROUP BY 'Y'
                    ORDER BY TOTAL_COUNT DESC
            3. If the user asks for Net overall sentiment the query should be like this:
                    SELECT ((SUM(Sentiment_Score))/(SUM(Review_Count))) * 100 AS Net_Sentiment 
                    FROM Sentiment_Data
                    ORDER BY Net_Sentiment DESC
            4. If the user asks for Net Sentiment across a column "X", the query should be like this:
                    SELECT X, ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment
                    FROM Sentiment_Data
                    GROUP BY X
                    ORDER BY Net_Sentiment DESC
            5. If the user asks for overall review count, the query should be like this:
                    SELECT SUM(Review_Count) 
                    FROM Sentiment_Data
            6. If the user asks for review distribution across column 'X', the query should be like this:
                    SELECT 'X', SUM(Review_Count) * 100 / (SELECT SUM(Review_Count) FROM Sentiment_Data) AS Review_Distribution
                    FROM Sentiment_Data 
                    GROUP BY 'X'
                    ORDER BY Review_Distribution DESC
            7. If the user asks for column 'X' Distribution across column 'Y', the query should be like this: 
                    SELECT 'Y', SUM('X') * 100 / (SELECT SUM('X') AS Reviews FROM Sentiment_Data) AS Distribution_PCT
                    FROM Sentiment_Data 
                    GROUP BY 'Y'
                    ORDER BY Distribution_PCT DESC
                    
                    
            Always include ORDER BY clause to sort the table based on the aggregate value calculated in the query.
            Use 'LIMIT' operator instead of TOP operator.Do not use TOP OPERATOR. Follow syntax that can be used with pandasql.
            Important: You Response should directly start from SQL query nothing else.
            Important: Always replace LIKE keyword with '=' symbol and add '%' before and after filter value in the generated SQL query.
            Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.
        
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
########################################################################################################################################
#########################################################################################
        

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment="Thruxton_R",
            api_version='2024-03-01-preview',
            temperature = 0.3)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for quantifiable review summarization: {e}"
        return err

#Function to convert user prompt to quantitative outputs for Copilot Review Summarization
def query_quant(user_question, vector_store_path="faiss_index_Windows_116k"):
    try:
        # Initialize the embeddings model
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        
        # Load the vector store with the embeddings model
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        
        # Rest of the function remains unchanged
        chain = get_conversational_chain_quant()
        docs = []
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        SQL_Query = response["output_text"]
        SQL_Query = convert_top_to_limit(SQL_Query)
        SQL_Query = process_tablename(SQL_Query,"Sentiment_Data")
        st.write(SQL_Query)
        data = ps.sqldf(SQL_Query, globals())
        data_1 = data
        html_table = data.to_html(index=False)
    #     return html_table
        return data_1
    except Exception as e:
        err = f"An error occurred while generating response for quantitative review summarization: {e}"
        return err



#Function to generate Quantitative Review Summarization from User Prompt
def quantifiable_data(user_question):
    try:
        response = query_quant(user_question)
        
        return response
    except Exception as e:
        err = f"An error occurred while generating quantitative review summarization: {e}"
        return err



#Function to generate chart based on output dataframe 

def generate_chart(df):
    # Determine the data types of the columns
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    date_cols = df.select_dtypes(include=['datetime']).columns
    
    # Simple heuristic to determine the most suitable chart
    if len(df.columns)==2:
        
        if len(num_cols) == 1 and len(cat_cols) == 0:

            plt.figure(figsize=(10, 6))
            sns.histplot(df[num_cols[0]], kde=True)
            plt.title(f"Frequency Distribution of '{num_cols[0]}'")
            st.pyplot(plt)


        elif len(num_cols) == 2:
   
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df[num_cols[0]], y=df[num_cols[1]])
            plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
            st.pyplot(plt)


        elif len(cat_cols) == 1 and len(num_cols) == 1:
    #         st.write(df[cat_cols[0]].nunique(),df[num_cols[0]].sum())
            if df[cat_cols[0]].nunique() <= 5 and df[num_cols[0]].sum()>=99 and df[num_cols[0]].sum()<=101:
                plt.figure(figsize=(5, 5))
                df.groupby(cat_cols[0])[num_cols[0]].sum().plot(kind='pie', autopct='%1.1f%%')
                plt.ylabel('') 
                plt.xlabel('')
                plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
                st.pyplot(plt)

            else:
                if df[cat_cols[0]].nunique()<=10:
                    plt.figure(figsize=(10, 6))
                    sns.barplot(y=df[cat_cols[0]], x=df[num_cols[0]])
                    plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")

                    st.pyplot(plt)
                else:
                    df1=df.head(10)
                    plt.figure(figsize=(10, 6))
                    sns.barplot(y=df1[cat_cols[0]], x=df1[num_cols[0]])
                    plt.title(f"Distribution of '{num_cols[0]}' across '{num_cols[0]}' : Top Categories")
                    st.pyplot(plt)
                    df2=df.tail(10)
                    plt.figure(figsize=(10, 6))
                    sns.barplot(y=df2[cat_cols[0]], x=df2[num_cols[0]])
                    plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}' : Bottom Categories")
                    st.pyplot(plt)


        elif len(cat_cols) == 2:

            plt.figure(figsize=(10, 6))
            sns.countplot(x=df[cat_cols[0]], hue=df[cat_cols[1]], data=df)
            plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
            st.pyplot(plt)


        elif len(date_cols) == 1 and len(num_cols) == 1:
   
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=df[date_cols[0]], y=df[num_cols[0]], data=df)
            plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
            st.pyplot(plt)


        else:
            sns.pairplot(df)
            st.pyplot(plt)
            
    elif len(df.columns)==3 and len(cat_cols)>=1:
        cat_list=df.iloc[:,0].unique()
        for i in cat_list:
            df1=df[df.iloc[:,0]==i].iloc[:,1:3]
            st.markdown(f"**{i} Overview**")
            st.write(df1)
            num_cols = df1.select_dtypes(include=['number']).columns
            cat_cols = df1.select_dtypes(include=['object', 'category']).columns
            if len(num_cols) == 2:
       
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=df1[num_cols[0]], y=df1[num_cols[1]])
                plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}' under {i}")
                st.pyplot(plt)


            elif len(cat_cols) == 1 and len(num_cols) == 1:
                if df1[cat_cols[0]].nunique() <= 5 and df1[num_cols[0]].sum()>=99 and df1[num_cols[0]].sum()<=101:
        
                    plt.figure(figsize=(5, 5))
                    df1.groupby(cat_cols[0])[num_cols[0]].sum().plot(kind='pie', autopct='%1.1f%%')
                    plt.ylabel('') 
                    plt.xlabel('')
                    plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}' under {i}")
                    st.pyplot(plt)

                else:
        
                    if df1[cat_cols[0]].nunique()<=10:
                        plt.figure(figsize=(10, 6))
                        sns.barplot(y=df1.iloc[:,0], x=df1.iloc[:,1])
                        plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}' under {i}")

                        st.pyplot(plt)
                    else:
                        df2=df1.head(5)
                        plt.figure(figsize=(10, 6))
                        sns.barplot(y=df2[cat_cols[0]], x=df2[num_cols[0]])
                        plt.title(f"Distribution of '{num_cols[0]}' across '{num_cols[0]}' under {i} : Top Categories")
                        st.pyplot(plt)
                        df3=df1.tail(5)
                        plt.figure(figsize=(10, 6))
                        sns.barplot(y=df3[cat_cols[0]], x=df3[num_cols[0]])
                        plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}' under {i} : Bottom Categories")
                        st.pyplot(plt)


            elif len(cat_cols) == 2:
        
                plt.figure(figsize=(10, 6))
                sns.countplot(x=df1[cat_cols[0]], hue=df1[cat_cols[1]], data=df1)
                plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}' under {i}")
                st.pyplot(plt)


            elif len(date_cols) == 1 and len(num_cols) == 1:
        
                plt.figure(figsize=(10, 6))
                sns.lineplot(x=df1[date_cols[0]], y=df1[num_cols[0]], data=df1)
                plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}' under {i}")
                st.pyplot(plt)


            else:
                sns.pairplot(df1)
                st.pyplot(plt)
            
        
#Insight generation from LLM

def generate_chart_insight_llm(user_question):
    try:
        prompt_template = """
        Based on the data available in the input, generate meaningful insights using the numbers and summarize them. Ensure to include all possible insights and findings that can be extracted, which reveals vital trends and patterns. You should generate insights on overall level, and across multiple features, if the dataframe has more than 2 columns.
        Important: If the maximum numerical value is less than or equal to 100, then the numerical column is indicating percentage results - therefore while referring to numbers in your insights, add % at the end of the number.
        IMPORTANT : Use the data from the input only and do not give information from pre-trained data.
        IMPORTANT : Dont provide any prompt message written here in the response, this is for your understanding purpose
           
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment="Thruxton_R",
            api_version='2024-03-01-preview',
            temperature=0.5)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": [], "question": user_question}, return_only_outputs=True)
        st.write("\n\n",response["output_text"])
        return response["output_text"]
            
    except Exception as e:
        err = f"An error occurred while generating conversation chain for identifying nature of prompt: {e}"
        return err

        
    
################################# Model Deployment #################################

def main():
    try:

        # Create a container for logos and title with horizontal layout
        col1, col2, col3 = st.columns([1, 2, 1])
      
        # Display logo on the left
        with col1:
            st.image("microsoft_logo.png", width=50)  # Adjust width as needed

        # Display title in the center
        with col2:
            st.header("Copilot LLM Review Generator")

        # Display logo on the right
        with col3:
            st.image("copilot_logo.svg", width=50)  # Align the logo to the right
      
        # User input section
        user_input = st.text_input("Enter your text:", placeholder="What would you like to process?")

        # Process button and output section
        if st.button("Process"):
            
           
            
            #output = device_llm_review_generator(user_input)
            output= quantifiable_data(user_input)

        
            # Display output based on type (string or dataframe)
            if isinstance(output, pd.DataFrame):
                st.dataframe(output)

                ########################## MODIFIED CODE ####################################
                generate_chart(output)
                
                string=output.to_string(index=False)
                charttype = generate_chart_insight_llm(string)
 

            else:
                st.write(output)

    except Exception as e:
        err = f"An error occurred while calling the final function: {e}"
        print(err)
        return err


if __name__ == "__main__":
    main()

