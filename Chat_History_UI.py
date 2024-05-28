#!/usr/bin/env python
# coding: utf-8

# In[43]:


import streamlit as st
from Feature_Ranking_Copilot import query_quant, make_desired_df, custom_color_gradient, show_aspect_info, query_verbatims, query_to_embedding_summarize

Products = ["All", "Copilot in Windows 11", "Copilot for Microsoft 365", "Microsoft Copilot", "Copilot for Security", "Copilot Pro", "Github Copilot", "Copilot for Mobile"]
Source = ["All", "LaptopMag", "PCMag", "Verge", "ZDNET", "PlayStore", "AppStore", "Reddit", "YouTube"]
Geography = ["All", "Brazil", "Australia", "Canada", "China", "Germany", "France"]

with st.sidebar:
    st.subheader("Select Options")
    selected_product = st.selectbox("Select Product", Products)
    selected_source = st.selectbox("Select Source", Source)
    selected_geography = st.selectbox("Select Geography", Geography)

if "messages" not in st.session_state:
    st.session_state['messages'] = []

st.title("Reviews Analyser")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "is_html" in message and message["is_html"]:
            st.markdown(message["content"], unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Enter the device name:"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    device_name = prompt

    with st.chat_message("assistant"):
        full_response = ""
        try:
            full_response += "Net Sentiment and Aspect Sentiment:\n"
            data = query_quant(f"Give me Net Sentiment and review count and different Aspect Sentiment for {device_name}?")
            data_new = make_desired_df(data)
            top_aspects = data_new.sort_values('ASPECT_RANKING', ascending=True).head(3)['ASPECT'].tolist()
            vmin = data_new['ASPECT_SENTIMENT'].min()
            vmax = data_new['ASPECT_SENTIMENT'].max()
            styled_df = data_new.style.applymap(lambda x: custom_color_gradient(x, vmin, vmax), subset=['ASPECT_SENTIMENT'])        
            
            full_response += "Heat-Map\n"
            df_html = styled_df.to_html(index=False)
            st.dataframe(data_new, hide_index=True)        
            full_response += df_html + "\n"  # Append HTML table to the full response

            full_response += "Feature Suggestion\n"
            for Aspect in top_aspects:
                aspect_response = f"Processing Aspect: {Aspect}\n"
                if Aspect in data_new['ASPECT'].values:
                    aspect_data = data_new[data_new['ASPECT'] == Aspect]
                    show_aspect_info(Aspect, aspect_data)
                    data_verbatims = query_verbatims(f"Give me reviews of {device_name} for {Aspect} Aspect")
                    if Aspect == "Security/Privacy":
                        Aspect = "Security"
                    data_verbatims.to_csv(f"{Aspect}_Verbatim.txt", sep='\t')
                    summary = query_to_embedding_summarize(
                        f"Summarize the reviews of {Aspect} Aspect which have {aspect_data['ASPECT_SENTIMENT'].iloc[0]}% Aspect Sentiment and {aspect_data['NORMALIZED_SENTIMENT'].iloc[0]}% normalized sentiment", 
                        f"{Aspect}_Verbatim.txt"
                    )
                    aspect_response += f"Summary for {Aspect}: {summary}\n"
                else:
                    aspect_response += f"Aspect '{Aspect}' not found in data columns.\n"
                st.write(aspect_response)
                full_response += aspect_response
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.write(error_message)  # Display the error message
            full_response += error_message
        
        st.session_state.messages.append({"role": "assistant", "content": full_response, "is_html": True})