import streamlit as st
from Device_Details import *
from fuzzywuzzy import process
from rapidfuzz import process, fuzz

def identify_devices(input_string):
    global full_response
    # First, check if any device in the list is exactly in the input string
    df = pd.read_csv('Windows_Data_116K.csv')
    devices_list = list(df['Product_Family'].unique())
    for device in devices_list:
        if device in input_string:
            return device
    
    # If no exact match is found, use fuzzy matching
    most_matching_device = process.extractOne(input_string, devices_list, scorer=fuzz.token_set_ratio)
    
    # Check the matching score
    if most_matching_device[1] >= 60:
        return most_matching_device[0]
    else:
        return "Device not available"
        
def device_summarization(user_input):
    user_input = identify_devices(user_input)
    if user_input == "Device not availabe":
        message = "I don't have sufficient data to provide a complete and accurate response at this time. Please provide more details or context."
        st.write(message)
        full_response += message
    else:
        device_name, img_path = get_device_image(user_input)
        net_Sentiment = get_net_sentiment(device_name)
        sales_device_name = get_sales_device_name(device_name)
        total_sales = get_sales_units(sales_device_name)
        asp = get_ASP(sales_device_name)
        high_specs, sale = get_highest_selling_specs(sales_device_name)
        star_rating_html = get_star_rating_html(net_Sentiment)
        html_code = f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); display: flex; align-items: center;">
            <div style="flex: 1; text-align: center;">
                <img src="data:image/jpeg;base64,{base64.b64encode(open(img_path, "rb").read()).decode()}"  style="width: 150px; display: block; margin: 0 auto;">
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
        summ = get_detailed_summary(user_input)
        st.write(summ)
        full_response += summ
        st.write(r"$\textsf{\Large Compete Devices}$")
        comp_devices = compete_device(sales_device_name)
        html_content = ""
        for device in comp_devices['SERIES']:
            com_device_name, img_path, com_sales, ASP, net_sentiment = get_comp_device_details(device, comp_devices)
            com_star_rating_html = get_star_rating_html(net_sentiment)
            html_content += f"""
                <div style="text-align: center; padding: 10px; border: 1px solid #ccc; display: inline-block; border-radius: 5px; margin: 10px;">
                    <img src="data:image/jpeg;base64,{base64.b64encode(open(img_path, "rb").read()).decode()}" width="150" style="margin-bottom: 10px;">
                    <div style="font-size: 16px; color: #333;">{com_device_name}</div>
                    <div style="font-size: 14px; color: #666;">Sales: {com_sales}</div>
                    <div style="font-size: 14px; color: #666;">Average Selling Price: {ASP}</div>
                    <p>{com_star_rating_html}</p>
                </div>
            """
        st.markdown(html_content, unsafe_allow_html=True)
