import streamlit as st

# Mock functions for demonstration
# def compete_device(device_name):
#     return ["Surface Laptop", "Envy 17", "Spectre x360"]

# def get_comp_device_image(device):
#     links = {
#         "Surface Laptop": "https://m.media-amazon.com/images/I/71DozMyPCBL.jpg",
#         "Envy 17": "https://images-cdn.ubuy.co.in/63685cab47f0116deb444b12-hp-envy-17-laptop-17-3-fhd.jpg",
#         "Spectre x360": "https://m.media-amazon.com/images/I/71QKMcZs-qL.jpg"
#     }
#     return device, links.get(device)

# # Assuming `device_name` is defined
# device_name = "Example Device"
# comp_devices = compete_device(device_name)
# device_links = {}
# for device in comp_devices:
#     dev, link = get_comp_device_image(device)
#     if dev is not None:
#         device_links[dev] = link

html_content = ""
device_links = {
    "Surface Laptop": "https://m.media-amazon.com/images/I/71DozMyPCBL.jpg",
    "Envy 17": "https://images-cdn.ubuy.co.in/63685cab47f0116deb444b12-hp-envy-17-laptop-17-3-fhd.jpg",
    "Spectre x360": "https://m.media-amazon.com/images/I/71QKMcZs-qL.jpg"
}

for com_device_name, link in device_links.items():
    html_content += f"""
        <div style="text-align: center; padding: 10px; border: 1px solid #ccc; display: inline-block; border-radius: 5px; margin: 10px;">
            <img src="{link}" width="150" style="margin-bottom: 10px;">
            <div style="font-size: 16px; color: #333;">{com_device_name}</div>
        </div>
    """
# Render the CSS and HTML
# st.markdown(html_content, unsafe_allow_html=True)


# Render the CSS and HTML
st.markdown(html_content, unsafe_allow_html=True)
