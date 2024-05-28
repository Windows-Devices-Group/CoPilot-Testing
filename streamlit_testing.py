import streamlit as st

html_string = """
<div class="container">
  <input type="checkbox" id="myCheckbox">
  <label for="myCheckbox">This is a checkbox</label>
</div>
"""

checkbox_state = st.checkbox("Checkbox (Outside Div)", key="myCheckbox")
st.markdown(html_string, unsafe_allow_html=True)

if checkbox_state:
    st.write("Checkbox (Outside Div) is checked!")
