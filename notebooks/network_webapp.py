import streamlit as st
import streamlit.components.v1 as components
import pyautogui
from pathlib import Path

# Set header title
st.title('Network Graph Visualization of network flow algorithms')

path = 'flow_network_vis'
pages = {}
for i, file in enumerate(sorted(Path('.').glob(f'{path}/*.html'))):
    name = f'iteration/augmenting path: {i}'
    pages[name] = str(file)

def open_page(page_name: str) -> None:
    st.header(page_name)
    htmlFile = open(pages[page_name], 'r', encoding = 'utf-8')
    components.html(htmlFile.read(), height=800, width = 900)

# create a button in the side bar that will move to the next page/radio button choice
next = st.sidebar.button('NEXT SLIDE')

choice = st.sidebar.radio("GO TO", tuple(sorted(pages.keys())))
if next:
    pyautogui.press("tab")
    pyautogui.press("down")
else:
    # finally get to whats on each page
    open_page(choice)
