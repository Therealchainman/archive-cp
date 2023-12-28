import streamlit as st
import streamlit.components.v1 as components
import pyautogui
from pathlib import Path

# run with streamlit run stoer_wagner_visualization.py

# Set header title
st.title('Network Graph Visualization of Stoer Wagner Algorithm')

contractions = [('jqt', 'rhn'), ('rsh', 'rzs'), ('frs', 'qnr'), ('xhk', 'ntq'), ('xhk', 'jqt'), ('lsr', 'pzl'), ('lsr', 'rsh'), ('nvd', 'lhk'), ('lsr', 'frs'), ('xhk', 'hfx'), ('lsr', 'nvd'), ('xhk', 'bvb'), ('lsr', 'cmg'), ('xhk', 'lsr')]
path = 'graph'
pages = {}
page_index = {}
pages_list = []
for i, file in enumerate(sorted(Path('.').glob(f'{path}/*.html'))):
    name = f'iteration/graph: {str(i).zfill(4)}'
    pages[name] = str(file)
    page_index[name] = i
    pages_list.append(name)

def open_page(page_name: str) -> None:
    st.header(page_name)
    index = page_index[page_name]
    if index > 0:
        st.header(f'Contraction: {contractions[index - 1]}')
        htmlFile = open(pages[pages_list[index - 1]], 'r', encoding = 'utf-8')
        components.html(htmlFile.read(), height = 400, width = 400)
    htmlFile = open(pages[page_name], 'r', encoding = 'utf-8')
    components.html(htmlFile.read(), height=400, width = 400)

# create a button in the side bar that will move to the next page/radio button choice
next = st.sidebar.button('NEXT SLIDE')

choice = st.sidebar.radio("GO TO", tuple(sorted(pages.keys())))
if next:
    pyautogui.press("tab")
    pyautogui.press("down")
else:
    # finally get to whats on each page
    open_page(choice)
