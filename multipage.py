# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
from PIL import Image

#------------------------------------------------

class MultiPage: 

    def __init__(self):
        st.set_page_config('ZOOM FILM', Image.open('resource/logo.png'), 'wide', 'auto')
        self.pages = []
    
    def add_page(self, title, func): 
        self.pages.append({'title': title, 'func': func})

    def run(self):
        page = st.sidebar.selectbox('Навигация', self.pages, index=0, format_func=lambda page: page['title'])
        if page:
            page['func']()