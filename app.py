# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
from PIL import Image
# from pages import utils
from multipage import MultiPage
from pages import home

#------------------------------------------------

# Create an instance of the app 
app = MultiPage()

display = np.array(Image.open('resource/logo.png'))

cols = st.columns(5)
cols[0].image(display, width=50)
cols[1].markdown('## ZOOM FILM')

# Add all your application here
app.add_page('Список проектов', home.app)

# The main app
app.run()
