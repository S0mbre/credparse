# -*- coding: utf-8 -*-
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
# import numpy as np
import pandas as pd
import qtawesome as qta
from . import utils
from db import dbsql

#------------------------------------------------

@st.experimental_singleton
def dbsession():
    return dbsql.Credb()

def on_btn_start():
    pass

def on_btn_add():
    pass

#------------------------------------------------

def app():
    st.markdown('### Список проектов')

    btn_cols = st.columns(5)
    with btn_cols[0]:
        st.button('Запустить парсинг', on_click=on_btn_start)
    with btn_cols[1]:
        st.button('Добавить проект', on_click=on_btn_add)

    project_edit_cont = st.container()
    
    # посмотреть в БД и вывести список проектов
    db = dbsession()
    df_projects = db.get_projects_df()
    gb = GridOptionsBuilder.from_dataframe(df_projects)
    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='count', editable=True)
    gb.configure_selection(selection_mode='multiple', use_checkbox=False)
    gridOptions = gb.build()
    grid_data = AgGrid(df_projects, key='grid', gridOptions=gridOptions, enable_enterprise_modules=True, 
                       update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED, 
                       theme='dark')
    selected_rows = pd.DataFrame(grid_data['selected_rows'])

    if btn_cols[0].button('Запустить парсинг'):
        pass

    if btn_cols[1].button('Добавить проект'):
        pass
