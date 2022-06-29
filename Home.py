# -*- coding: utf-8 -*-
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import pandas as pd
import numpy as np
import os
from PIL import Image
# import qtawesome as qta
# from common import utils
from db import dbsql
from parser import parser
from common import utils

#------------------------------------------------

@st.experimental_singleton
def dbsession():
    return dbsql.Credb()

@st.experimental_memo
def get_extractor():
    if not st.session_state.get('txt_url', ''):
        return None
    return parser.Extractor(st.session_state['txt_url'], parser.DATA_DIR)

def create_project_menu(container):
    if not st.session_state.get('project_mode', ''):
        return
    with container:
        proj_cols1 = st.columns(2)

        with proj_cols1[0]:          
            uploaded_file = st.file_uploader('Перетащите файл сюда', type=['mp4', 'flv', 'mpg', 'avi', 'mkv'], key='file_upload', 
                                             help='Видеофайлы до 4 ГБ (*.mp4, *.avi, *.flv, *.mpg, *.mkv)')
            if not uploaded_file is None:
                temp_path = os.path.join(parser.DATA_DIR, utils.make_uid(), os.path.splitext(uploaded_file.name)[1])
                try:
                    with open(temp_path, 'wb') as videofile:
                        videofile.write(uploaded_file.read())
                    st.session_state['txt_url'] = temp_path
                except Exception as err:
                    st.exception(err)
                    st.stop()               
            st.text_input('URL', key='txt_url')

        video_object = get_extractor()

        with proj_cols1[1]:
            st.text_input('Название', key='txt_name')            
            st.selectbox('Язык', ['Русский', 'Английский'], key='txt_language', index=0)
            st.number_input('Год', key='num_year', min_value=1900, max_value=utils.current_dt().year, value=utils.current_dt().year)

        if video_object:
            dur = int(video_object.duration)
            st.video(st.session_state['txt_url'])
            st.caption(f'{video_object.frame_width}x{video_object.frame_height}, {video_object.fps} FPS, {dur} сек.')

            proj_cols2 = st.columns(3)
            with proj_cols2[0]:
                st.number_input('Начало', key='num_start_sec', min_value=0, max_value=dur, value=0)
            with proj_cols2[1]:
                st.number_input('Конец', key='num_end_sec', min_value=0, max_value=dur, value=dur)
            with proj_cols2[2]:
                st.number_input('Интервал', key='num_sample_sec', min_value=0.5, max_value=10.0, value=1.0, step=0.5)

            proj_cols2 = st.columns(2)
            with proj_cols2[0]:
                st.button('OK', key='btn_proj_submit', on_click=on_projectdata_submit)
            with proj_cols2[1]:
                st.button('Отмена', key='btn_proj_cancel', on_click=on_projectdata_cancel)
        else:
            st.button('Отмена', key='btn_proj_cancel', on_click=on_projectdata_cancel)    

def on_btn_start():
    pass

def on_btn_add(container):
    st.session_state['project_mode'] = 'add'
    create_project_menu(container)

def on_btn_delete():
    pass

def on_projectdata_submit():
    if not st.session_state.get('project_mode', ''):
        return

    pass

    del st.session_state['project_mode']
    st.experimental_rerun()

def on_projectdata_cancel():
    if not st.session_state.get('project_mode', ''):
        return
    del st.session_state['project_mode']
    st.experimental_rerun()

#------------------------------------------------

imlogo = Image.open('resource/logo.png')

st.set_page_config('ZOOM FILM', imlogo, 'wide', 'auto')

display = np.array(imlogo)

cols = st.columns(5)
cols[0].image(display, width=50)
cols[1].markdown('## ZOOM FILM')

btn_cols = st.columns(5)
project_edit_form = st.container()

with btn_cols[0]:
    st.button('Запустить парсинг', key='btn_start', on_click=on_btn_start)
with btn_cols[1]:
    st.button('Добавить проект', key='btn_add', on_click=on_btn_add, args=(project_edit_form,))

# посмотреть в БД и вывести список проектов
st.markdown('### Список проектов')

db = dbsession()
df_projects = db.get_projects_df()
gb = GridOptionsBuilder.from_dataframe(df_projects)
gb.configure_pagination()
gb.configure_side_bar()
# gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='count', editable=False)
gb.configure_column('UID', hide=True)
gb.configure_selection(selection_mode='multiple', use_checkbox=False)
gridOptions = gb.build()

if not 'grid' in st.session_state:
    st.session_state['grid'] = None
grid_data = AgGrid(df_projects, key='grid', gridOptions=gridOptions, enable_enterprise_modules=True, 
                    update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED, 
                    theme='dark')
selected_rows = pd.DataFrame(grid_data['selected_rows'])

# если есть выделенные строки...
sel_length = len(selected_rows)
if sel_length > 0:
    with btn_cols[2]:
        st.button('Удалить', key='btn_delete', on_click=on_btn_delete)

# если выделена 1 строка - отобразить данные проекта
if sel_length == 1:
    proj_data = selected_rows[0]
    st.session_state['txt_url'] = proj_data['URL']
    st.session_state['txt_name'] = proj_data['Название']
    st.session_state['txt_language'] = proj_data['Язык']
    st.session_state['num_year'] = proj_data['Год']
    st.session_state['num_start_sec'] = proj_data['Начало']
    st.session_state['num_end_sec'] = proj_data['Конец']
    st.session_state['num_year'] = proj_data['Интервал']
    st.session_state['num_sample_sec'] = 'edit'
    create_project_menu(project_edit_form)