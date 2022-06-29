# -*- coding: utf-8 -*-

from os import path
import sqlite3
import re
import pandas as pd
import uuid

#-------------------- CONST --------------------
NL = '\n'
SQL_MAX_ROWS = 10000

SQL_str_replacements = {"'": "’"}

SQL_CLEAR =\
"""
DROP TABLE IF EXISTS projects;
DROP TABLE IF EXISTS professions;
DROP TABLE IF EXISTS names;
"""

SQL_CREATE =\
"""
DROP TABLE IF EXISTS projects;
DROP TABLE IF EXISTS professions;
DROP TABLE IF EXISTS names;

CREATE TABLE projects (
    id_proj INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT,
    title TEXT DEFAULT '',
    year INTEGER DEFAULT NULL,
    url TEXT DEFAULT '', 
    language TEXT DEFAULT 'Русский',
    cred_start INTEGER DEFAULT -1,
    cred_end INTEGER DEFAULT -1,
    cred_sample INTEGER DEFAULT 1,
    status TEXT DEFAULT ''
);

CREATE TABLE professions (
    id_prof INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT DEFAULT '',
    prof TEXT DEFAULT ''
);
 
CREATE TABLE names (
    id_names INTEGER PRIMARY KEY AUTOINCREMENT,
    id_proj2 INTEGER REFERENCES projects(id_proj) ON UPDATE CASCADE ON DELETE SET NULL,
    id_prof2 INTEGER REFERENCES professions(id_prof) ON UPDATE CASCADE ON DELETE SET NULL,
    name TEXT DEFAULT ''
);
"""

SQL_GETPROJECTS =\
"""
select projects.title as "Название фильма", projects.year as "Год выпуска", projects.url as "Ссылка", 
    projects.cred_start as "Начало титров", projects.cred_end as "Конец титров", projects.status as "Статус"
from projects;
"""

#------------------------------------------------

def regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None

#------------------------------------------------

class tsqlite():
    """
    """
    def __init__(self, sqldbfile='', opendb=True, forcecreate=False):
        self.SQL_CON = None
        self.SQL_CUR = None        
        self.SQLDB = sqldbfile
        self.__chunkflag = False
        if opendb: self.opendb(forcecreate)  
    
    def __del__(self):
        self.closedb()
        
    def connect(self, forceconnect=False):
        if not forceconnect and self.check_conn(): return {}         
        self.SQL_CON = sqlite3.connect(self.SQLDB, check_same_thread=False)
        self.SQL_CON.create_function('REGEXP', 2, regexp)
        self.SQL_CUR = self.SQL_CON.cursor()  
        
    def check_conn(self):
        return bool(self.SQL_CUR)
    
    def commit(self):
        if self.SQL_CON: self.SQL_CON.commit() 
        
    def rollback(self):
        if self.SQL_CON: self.SQL_CON.rollback()  
    
    def recreate_db(self):
        self.opendb(True) 
        
    def clear_db(self):
        self.connect()
        self.SQL_CUR.executescript(SQL_CLEAR)
    
    def opendb(self, forcecreate=False):
        """
        """
        db_existed = path.isfile(self.SQLDB)
        self.connect(True)
        if not db_existed or forcecreate:               
            self.SQL_CUR.executescript(SQL_CREATE)
        
    def closedb(self, commitall=True):  
        """
        """            
        if self.SQL_CUR: 
            self.SQL_CUR.close()    # Закрываем объект-курсора
            self.SQL_CUR = None
            
        if self.SQL_CON: 
            if commitall:
                self.SQL_CON.commit()
            else:
                self.SQL_CON.rollback()
                
            self.SQL_CON.close()    # Закрываем соединение
            self.SQL_CON = None
        
    def sql_format_str(self, string):
        st = string
        for k, v in SQL_str_replacements.items():
            st = st.replace(k, v)
        return st     
    
    def exec_sql(self, sql, commit=False):
        """
        """
        self.connect()
        self.SQL_CUR.executescript(sql)
        if commit: self.SQL_CON.commit()
    
    def select(self, sql, maxrows=SQL_MAX_ROWS, newselect=True, fieldnames=True):
        """
        """   
        if newselect:
            self.__chunkflag = False        
            
        if not self.__chunkflag or maxrows < 0:
            self.connect()
            self.SQL_CUR.execute(sql)
            
        if maxrows < 0:
            res = self.SQL_CUR.fetchall()
            if fieldnames and self.SQL_CUR.description:
                res.insert(0, tuple(description[0] for description in self.SQL_CUR.description))
            return res
        
        res = self.SQL_CUR.fetchmany(maxrows) 
        if fieldnames and self.SQL_CUR.description and not self.__chunkflag:
            res.insert(0, tuple(description[0] for description in self.SQL_CUR.description))
        self.__chunkflag = bool(res)            
        return res

    def asdataframe(self, sql, startcol=0, endcol=None, columns=None):
        records = self.select(sql, fieldnames=True)
        if startcol is None or startcol < 0: 
            startcol = 0
        return pd.DataFrame.from_records(records[startcol:endcol] if len(records) > 1 else [], columns=columns or records[0])

    def _get_column_names(self, cur):
        return tuple(c.name for c in cur.description) if cur else tuple()

#------------------------------------------------

class Credb(tsqlite):

    def __init__(self):
        super().__init__('db/data.db')

    def get_projects_df(self):
        return self.asdataframe(SQL_GETPROJECTS, 1, columns=['UID', 'Название', 'Год', 'URL', 'Язык', 'Начало', 'Конец', 'Интервал', 'Статус'])

    def add_project(self, title, year, url, language, cred_start, cred_end, cred_sample, status):
        sql = 'insert into projects values (?, ?, ?, ?, ?, ?, ?, ?, ?)'
        self.SQL_CUR.execute(sql, (uuid.uuid4().hex, title, year, url, language, cred_start, cred_end, cred_sample, status))
        self.SQL_CON.commit()
