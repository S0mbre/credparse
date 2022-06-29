# -*- coding: utf-8 -*-
# import numpy as np 
# import pandas as pd 
import os
import uuid
from datetime import datetime

def absp(fpath):
    if not os.path.isabs(fpath):
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), fpath)
    return fpath

def make_uid():
    return uuid.uuid4().hex

def current_dt():
    return datetime.now()