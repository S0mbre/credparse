# -*- coding: utf-8 -*-
# import numpy as np 
# import pandas as pd 
import os

def absp(fpath):
    if not os.path.isabs(fpath):
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), fpath)
    return fpath