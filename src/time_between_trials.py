#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 12:42:13 2020

@author: Melisa
"""

import numpy as np
from datetime import datetime
import pandas as pd

path = '/home/sebastian/Documents/Melisa/calcium_imaging_analysis/references/analysis/calcium_analysis_checked_videos.xlsx'


table = pd.read_excel(path)

x = table['timestamp']

time = []
for i in range(0,len(table),2):
    string = str(int(x[i]))
    time_new = int(string[:-4])*360 + int(string[-4:-2])*60 + int(string[-2:])
    time.append(time_new)


difference = np.diff(time)
difference2 = difference[difference > 0 ]
differece2 = difference2/60

