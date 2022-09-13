# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 09:13:37 2022

@author: gxq
"""

import os
filePath = os.path.dirname(os.path.realpath(__file__))
for file in os.listdir(filePath):
    if '.npy' in file:
        os.remove(file)
    if '.txt' in file:
        os.remove(file)
    if '.pkl' in file:
        os.remove(file)