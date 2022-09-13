# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:47:38 2022

@author: gxq
"""

import os
import json
import numpy as np
filePath = os.path.dirname(os.path.realpath(__file__))
replaybuffer_stateFile = 'replaybuffer_state.npy'       
replaybuffer_stateFile_path = os.path.join(filePath,replaybuffer_stateFile)
print(replaybuffer_stateFile_path)
rb_nextstate=list(np.load(replaybuffer_stateFile_path))
print(len(rb_nextstate))
#print(rb_nextstate)