import numpy as np
import os
"""
file=np.load('preVM.npy')
print("------------------------------------")
print("preVM:")
print(file)
print(file.shape)
print("------------------------------------")

print("------------------------------------")
print("preHostState:")
file=np.load('preHostStat.npy')
print(file)
print(file.shape)
print("------------------------------------")
file=np.load('record.npy')
print("------------------------------------")
print("record:")
print(file)
print("------------------------------------")
"""

file=np.load('replaybuffer_state.npy')
print(file)
file=np.load('replaybuffer_action.npy')
print(file)



