import numpy as np
import torch
import pandas as pd
import os
from Network import Network
from STDPsynapses import STDPSynapse, LIFNeuronGroup
from struct import unpack
from plotting import plotWeights


path = os.path.join(os.getcwd(),'Weight evolution')
#path = os.path.join(os.curdir,'Weight evolution')


print(path)



