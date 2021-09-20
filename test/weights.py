import sys
import os
import numpy as np

SIMULATOR_PATH="/i3c/hpcl/sms821/Research/SpikSim/SpikingNN/PUMA/puma-simulator"
sys.path.insert (0, SIMULATOR_PATH + '/src/')
sys.path.insert (0, SIMULATOR_PATH + '/include/')
sys.path.insert (0, SIMULATOR_PATH +'/')

THIS_PATH = os.getcwd()

from src.data_convert import *
import config as cfg

# conv params
kh, kw = 3, 3
ih, iw = 14, 14
ic, oc = 32, 64

# fc params
in_sz, out_sz = 5, 5

def create_conv_wts(filename):
    fh = open(filename, "w")
    #weights = 2*np.random.randn(oc, ic, kh, kw) - 1.0
    weights = 5*np.random.randn(oc, ic, kh, kw)
    for outC in range(oc):
        for inC in range(ic):
            for kH in range(kh):
                for kW in range(kw):
                    temp_val = float2fixed(weights[outC][inC][kH][kW], cfg.int_bits, cfg.frac_bits)
                    temp_val = fixed2float(temp_val, cfg.int_bits, cfg.frac_bits)
                    fh.write(str(temp_val) + ' ')
                fh.write("\n")
            fh.write("\n")
        fh.write("\n")
        fh.write("\n")

def create_fc_wts(filename):
    fh = open(filename, "w")
    weights = np.random.randn(out_sz, in_sz)
    for o in range(out_sz):
        for i in range(in_sz):
            temp_val = float2fixed(weights[o][i], cfg.int_bits, cfg.frac_bits)
            temp_val = fixed2float(temp_val, cfg.int_bits, cfg.frac_bits)
            fh.write(str(temp_val) + ' ')
        fh.write("\n")

def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ('conv', 'fc'):
        print('Eg usage: python weights.py conv/fc <filename>.txt')
        exit()
    if sys.argv[1] == 'conv':
        create_conv_wts(sys.argv[2]+'.txt')
    else:
        create_fc_wts(sys.argv[2]+'.txt')

main()
