#!/usr/bin/env python

import numpy as np
import argparse
from ctypes import *
import sys

################################ Load library: ################################

lib_name = 'libhypot.so'
try:
    # try to use the one the OS finds (e.g. in /usr/local/lib)
    libhypot = CDLL(lib_name)
except OSError:
    # library probably wasn't installed; look in local dir instead:
    libhypot = CDLL('./' + lib_name)

############################## Parse input args: ##############################

parser = argparse.ArgumentParser(description='Compute hypotenuses for right triangles, ' +\
                                 'given lists of the other two sides.  '+\
                                 'Values will be treated as floats.  ' +\
                                 'Inputs should be ASCII files.')
parser.add_argument("A", metavar='A.txt', help="List of side 1 values")
parser.add_argument("B", metavar='B.txt', help="List of side 2 values")
parser.add_argument("C", metavar='C.txt', help="Output, list of hypotenuse values")
args = parser.parse_args()

############################# Load input file: ################################

A = np.loadtxt(args.A, dtype='float32')
A_p = A.ctypes.data_as( POINTER(c_float) )

B = np.loadtxt(args.B, dtype='float32')
B_p = B.ctypes.data_as( POINTER(c_float) )

assert len(A) == len(B)

########################### Prepare output array: #############################

C = np.zeros( len(A) ).astype('float32')
C_p = C.ctypes.data_as( POINTER(c_float) )

################################# Get result: #################################

retval = libhypot.gpuHypot( A_p, B_p, C_p, len(A) )
if retval:
    print("hypot() failed!")

# Results are already stored in C.

################################ Save to disk: ################################

np.savetxt(args.C, C)

#################################### Done. ####################################

sys.exit(retval)

###############################################################################
