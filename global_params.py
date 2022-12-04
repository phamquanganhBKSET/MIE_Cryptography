import numpy as np
from rig.type_casts import float_to_fp

K       = 8 # Number of images
k2      = 8 # Word length in bits of kP0 and kC0

# Function: Convert unsigned floating point number to unsigned fixed point number 
# width word length = k2 bits, fractional length = 0
fl2fx = float_to_fp(signed = False, n_bits = k2, n_frac = 0)

kP0     = np.asarray([[64 ], [154], [37], [73], [17], [56 ], [72], [68]]) # Initial value
kC0     = np.asarray([[123], [11 ], [27], [88], [33], [211], [97], [63]]) # Initial value
kP_MN   = np.asarray([[64 ], [154], [37], [73], [17], [56 ], [72], [68]]) # Initial value
M       = 256
N       = 256
Ne      = 5
NamePCM = 'Cat' # Type of PCM to use
