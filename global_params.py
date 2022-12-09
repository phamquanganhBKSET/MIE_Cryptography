import numpy as np

K       = 8 # Number of images
k2      = 8 # Word length in bits of kP0 and kC0
kP0     = np.array([[64 ], [154], [37], [73], [17], [56 ], [72], [68]], dtype=np.ubyte) # Initial value
kC0     = np.array([[123], [11 ], [27], [88], [33], [211], [97], [63]], dtype=np.ubyte) # Initial value
kP_MN   = np.array([[64 ], [154], [37], [73], [17], [56 ], [72], [68]], dtype=np.ubyte) # Initial value
M       = 8
N       = 8
Ne      = 5
NamePCM = 'Cat' # Type of PCM to use
