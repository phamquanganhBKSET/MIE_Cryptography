import math
import global_params
import cat_params
from fxpmath import Fxp

# Function: XOR two binary strings
def xor(a, b):
    ans = ""
     
    # Loop to iterate over the
    # Binary Strings
    for i in range(len(a)):
         
        # If the Character matches
        if (a[i] == b[i]):
            ans += "0"
        else:
            ans += "1"
    return ans

# Function: MIE Bit Manipulation
def MIE_Bit_Manipulation(kC, kP, output_size):
    E1 = ''
    E2 = ''
    for k in range(global_params.K):
        kP_fxp = Fxp(kP[k][0], False, global_params.k2, 0)   # (val, signed, n_word, n_frac)
        kC_fxp = Fxp(kC[k][0], False, global_params.k2, 0)   # (val, signed, n_word, n_frac)
        E1 = E1 + kC_fxp.bin()
        E2 = E2 + kP_fxp.bin()
    
    T = ''
    for i in range(len(E2)):
        T = T + E1[i] + E2[i]

    times = math.ceil(len(T) * 1.0 / output_size)
    if times < 1: # |E1| + |E2| < |E|
        m = math.ceil(output_size * 1.0 / len(T)) # (m-1) * |T| < |E| < m * |T|
        T = T * int(m)
    
    # Number of bit 0 needed to padd to Tn = |E| - |Tn|
    zero_length_pad = output_size * math.ceil(len(T) * 1.0 / output_size) - len(T) * 1.0
    str_zeros = '0' * int(zero_length_pad)
    T = T + str_zeros
    E = T[0:output_size]
    for i in range(1, times):
        E = xor(E, T[i*output_size:(i+1)*output_size])
    return E

# Function: Bit arrangement
# Y: List of numpy array
# B: String
def bit_arrangement(Y: list, B: str):
    B_fliplr = B[::-1] # [start:stop:step]
    matrix_B = B_fliplr.split('')
    heigh_Y = Y[0].shape[1]
    width_Y = Y[0].shape[0]
    depth_Y = len(Y)

    A1 = []

    for j in range(width_Y):
        for k in range(depth_Y):
            if ((Y[k][0][j] == 0) & (Y[k][1][j] == 0)):
                A1[k][j] = '0'
            elif ((Y[k][0][j] == 100) & (Y[k][1][j] == 100)):
                A1[k][j] = '1'
            else:
                A1[k][j] = matrix_B[Y[k][0][j]][Y[k][1][j]]

# Function: PCM Cat map
def PCM_Cat(E, R):
    pass
