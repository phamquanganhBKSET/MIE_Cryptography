import math
import numpy as np
import global_params
import cat_params
from fxpmath import Fxp

# Function: XOR two binary strings
# a, b: Operators (string)
# Return: a XOR b (string)
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
# kC: Cipher text
# kI: Plain text
# output_size: Size of output list
# Return: Bit string after Bit Manipulation
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

# Function: Bit arrangement 1D to nD - from 1D matrix to nD matrix
# Y: Rule of bit arrangement (list of numpy arrays)
# B: Source bit-string array (string array-list of strings)
# Return: Destination bit-string array (string array or list of strings)
def bit_rearrangement_1d_to_nd(Y: list, B) -> list:
    # Flip left/right verison of B
    B_fliplr = B.copy()
    for i in range(len(B_fliplr)):
        B_fliplr[i] = B[i][::-1] # [start:stop:step]

    matrix_B = [['']]*len(B_fliplr)
    for i in range(len(matrix_B)):
        matrix_B[i] = [*B_fliplr[i]] # Unpack string to a list

    # Y.shape
    width_Y = Y[0].shape[1]
    depth_Y = len(Y)

    A1 = [[''] * width_Y] * depth_Y # size = (depth_Y, width_Y): # [['', '',..., ''], ['', '',..., ''],..., ['', '',..., '']]

    for j in range(width_Y):
        for k in range(depth_Y):
            if ((Y[k][0][j] == 0) & (Y[k][1][j] == 0)):
                A1[k][j] = '0'
            elif ((Y[k][0][j] == 100) & (Y[k][1][j] == 100)):
                A1[k][j] = '1'
            else:
                A1[k][j] = matrix_B[Y[k][0][j]-1][Y[k][1][j]-1]

    # Result: New string array (list of string)
    A = []
    for k in range(depth_Y):
        A.append(''.join(A1[k]))

    return A

# Function: Bit arrangement MIE nD
# Y: Rule of bit arrangement (list of numpy arrays)
# B: Source bit-string array (string array or list of strings)
# Return: Destination bit-string array (string array or list of strings)
def bit_rearrangement_MIE_nd(Y: list, B) -> list:
    #Size of B
    height_B = len(B)
    matrix_B = [['']] * height_B # [[''], [''],..., ['']]
    for i in range(height_B):
        matrix_B[i] = [*B[i]] # List of bits
    
    # Y.shape
    width_Y = Y[0].shape[1]
    depth_Y = len(Y)

    A1 = [[''] * width_Y] * depth_Y # size = (depth_Y, width_Y): # [['', '',..., ''], ['', '',..., ''],..., ['', '',..., '']]

    for j in range(width_Y):
        for k in range(depth_Y):
            if ((Y[k][0][j] == 0) & (Y[k][1][j] == 0)):
                A1[k][j] = '0'
            else:
                A1[k][j] = matrix_B[Y[k][0][j]-1][Y[k][1][j]-1]

    # Result: New string array (list of string)
    A = []
    for k in range(depth_Y):
        A.append(''.join(A1[k]))

    return A


# Function: Cat chaostic map
# pq: gamma (fixed-point numbers array)
# xy: Xn (fixed-point numbers array - numpy array with dtype = Fxp)
# N: Number of bits to present xy (float)
# Return: Numpy array with dtype = Fxp
def Cat_fi(pq, xy: np.ndarray, N):
    xy_out = np.copy(xy)
    xy_out[0][0] = Fxp(xy[0][0] + pq[0][0]*xy[1][0] - math.floor(xy[0][0] + pq[0][0]*xy[1][0]),0, N, N-1)
    xy_out[1][0] = Fxp(pq[1][0]*xy[0][0]+(pq[0][0]*pq[1][0]+1)*xy[1][0] - math.floor(pq[1][0]*xy[0][0]+(pq[0][0]*pq[1][0]+1)*xy[1][0]),0, N, N-1)
    return xy_out

# Function: PCM Cat map
# E: Result after Bit Manipulation (string)
# R: Number of iterations (integer)
# Return: Numpy array with dtype = Fxp
def PCM_Cat(E: str, R):
    delta_gamma = bit_rearrangement_1d_to_nd(cat_params.Y1_FAST_Cat, [E])
    gamma_tmp = np.copy(cat_params.Gamma0_Cat)
    for i in range(len(gamma_tmp)):
        gamma_tmp_bin = xor(gamma_tmp[i][0].bin(), delta_gamma[i])
        gamma_tmp[i][0] = Fxp('0b' + gamma_tmp_bin, False, cat_params.m2_cat, cat_params.m2_cat - 6)

    delta_X = bit_rearrangement_1d_to_nd(cat_params.Y3_FAST_Cat, [E])
    Xn_tmp = np.copy(cat_params.IV0_Cat)
    for i in range(len(Xn_tmp)):
        Xn_tmp_bin = xor(Xn_tmp[i][0].bin(), delta_X[i])
        Xn_tmp[i][0] = Fxp('0b' + Xn_tmp_bin, False, cat_params.m1_cat, cat_params.m1_cat - 1)

    for r in range(R):
        X_r = Cat_fi(gamma_tmp, Xn_tmp, cat_params.m1_cat)
        Xn_tmp = X_r

        delta_gamma = bit_rearrangement_MIE_nd(cat_params.Y2_FAST_Cat, [X_r[0][0].bin(), X_r[1][0].bin()])
        for i in range(len(gamma_tmp)):
            gamma_tmp_bin = xor(gamma_tmp[i][0].bin(), delta_gamma[i])
            gamma_tmp[i][0] = Fxp('0b' + gamma_tmp_bin, False, cat_params.m2_cat, cat_params.m2_cat - 6)

        delta_X = bit_rearrangement_MIE_nd(cat_params.Y4_FAST_Cat, [X_r[0][0].bin(), X_r[1][0].bin()])
        for i in range(len(Xn_tmp)):
            Xn_tmp_bin = xor(Xn_tmp[i][0].bin(), delta_X[i])
            Xn_tmp[i][0] = Fxp('0b' + Xn_tmp_bin, False, cat_params.m1_cat, cat_params.m1_cat - 1)
    
    return X_r

# Function: 
# Xn: 
# Yp_MN: 
# Y_inter_images_p: Number of images at the input
# NamePCM: 
# Return: 1. XY_new: 
#         2. pseudoVal_string_C: 
#         3. pseudoVal_string_Cx: 
def MIE_FAST_XYnew_pseudoVal(Xn, XY, Yp_MN, Y_inter_images_p, NamePCM = 'Cat'):
    if (NamePCM == 'Cat'):
        Yd_C = np.copy(cat_params.Yd_C_Cat)
        Yd_Cx = np.copy(cat_params.Yd_Cx_Cat)
    
    pseudoVal_string_C = bit_rearrangement_MIE_nd(Yd_C, [Xn[0][0].bin(), Xn[1][0].bin()])
    pseudoVal_string_Cx = bit_rearrangement_MIE_nd(Yd_Cx, [Xn[0][0].bin(), Xn[1][0].bin()])

    # Xn.shape
    height_Xn = Xn.shape[0]
    matrix_Xn = [['']] * height_Xn # [[''], [''],..., ['']]
    for i in range(height_Xn):
        matrix_Xn[i] = [*Xn[i][0].bin()] # List of bits

    # Yp_MN.shape
    width_Yp_MN = Yp_MN[0].shape[1]

    A1 = [[''] * width_Yp_MN] * global_params.K # size = (global_params.K, width_Yp_MN): # [['', '',..., ''], ['', '',..., ''],..., ['', '',..., '']]

    for j in range(width_Yp_MN):
        for k in range(global_params.K):
            A1[k][j] = matrix_Xn[Yp_MN[k][0][j]-1][Yp_MN[k][1][j]-1]

    # Y_inter_images_p.shape
    width_Y_inter_images_p = Y_inter_images_p[0].shape[1]

    A2 = [[''] * width_Y_inter_images_p] * global_params.K # size = (global_params.K, width_Y_inter_images_p): # [['', '',..., ''], ['', '',..., ''],..., ['', '',..., '']]

    for j in range(width_Y_inter_images_p):
        for k in range(global_params.K):
            A2[k][j] = matrix_Xn[Y_inter_images_p[k][0][j]-1][Y_inter_images_p[k][1][j]-1]

    # XYnew
    XY_new = []

    image_k = Fxp(0, 0, math.log2(global_params.K), 0)

    for k in range(global_params.K):
        X_new = Fxp('0b' + ''.join(A1[k][0:math.log2(global_params.M)]), 0, math.log2(global_params.M), 0)
        X_new = X_new + 1
        Y_new = Fxp('0b' + ''.join(A1[k][math.log2(global_params.M):]), 0, math.log2(global_params.N), 0)
        Y_new = X_new + 1

        image_k = image_k + 1 # if image_k = k -> Permutation in internal of image
        

# Function: 
# kI: 
# XY: 
# XY_new: 
# pseudoVal_string_C: 
# pseudoVal_string_Cx: 
# kC_minus: 
# Return: 1. kC_ij: 
#         2. kP_plus: 
#         3. kI: 
def MIE_FAST_Perm_and_Diff_pixel(kI, XY, XYnew, pseudoVal_string_C, pseudoVal_string_Cx, kC_minus):
    pass