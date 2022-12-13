import math
import numpy as np
import global_params
import cat_params
from fxpmath import Fxp


# Function: XOR two binary strings
# a, b: Operators (string)
# Return: a XOR b (string)
def xor(a, b):
    if (len(a) != len(b)):
        print("BIT XOR ERROR")
        return None
    
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
# kC: Cipher text (Type: the same as kC0)
# kP: Plain text (Type: the same as kP0)
# output_size: Size of output list
# Return: Bit string after Bit Manipulation
def MIE_Bit_Manipulation(kC, kP, output_size):
    E1 = ''
    E2 = ''
    # kP_fxp = Fxp(0, False, global_params.k2, 0)   # (val, signed, n_word, n_frac)
    # kC_fxp = Fxp(0, False, global_params.k2, 0)   # (val, signed, n_word, n_frac)

    for k in range(global_params.K):
        # kP_fxp.set_val(kP[k][0])
        # kC_fxp.set_val(kC[k][0])
        E1 = E1 + np.binary_repr(kC[k][0], width = global_params.k2)
        E2 = E2 + np.binary_repr(kP[k][0], width = global_params.k2)
    
    T = ''
    for i in range(len(E2)):
        T = T + E2[i] + E1[i]

    times = math.ceil(len(T) * 1.0 / output_size)
    E = ''
    if times <= 1.0: # |E1| + |E2| < |E|
        m = math.ceil(output_size * 1.0 / len(T)) # (m-1) * |T'| < |E| < m * |T'|
        T = T * int(m) # T = T' * m

        # Split E into n bit sequences
        n = math.ceil(len(T) * 1.0 / output_size) # (n-1) * |E| < |T| < n * |E|

        # Number of bit 0 needed to padd to Tn = |E| - |Tn|
        if ((len(T) % output_size) != 0):
            zero_length_pad = output_size * math.ceil(len(T) * 1.0 / output_size) - len(T) * 1.0
            str_zeros = '0' * int(zero_length_pad)
            T = T + str_zeros

        E = T[0:output_size]
        for i in range(1, n):
            E = xor(E, T[i*output_size:(i+1)*output_size])
        
    else: # |E1| + |E2| > |E|
        # Split E into n bit sequences
        n = math.ceil(len(T) * 1.0 / output_size) # (n-1) * |E| < |T| < n * |E|

        # Number of bit 0 needed to padd to Tn = |E| - |Tn|
        if ((len(T) % output_size) != 0):
            zero_length_pad = output_size * math.ceil(len(T) * 1.0 / output_size) - len(T) * 1.0
            str_zeros = '0' * int(zero_length_pad)
            T = T + str_zeros

        E = T[0:output_size]
        for i in range(1, n):
            E = xor(E, T[i*output_size:(i+1)*output_size])
        
    return E


# Function: Bit arrangement 1D to nD - from 1D matrix to nD matrix
# Y: Rule of bit arrangement (list of numpy arrays)
# B: Source bit-string array (string array-list of strings)
# Return: Destination bit-string array (string array or list of strings)
def bit_rearrangement_1d_to_nd(Y: list, B) -> list:
    # Flip left/right verison of B
    B_fliplr = B.copy()
    # for i in range(len(B_fliplr)):
    #     B_fliplr[i] = B[i][::-1] # [start:stop:step]

    # matrix_B = [['']]*len(B_fliplr)
    matrix_B = []
    for i in range(len(B_fliplr)):
        # matrix_B[i] = [*B_fliplr[i]] # Unpack string to a list
        matrix_B.append([*B_fliplr[i]]) # Unpack string to a list

    # Y.shape
    width_Y = Y[0].shape[1]
    depth_Y = len(Y)

    # A1 = [[''] * width_Y] * depth_Y # size = (depth_Y, width_Y): # [['', '',..., ''], ['', '',..., ''],..., ['', '',..., '']]

    # for k in range(depth_Y):
    #     for j in range(width_Y):
    #         if ((Y[k][0][j] == 0) & (Y[k][1][j] == 0)):
    #             A1[k][j] = '0'
    #         elif ((Y[k][0][j] == 100) & (Y[k][1][j] == 100)):
    #             A1[k][j] = '1'
    #         else:
    #             A1[k][j] = matrix_B[Y[k][0][j]-1][Y[k][1][j]-1]

    A1 = []
    temp_A1 = []

    for k in range(depth_Y):
        for j in range(width_Y):
            if ((Y[k][0][j] == 0) & (Y[k][1][j] == 0)):
                temp_A1.append('0')
            elif ((Y[k][0][j] == 100) & (Y[k][1][j] == 100)):
                temp_A1.append('1')
            else:
                temp_A1.append(str(matrix_B[Y[k][0][j]-1][Y[k][1][j]-1]))
        A1.append(temp_A1)
        temp_A1 = []

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
    # matrix_B = [] * height_B # [[''], [''],..., ['']]
    matrix_B = []
    for i in range(height_B):
        # matrix_B[i] = [*B[i]] # List of bits
        matrix_B.append([*B[i]]) # List of bits
    
    # Y.shape
    width_Y = Y[0].shape[1]
    depth_Y = len(Y)

    # A1 = [[''] * width_Y] * depth_Y # size = (depth_Y, width_Y): # [['', '',..., ''], ['', '',..., ''],..., ['', '',..., '']]

    # for k in range(depth_Y):
    #     for j in range(width_Y):
    #         if ((Y[k][0][j] == 0) & (Y[k][1][j] == 0)):
    #             A1[k][j] = '0'
    #         elif ((Y[k][0][j] == 100) & (Y[k][1][j] == 100)):
    #             A1[k][j] = '1'
    #         else:
    #             A1[k][j] = matrix_B[Y[k][0][j]-1][Y[k][1][j]-1]

    A1 = []
    temp_A1 = []

    for k in range(depth_Y):
        for j in range(width_Y):
            if ((Y[k][0][j] == 0) & (Y[k][1][j] == 0)):
                temp_A1.append('0')
            elif ((Y[k][0][j] == 100) & (Y[k][1][j] == 100)):
                temp_A1.append('1')
            else:
                temp_A1.append(str(matrix_B[Y[k][0][j]-1][Y[k][1][j]-1]))
        A1.append(temp_A1)
        temp_A1 = []

    # Result: New string array (list of string)
    A = []
    for k in range(depth_Y):
        A.append(''.join(A1[k]))

    return A


# Function: Cat chaotic map
# pq: gamma (fixed-point numbers array)
# xy: Xn (fixed-point numbers array - numpy array with dtype = Fxp)
# N: Number of bits to present xy (float)
# Return: Numpy array with dtype = Fxp
def Cat_fi(pq, xy: np.ndarray, N):
    xy_out = np.copy(xy)
    xy_out[0][0] = Fxp(xy[0][0] + pq[0][0]*xy[1][0] - math.floor(xy[0][0] + pq[0][0]*xy[1][0]), False, N, N-1)
    xy_out[1][0] = Fxp(pq[1][0]*xy[0][0] + (pq[0][0]*pq[1][0] + 1)*xy[1][0] - math.floor(pq[1][0]*xy[0][0]+(pq[0][0]*pq[1][0] + 1)*xy[1][0]), False, N, N-1)
    return xy_out


# Function: PCM Cat map
# E: Result after Bit Manipulation (string)
# Y1_FAST_Cat: Rule of bit arrangement (list of numpy arrays)
# Y2_FAST_Cat: Rule of bit arrangement (list of numpy arrays)
# Y3_FAST_Cat: Rule of bit arrangement (list of numpy arrays)
# Y4_FAST_Cat: Rule of bit arrangement (list of numpy arrays)
# R: Number of iterations (integer)
# Return: Numpy array with dtype = Fxp
def PCM_Cat(E: str, Y1_FAST_Cat, Y2_FAST_Cat, Y3_FAST_Cat, Y4_FAST_Cat, R):
    delta_gamma = bit_rearrangement_1d_to_nd(Y1_FAST_Cat, [E])
    gamma_tmp = np.copy(cat_params.Gamma0_Cat)
    for i in range(len(gamma_tmp)):
        gamma_tmp_bin = xor(gamma_tmp[i][0].bin(), delta_gamma[i])
        gamma_tmp[i][0] = Fxp('0b' + gamma_tmp_bin, False, cat_params.m2_cat, cat_params.m2_cat - 6)

    delta_X = bit_rearrangement_1d_to_nd(Y3_FAST_Cat, [E])
    Xn_tmp = np.copy(cat_params.IV0_Cat)
    for i in range(len(Xn_tmp)):
        Xn_tmp_bin = xor(Xn_tmp[i][0].bin(), delta_X[i])
        Xn_tmp[i][0] = Fxp('0b' + Xn_tmp_bin, False, cat_params.m1_cat, cat_params.m1_cat - 1)

    for r in range(R):
        X_r = Cat_fi(gamma_tmp, Xn_tmp, cat_params.m1_cat)
        Xn_tmp = X_r

        delta_gamma = bit_rearrangement_MIE_nd(Y2_FAST_Cat, [X_r[0][0].bin(), X_r[1][0].bin()])
        for i in range(len(gamma_tmp)):
            gamma_tmp_bin = xor(gamma_tmp[i][0].bin(), delta_gamma[i])
            gamma_tmp[i][0] = Fxp('0b' + gamma_tmp_bin, False, cat_params.m2_cat, cat_params.m2_cat - 6)

        delta_X = bit_rearrangement_MIE_nd(Y4_FAST_Cat, [X_r[0][0].bin(), X_r[1][0].bin()])
        for i in range(len(Xn_tmp)):
            Xn_tmp_bin = xor(Xn_tmp[i][0].bin(), delta_X[i])
            Xn_tmp[i][0] = Fxp('0b' + Xn_tmp_bin, False, cat_params.m1_cat, cat_params.m1_cat - 1)
    
    return X_r


# Function: Find new XY to pass to Permutation and Diffusion (XYk for k = 1...K)
# Xn: X_R (numpy array with dtype = Fxp)
# XY: Current i and j (list [i, j])
# Yp_MN: To find the next i and j (list of numpy arrays)
# Y_inter_images_p: To find k (list of numpy arrays)
# Yd_C: To find pseudoVal_string_C
# Yd_Cx: To find pseudoVal_string_Cx
# Return: 1. XY_new: New position [(i', j'), (i'', j''),...]
#         2. pseudoVal_string_C: After permutation, source plain pixel is XORed with pseudoVal_string_C
#         3. pseudoVal_string_Cx: After permutation, destination plain pixel is XORed with pseudoVal_string_Cx
def MIE_FAST_XYnew_pseudoVal(Xn, XY, Yp_MN, Y_inter_images_p, Yd_C, Yd_Cx):
    pseudoVal_string_C = bit_rearrangement_MIE_nd(Yd_C, [Xn[0][0].bin(), Xn[1][0].bin()])
    pseudoVal_string_Cx = bit_rearrangement_MIE_nd(Yd_Cx, [Xn[0][0].bin(), Xn[1][0].bin()])

    # Xn.shape
    height_Xn = Xn.shape[0]
    # matrix_Xn = [['']] * height_Xn # [[''], [''],..., ['']]
    matrix_Xn = []
    for i in range(height_Xn):
        # matrix_Xn[i] = [*Xn[i][0].bin()] # List of bits
        matrix_Xn.append([*Xn[i][0].bin()]) # List of bits

    print("\nMIE_FAST_XYnew_pseudoVal.matrix_Xn: \n", matrix_Xn)

    # Yp_MN.shape
    width_Yp_MN = Yp_MN[0].shape[1]

    # XY_choose = [[''] * width_Yp_MN] * global_params.K # size = (global_params.K, width_Yp_MN): # [['', '',..., ''], ['', '',..., ''],..., ['', '',..., '']]
    XY_choose = []
    temp_XY_choose = []

    for k in range(global_params.K):
        for j in range(width_Yp_MN):
            # XY_choose[k][j] = str(matrix_Xn[Yp_MN[k][0][j]-1][Yp_MN[k][1][j]-1])
            temp_XY_choose.append(str(matrix_Xn[Yp_MN[k][0][j]-1][Yp_MN[k][1][j]-1]))
        XY_choose.append(temp_XY_choose)
        temp_XY_choose = []

    print("\nMIE_FAST_XYnew_pseudoVal.XY_choose: \n", XY_choose)

    # Y_inter_images_p.shape
    width_Y_inter_images_p = Y_inter_images_p[0].shape[1]

    # K_choose = [[''] * width_Y_inter_images_p] * global_params.K # size = (global_params.K, width_Y_inter_images_p): # [['', '',..., ''], ['', '',..., ''],..., ['', '',..., '']]
    K_choose = []
    temp_K_choose = []

    for k in range(global_params.K):
        for j in range(width_Y_inter_images_p):
            # K_choose[k][j] = str(matrix_Xn[Y_inter_images_p[k][0][j]-1][Y_inter_images_p[k][1][j]-1])
            temp_K_choose.append(str(matrix_Xn[Y_inter_images_p[k][0][j]-1][Y_inter_images_p[k][1][j]-1]))
        K_choose.append(temp_K_choose)
        temp_K_choose = []

    print("\nMIE_FAST_XYnew_pseudoVal.K_choose: \n", K_choose, "\n")

    # XYnew
    XY_new  = []
    X_new   = Fxp(0, False, int(math.log2(global_params.M)), 0)
    Y_new   = Fxp(0, False, int(math.log2(global_params.N)), 0)
    image_k = Fxp(0, False, int(math.log2(global_params.K)), 0)

    for k in range(global_params.K):
        X_new.set_val('0b' + ''.join(XY_choose[k][0:int(math.log2(global_params.M))]))
        # X_new = X_new + 1
        Y_new.set_val('0b' + ''.join(XY_choose[k][int(math.log2(global_params.M)):]))
        # Y_new = X_new + 1

        image_k.set_val('0b' + ''.join(K_choose[k][:]))
        # image_k = image_k + 1 # if image_k = k -> Permutation in internal of image

        XY_new_in_front_of_XY_1_pixel  = (((X_new == XY[0]) & (Y_new == XY[1] - 1))  | ((X_new == XY[0] - 1) & (Y_new == global_params.N - 1) & (XY[1] == 0)))
        XY_new_after_XY_1_pixel        = (((X_new == XY[0]) & (Y_new == XY[1] + 1))  | ((X_new == XY[0] + 1) & (Y_new == 0) & (XY[1] == global_params.N - 1)))
        XY_new_is_XY                   = ((X_new  == XY[0]) & (Y_new == XY[1]     ))
        XY_new_in_front_of_XY_2_pixels = ((X_new  == XY[0]) & (Y_new == XY[1] - 2 )) | ((X_new == XY[0] - 1) & (Y_new == global_params.N - 1) & (XY[1] == 1)) | ((X_new == XY[0] - 1) & (Y_new == global_params.N - 2) & (XY[1] == 0))
        XY_new_after_XY_2_pixels       = ((X_new  == XY[0]) & (Y_new == XY[1] + 2 )) | ((X_new == XY[0] + 1) & (Y_new == 0) & (XY[1] == global_params.N - 2)) | ((X_new == XY[0] + 1) & (Y_new == 1) & (XY[1] == global_params.N - 1))
        
        if ((image_k == k) & (XY_new_in_front_of_XY_1_pixel | XY_new_after_XY_1_pixel | XY_new_is_XY | XY_new_in_front_of_XY_2_pixels | XY_new_after_XY_2_pixels)):
            # if (X_new < global_params.M - 1):
            #     X_new = X_new + 1
            # else:
            #     X_new = X_new - 1
            if (image_k < global_params.K - 1):
                image_k = image_k + 1
            else:
                image_k = image_k - 1

        XY_new.append([np.uint16(X_new), np.uint16(Y_new), np.uint8(image_k)])

    return XY_new, pseudoVal_string_C, pseudoVal_string_Cx


# Function: MIE Permutation and Diffusion of Encryption processing
# kI: Images (list of image matrices)
# XY: Current i and j (list [i, j])
# XY_new: New position [(i', j'), (i'', j''),...]
# pseudoVal_string_C: After permutation, source plain pixel is XORed with pseudoVal_string_C
# pseudoVal_string_Cx: After permutation, destination plain pixel is XORed with pseudoVal_string_Cx
# kC_minus: kC- (the same type as kC0 = np.asarray([[123], [11 ], [27], [88], [33], [211], [97], [63]]))
# n: Current iteration's order
# Return: 1. kC_ij: Pixel's value used to impact chaotic map (the same type as kC_minus)
#         2. kP_plus: kP+
#         3. kI: Images after Permutation and Diffusion (list of image matrices)
def MIE_FAST_Perm_and_Diff_pixels_ENC(kI, XY, XY_new, pseudoVal_string_C, pseudoVal_string_Cx, kC_minus, n):
    i = XY[0]
    j = XY[1]

    for k in range(global_params.K):
        # Permutation
        temp = kI[k][i][j]
        kI[k][i][j] = kI[XY_new[k][2]][XY_new[k][0]][XY_new[k][1]]
        kI[XY_new[k][2]][XY_new[k][0]][XY_new[k][1]] = temp

        # Diffusion
        temp = kI[k][i][j] # Current pixel after Permutation
        temp_str = xor(np.binary_repr(temp, width = global_params.k2), np.binary_repr(kC_minus[k][0], width = global_params.k2)) # temp_value = I[i][j] XOR C[i-1][j]
        temp_str = xor(temp_str, pseudoVal_string_C[k]) # temp_value XOR pseudoVal_string_C (result of chaotic map)
        kI[k][i][j] = np.uint8(int(temp_str, 2))

        # The pixel permuted with current pixel is also diffused
        temp = kI[XY_new[k][2]][XY_new[k][0]][XY_new[k][1]]
        temp = xor(np.binary_repr(temp, width = global_params.k2), pseudoVal_string_Cx[k])
        kI[XY_new[k][2]][XY_new[k][0]][XY_new[k][1]] = np.uint8(int(temp, 2))

    # For the next pixel: Pass the diffused pixel's value to chaotic map
    kC_ij = np.zeros_like(global_params.kC0, dtype=np.uint8)
    for k in range(global_params.K):
        kC_ij[k][0] = kI[k][i][j]

    # Find kP_plus for the next pixel
    kP_plus = np.zeros((global_params.K, 1), dtype=np.uint8)
    if (i < global_params.M - 1): # This means i + 1 is not out of range (i + 1 <= M - 1)
        if (j < global_params.N - 2): # This means k[i][j+2] is inside image because j + 2 is not out of range (j + 2 <= N - 1)
            for k in range(global_params.K):
                kP_plus[k][0] = np.uint8(kI[k][i][j+2])
        else: # This means j + 2 is out of range (j + 2 > N - 1)
            for k in range(global_params.K):
                kP_plus[k][0] = np.uint8(kI[k][i+1][j-(global_params.N-1)+1]) # Go to the (j-(global_params.N-1)+1)th pixel of the next rows
    else: # This means i + 1 is out of range (i + 1 > M - 1)
        if (j < global_params.N - 2): # This means j + 2 is not out of range (j + 2 <= N - 1)
            for k in range(global_params.K):
                kP_plus[k][0] = np.uint8(kI[k][i][j+2])
        elif(j == global_params.N - 2): # This mean j + 2 == N
            if (n == global_params.Ne - 1): # The last iteration
                for k in range(global_params.K):
                    kP_plus[k][0] = np.uint8(global_params.kP0[k][0])
            else:
                kP_plus[k][0] = np.uint8(kI[k][0][0])

    return kC_ij, kP_plus, kI


# Function: MIE Permutation and Diffusion of Decryption processing
# kC: Cipher images (list of image matrices)
# XY: Current i and j (list [i, j])
# XY_new: New position [(i', j'), (i'', j''),...]
# pseudoVal_string_C: After permutation, source plain pixel is XORed with pseudoVal_string_C
# pseudoVal_string_Cx: After permutation, destination plain pixel is XORed with pseudoVal_string_Cx
# kC_minus: kC-
# n: Current iteration's order
# Return: 1. kC_m: kC- for the next pixel
#         2. kC_ij: Pixel's value used to impact chaotic map (the same type as kC_minus)
#         3. kC: kC
def MIE_FAST_Perm_and_Diff_pixels_DEC(kC, XY, XY_new, pseudoVal_string_C, pseudoVal_string_Cx, kC_minus, n):
    i = XY[0]
    j = XY[1]

    for k in range(global_params.K):
        # Permutation
        # First, decrypt for the lastest pixel
        temp = kC[XY_new[k][2]][XY_new[k][0]][XY_new[k][1]]
        temp_str = xor(np.binary_repr(temp, width = global_params.k2), pseudoVal_string_Cx[k]) # temp_value = I[i][j] XOR C[i-1][j]
        kC[XY_new[k][2]][XY_new[k][0]][XY_new[k][1]] = np.uint8(int(temp_str, 2))

        # Diffusion
        temp = kC[k][i][j] # Current pixel after Permutations
        temp_str = xor(np.binary_repr(temp, width = global_params.k2), np.binary_repr(kC_minus[k][0], width = global_params.k2)) # temp_value = I[i][j] XOR C[i-1][j]
        temp_str = xor(temp_str, pseudoVal_string_C[k]) # temp_value XOR pseudoVal_string_C (result of chaotic map)
        kC[k][i][j] = np.uint8(int(temp_str, 2))

        # The pixel permuted with current pixel is also diffused
        temp = kC[k][i][j]
        kC[k][i][j] = kC[XY_new[k][2]][XY_new[k][0]][XY_new[k][1]]
        kC[XY_new[k][2]][XY_new[k][0]][XY_new[k][1]] = temp

    # For the next pixel: Pass the diffused pixel's value to chaotic map
    kP_ij = np.zeros_like(global_params.kC0, dtype=np.uint8)
    for k in range(global_params.K):
        kP_ij[k][0] = kC[k][i][j]

    # Find kC_minus for the next pixel
    kC_m = np.zeros((global_params.K, 1), dtype=np.uint8)
    if (i > 0): # This means i - 1 is not out of range (i - 1 >= 0)
        if (j > 1): # This means k[i][j-2] is inside image because j - 2 is not out of range (j - 2 > 0)
            for k in range(global_params.K):
                kC_m[k][0] = np.uint8(kC[k][i][j-2])
        else: # This means j - 2 is out of range (j - 2 < 0)
            for k in range(global_params.K):
                kC_m[k][0] = np.uint8(kC[k][i-1][global_params.N-(2-j)]) # Go to the (global_params.N-(2-j))th pixel of the previous rows
    else: # This means i - 1 is out of range (i - 1 < 0)
        if (j > 1): # This means j - 2 is not out of range (j - 2 >= 0)
            for k in range(global_params.K):
                kC_m[k][0] = np.uint8(kC[k][i][j-2])
        elif(j == 1): # This mean j - 2 == -1
            if (n == 0):
                for k in range(global_params.K):
                    kC_m[k][0] = np.uint8(kC[k][global_params.M-1][global_params.N-1])
            else: # The last iteration
                for k in range(global_params.K):
                    kC_m[k][0] = np.uint8(global_params.kC0[k][0])

    return kC_m, kP_ij, kC
