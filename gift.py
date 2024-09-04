import numpy as np
from os import urandom

BLOCK_SIZE = 128
diff_arr = [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x0A]
GIFT_RC = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F,
           0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B,
           0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E,
           0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A]

def SWAPMOVE_1(X,M,n):
    return( (X ^ ((X ^ (X >> n)) & M)) ^ (((X ^ (X >> n)) & M)<< n));

def SWAPMOVE_2(A,B,M,n):
    return(A ^ (((B ^ (A >> n)) & M)<< n), B ^ ((B ^ (A >> n)) & M));


def rowperm(S, B0_pos, B1_pos, B2_pos, B3_pos):
    T = np.zeros_like(S)
    for b in range(BLOCK_SIZE//16):
        T |= ((S >> (4 * b + 0)) & 0x1) << (b + BLOCK_SIZE//16 * B0_pos)
        T |= ((S >> (4 * b + 1)) & 0x1) << (b + BLOCK_SIZE//16 * B1_pos)
        T |= ((S >> (4 * b + 2)) & 0x1) << (b + BLOCK_SIZE//16 * B2_pos)
        T |= ((S >> (4 * b + 3)) & 0x1) << (b + BLOCK_SIZE//16 * B3_pos)
    return T



def giftb128(P, K, C, nr):
    S = np.zeros((P.shape[0], 4), dtype=np.uint32)
    W = np.zeros((P.shape[0], 8), dtype=np.uint16)
   
    S[:, 0] = (np.uint32(P[:, 6]) << 24) | (np.uint32(P[:, 7]) << 16) | (np.uint32(P[:, 14]) << 8) | np.uint32(P[:, 15])
    S[:, 1] = (np.uint32(P[:, 4]) << 24) | (np.uint32(P[:, 5]) << 16) | (np.uint32(P[:, 12]) << 8) | np.uint32(P[:, 13])
    S[:, 2] = (np.uint32(P[:, 2]) << 24) | (np.uint32(P[:, 3]) << 16) | (np.uint32(P[:, 10]) << 8) | np.uint32(P[:, 11])
    S[:, 3] = (np.uint32(P[:, 0]) << 24) | (np.uint32(P[:, 1]) << 16) | (np.uint32(P[:, 8]) << 8) | np.uint32(P[:, 9])

    for i in range(0,4): S[:, i] = SWAPMOVE_1(S[:, i], 0x0a0a0a0a, 3);
    for i in range(0,4): S[:, i] = SWAPMOVE_1(S[:, i], 0x00cc00cc, 6);
    for i in range(1,4): S[:, 0], S[:, i] = SWAPMOVE_2(S[:, 0], S[:, i], 0x00f000f, 4*i);
    for i in range(2,4): S[:, 1], S[:, i] = SWAPMOVE_2(S[:, 1], S[:, i], 0x00f000f0, 4*(i-1));
    for i in range(3,4): S[:, 2], S[:, i] = SWAPMOVE_2(S[:, 2], S[:, i], 0x0f000f00, 4*(i-2));
    
    
    W[:, 0] = (np.uint16(K[:, 0]) << 8) | np.uint16(K[:, 1])
    W[:, 1] = (np.uint16(K[:, 2]) << 8) | np.uint16(K[:, 3])
    W[:, 2] = (np.uint16(K[:, 4]) << 8) | np.uint16(K[:, 5])
    W[:, 3] = (np.uint16(K[:, 6]) << 8) | np.uint16(K[:, 7])
    W[:, 4] = (np.uint16(K[:, 8]) << 8) | np.uint16(K[:, 9])
    W[:, 5] = (np.uint16(K[:, 10]) << 8) | np.uint16(K[:, 11])
    W[:, 6] = (np.uint16(K[:, 12]) << 8) | np.uint16(K[:, 13])
    W[:, 7] = (np.uint16(K[:, 14]) << 8) | np.uint16(K[:, 15])

    for round in range(nr):
        S[:, 1] ^= S[:, 0] & S[:, 2]
        S[:, 0] ^= S[:, 1] & S[:, 3]
        S[:, 2] ^= S[:, 0] | S[:, 1]
        S[:, 3] ^= S[:, 2]
        S[:, 1] ^= S[:, 3]
        S[:, 3] ^= 0xFFFFFFFF
        S[:, 2] ^= S[:, 0] & S[:, 1]

        T = S[:, 0].copy()
        S[:, 0] = S[:, 3]
        S[:, 3] = T

        S[:, 0] = rowperm(S[:, 0], 0, 3, 2, 1)
        S[:, 1] = rowperm(S[:, 1], 1, 0, 3, 2)
        S[:, 2] = rowperm(S[:, 2], 2, 1, 0, 3)
        S[:, 3] = rowperm(S[:, 3], 3, 2, 1, 0)
        

        S[:, 2] ^= (np.uint32(W[:, 2]) << 16) | np.uint32(W[:, 3])
        S[:, 1] ^= (np.uint32(W[:, 6]) << 16) | np.uint32(W[:, 7])

        S[:, 3] ^= (0x80000000 ^ GIFT_RC[round])

        T6 = ((W[:, 6] >> 2) & 0xFFFF) | ((W[:, 6] << 14) & 0xFFFF)
        T7 = ((W[:, 7] >> 12) & 0xFFFF) | ((W[:, 7] << 4) & 0xFFFF)
        W[:, 7] = W[:, 5]
        W[:, 6] = W[:, 4]
        W[:, 5] = W[:, 3]
        W[:, 4] = W[:, 2]
        W[:, 3] = W[:, 1]
        W[:, 2] = W[:, 0]
        W[:, 1] = T7
        W[:, 0] = T6
     
    for i in range(3,4): S[:, 2], S[:, i] = SWAPMOVE_2(S[:, 2], S[:, i], 0x0f000f00, 4*(i-2));
    for i in range(2,4): S[:, 1], S[:, i] = SWAPMOVE_2(S[:, 1], S[:, i], 0x00f000f0, 4*(i-1));
    for i in range(1,4): S[:, 0], S[:, i] = SWAPMOVE_2(S[:, 0], S[:, i], 0x00f000f, 4*i);
    for i in range(0,4): S[:, i] = SWAPMOVE_1(S[:, i], 0x00cc00cc, 6);
    for i in range(0,4): S[:, i] = SWAPMOVE_1(S[:, i], 0x0a0a0a0a, 3);
    
    C[:, 0] = (S[:, 3] >> 24) & 0xFF
    C[:, 1] = (S[:, 3] >> 16) & 0xFF
    C[:, 2] = (S[:, 2] >> 24) & 0xFF
    C[:, 3] = (S[:, 2] >> 16) & 0xFF
    C[:, 4] = (S[:, 1] >> 24) & 0xFF
    C[:, 5] = (S[:, 1] >> 16) & 0xFF
    C[:, 6] = (S[:, 0] >> 24) & 0xFF
    C[:, 7] = (S[:, 0] >> 16) & 0xFF
    C[:, 8] = (S[:, 3] >> 8) & 0xFF
    C[:, 9] = S[:, 3] & 0xFF
    C[:, 10] = (S[:, 2] >> 8) & 0xFF
    C[:, 11] = S[:, 2] & 0xFF
    C[:, 12] = (S[:, 1] >> 8) & 0xFF
    C[:, 13] = S[:, 1] & 0xFF
    C[:, 14] = (S[:, 0] >> 8) & 0xFF
    C[:, 15] = S[:, 0] & 0xFF
    
    
def to_binary(arr):
    binary_array = np.unpackbits(arr, axis=1)
    return binary_array


def make_td(n, s, nr):
    WORDS = BLOCK_SIZE // 8
    
    Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1

    plain0 = np.random.randint(0, 2 ** 8, size=(n, WORDS * s), dtype=np.uint8)
    diff = np.zeros(WORDS * s, dtype=np.uint8)
    diff[np.arange(4, WORDS * s, WORDS)] = 8
    plain1 = plain0 ^ diff
    num_rand_samples = np.sum(Y == 0)
    plain1[Y == 0, :] = np.random.randint(0, 2**8, size=(num_rand_samples, WORDS*s), dtype=np.uint8)
    key = np.random.randint(0, 2 ** 8, size=(n*s, 16), dtype=np.uint8)
    c0 = np.zeros((n*s, WORDS), dtype=np.uint8)
    c1 = np.zeros((n*s, WORDS), dtype=np.uint8)
    plain0 = plain0.reshape(n*s, WORDS)
    plain1 = plain1.reshape(n*s, WORDS)
    giftb128(plain0, key, c0, nr)
    giftb128(plain1, key, c1, nr)
    X = np.concatenate((to_binary(c0), to_binary(c1)), axis=1)
    return X.reshape(n, 2*s*BLOCK_SIZE), Y


def make_td_diff(n, s, nr,data=2):
    WORDS = BLOCK_SIZE // 8
    if (data==0):
        Y = np.frombuffer(urandom(n), dtype=np.uint8) & 0
    elif(data==1):
        Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1 | 1
    elif(data==2):
        Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
    
    plain0 = np.random.randint(0, 2 ** 8, size=(n, WORDS * s), dtype=np.uint8)
    diff = np.zeros(WORDS * s, dtype=np.uint8)
    diff[np.arange(0, WORDS * s, WORDS)] = diff_arr[0]
    diff[np.arange(1, WORDS * s, WORDS)] = diff_arr[1]
    diff[np.arange(2, WORDS * s, WORDS)] = diff_arr[2]
    diff[np.arange(3, WORDS * s, WORDS)] = diff_arr[3]
    diff[np.arange(4, WORDS * s, WORDS)] = diff_arr[4]
    diff[np.arange(5, WORDS * s, WORDS)] = diff_arr[5]
    diff[np.arange(6, WORDS * s, WORDS)] = diff_arr[6]
    diff[np.arange(7, WORDS * s, WORDS)] = diff_arr[7]
    if(BLOCK_SIZE==128):
        diff[np.arange(8, WORDS * s, WORDS)] = diff_arr[8]
        diff[np.arange(9, WORDS * s, WORDS)] = diff_arr[9]
        diff[np.arange(10, WORDS * s, WORDS)] = diff_arr[10]
        diff[np.arange(11, WORDS * s, WORDS)] = diff_arr[11]
        diff[np.arange(12, WORDS * s, WORDS)] = diff_arr[12]
        diff[np.arange(13, WORDS * s, WORDS)] = diff_arr[13]
        diff[np.arange(14, WORDS * s, WORDS)] = diff_arr[14]
        diff[np.arange(15, WORDS * s, WORDS)] = diff_arr[15]
    plain1 = plain0 ^ diff
    num_rand_samples = np.sum(Y == 0)
    plain1[Y == 0, :] = np.random.randint(0, 2 ** 8, size=(num_rand_samples, WORDS * s), dtype=np.uint8)
    key = np.random.randint(0, 2 ** 8, size=(n * s, 16), dtype=np.uint8)
    c0 = np.zeros((n * s, WORDS), dtype=np.uint8)
    c1 = np.zeros((n * s, WORDS), dtype=np.uint8)
    plain0 = plain0.reshape(n * s, WORDS)
    plain1 = plain1.reshape(n * s, WORDS)
    giftb128(plain0, key, c0, nr)
    giftb128(plain1, key, c1, nr)
    X = to_binary(c0 ^ c1)
    return X, Y

def make_with_diff(BLOCK_SIZE, n, nr, diff):
    WORDS = BLOCK_SIZE // 8
    Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
    plain0 = np.random.randint(0, 2 ** 8, size=(n, WORDS), dtype=np.uint8)
    plain1 = plain0 ^ diff
    num_rand_samples = np.sum(Y == 0)
    plain1[Y == 0, :] = np.random.randint(0, 2 ** 8, size=(num_rand_samples, WORDS), dtype=np.uint8)
    key = np.random.randint(0, 2 ** 8, size=(n, 16), dtype=np.uint8)
    c0 = np.zeros((n, WORDS), dtype=np.uint8)
    c1 = np.zeros((n, WORDS), dtype=np.uint8)
    giftb128(plain0, key, c0, nr)
    giftb128(plain1, key, c1, nr)
    X = to_binary(c0 ^ c1)
    return X, Y


def make_real_data(BLOCK_SIZE, s, nr):
    WORDS = BLOCK_SIZE // 8
    plain0 = np.random.randint(0, 2 ** 8, size=(s, WORDS), dtype=np.uint8)
    diff = np.zeros((s, WORDS), dtype=np.uint8)
    diff[np.arange(0, WORDS * s, WORDS)] = 0x00
    diff[np.arange(1, WORDS * s, WORDS)] = 0x44
    diff[np.arange(2, WORDS * s, WORDS)] = 0x00
    diff[np.arange(3, WORDS * s, WORDS)] = 0x00
    diff[np.arange(4, WORDS * s, WORDS)] = 0x00
    diff[np.arange(5, WORDS * s, WORDS)] = 0x11
    diff[np.arange(6, WORDS * s, WORDS)] = 0x00
    diff[np.arange(7, WORDS * s, WORDS)] = 0x00
    plain1 = plain0 ^ diff
    key = np.random.randint(0, 2 ** 8, size=(s, 16), dtype=np.uint8)
    c0 = np.zeros((s, WORDS), dtype=np.uint8)
    c1 = np.zeros((s, WORDS), dtype=np.uint8)
    giftb128(plain0, key, c0, nr)
    giftb128(plain1, key, c1, nr)
    X = to_binary(c0 ^ c1)
    return X





