import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import math
import gift as gift


wdir = './saved_model/'
BLOCK_SIZE = 128

def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_accuracy', save_best_only=True)
    return res

def train(n, m, num_epochs, num_rounds, batch_size,bit_size,input_diff):
    gift.diff_arr = input_diff
    print("Input Parameters--> GIFT128 | Training Bits: "+ str(bit_size)+ " | Difference:  " + str("".join([hex(i)[2:].zfill(2) for i in input_diff])).upper()+ " | Training Data: 2^" + str(int(math.log(n,2)))+" | Testing Data: 2^" + str(int(math.log(m,2))) + " | Number of Rounds: " + str(num_rounds) )
    X, Y = gift.make_td_diff( n, 1, num_rounds) 
    X_eval, Y_eval = gift.make_td_diff(m, 1, num_rounds)
    with tf.device('/CPU:0'):
        
        if (bit_size==16):
            blocks = 8; iter_loop = 1
            for k in range(0,iter_loop):
                vars()['model_{0}'.format(k)] = Sequential()
                vars()['model_{0}'.format(k)].add(Dense(BLOCK_SIZE//2, input_dim=BLOCK_SIZE//blocks, activation='relu', kernel_regularizer=l2(10**-5)))
                vars()['model_{0}'.format(k)].add(Dense(BLOCK_SIZE//2, activation='relu', kernel_regularizer=l2(10**-5)))
                vars()['model_{0}'.format(k)].add(Dense(BLOCK_SIZE//2, activation='relu', kernel_regularizer=l2(10**-5)))
                vars()['model_{0}'.format(k)].add(Dense(1, activation='sigmoid', kernel_regularizer=l2(10**-5)))
            
                vars()['model_{0}'.format(k)].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                check = make_checkpoint(wdir + 'model_'+str(k)+'_GIFT' + str(BLOCK_SIZE) + "_"+ str(bit_size)  +'_bits_' +str(num_rounds) +"_" + str("".join([hex(i)[2:].zfill(2) for i in input_diff])).upper() +'_acc_{val_accuracy:.2f}.h5')
                
                h = vars()['model_{0}'.format(k)].fit(X[:,[i for i in range(k,BLOCK_SIZE,blocks)]], Y, epochs=num_epochs, batch_size=batch_size, shuffle=True, validation_data=(X_eval[:,[i for i in range(k,BLOCK_SIZE,blocks)]], Y_eval), callbacks=[check])
                print( 'model_'+str(k) + " Best validation accuracy: ", np.max(h.history['val_accuracy']))
                
        elif (bit_size==128):
            model = Sequential()
            model.add(Dense(BLOCK_SIZE, input_dim=BLOCK_SIZE, activation='relu', kernel_regularizer=l2(10**-5)))
            model.add(Dense(BLOCK_SIZE, activation='relu', kernel_regularizer=l2(10**-5)))
            model.add(Dense(BLOCK_SIZE, activation='relu', kernel_regularizer=l2(10**-5)))
            model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(10**-5)))
        
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            check = make_checkpoint(wdir + 'model_'+str(8)+'_GIFT' + str(BLOCK_SIZE) + "_"+ str(bit_size)  +'_bits_' +str(num_rounds) +"_" + str("".join([hex(i)[2:].zfill(2) for i in input_diff])).upper() +'_acc_{val_accuracy:.2f}.h5')
            
            h = model.fit(X, Y, epochs=num_epochs, batch_size=batch_size, shuffle=True,
                          validation_data=(X_eval, Y_eval), callbacks=[check])
            print('model_'+ str(8) + "Best validation accuracy: ", np.max(h.history['val_accuracy']))

if __name__ == '__main__':
    #bit_size = 128 or 16
    train(n=2**23, m= 2**18, num_epochs=10, num_rounds=6, batch_size=2**13, bit_size=128,input_diff=[0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01])
#input_diff = [0x00,0x02,0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
    


