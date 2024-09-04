import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import gift as gift
import statistics
import math
import os
from tqdm import trange
import matplotlib.pyplot as plt

BLOCK_SIZE = 128
if not os.path.exists("./images"):
    os.makedirs("./images/")
        
def prediction(n,input_diff,input_diff_ML,num_rounds,class_data,ml_rounds,acc,accuracy1,accuracy0,bit_size,loop):
    print("Input Parameters--> Model Trained on "+str(bit_size)+" output bits of GIFT-128"+" | Input Difference for Data:  " + str("".join([hex(i)[2:].zfill(2) for i in input_diff])).upper()+ " | Input Difference for ML model:  " + str("".join([hex(i)[2:].zfill(2) for i in input_diff_ML])).upper()+ " | Prediction Data: 2^" + str(int(math.log(n,2)))+" | Classical Data: 2^" + str(int(math.log(class_data,2))) + " | Total Number of Rounds: " + str(num_rounds) +  " | ML Number of Rounds: " + str(ml_rounds) + " | Model Accuracy: " + str(acc) + " | TP_Real Accuracy: " + str(accuracy1) + " | TP_Random Accuracy: " + str(accuracy0) + " | Number of Experiments: " + str(2*loop)+"\n")
    with tf.device('/CPU:0'):
        gift.diff_arr = input_diff_ML
        if (bit_size==16):
            model = load_model('./saved_model/model_0_GIFT'+str(BLOCK_SIZE)+"_"+ str(bit_size)  +'_bits_'+str(ml_rounds)+'_'+ str("".join([hex(i)[2:].zfill(2) for i in input_diff_ML])).upper() + '_acc_'+str(acc).ljust(4,'0')+'.h5') 
        elif (bit_size==128):
            model = load_model('./saved_model/model_8_GIFT'+str(BLOCK_SIZE)+"_"+ str(bit_size)  +'_bits_'+str(ml_rounds)+'_'+ str("".join([hex(i)[2:].zfill(2) for i in input_diff_ML])).upper() + '_acc_'+str(acc).ljust(4,'0')+'.h5') 
        print(str('./saved_model/model_0_GIFT'+str(BLOCK_SIZE)+"_"+ str(bit_size)  +'_bits_'+str(ml_rounds)+'_'+ str("".join([hex(i)[2:].zfill(2) for i in input_diff_ML])).upper() + '_acc_'+str(acc).ljust(4,'0')+'.h5')+ " Model Loaded!\n")
        if (accuracy1==0 and accuracy0==0):
            print("Calculating Accuracies for Real and Random Case: ")
            X, Y = gift.make_td_diff( n , 1 ,ml_rounds,data=2)
            if (bit_size==16):
                loss2, accuracy2 = model.evaluate(X[:,[i for i in range(0,BLOCK_SIZE,8)]], Y)
            elif (bit_size==128):
                loss2, accuracy2 = model.evaluate(X,Y)
            print('loss=', loss2)
            print('acc=', accuracy2)
            
            print("All Real Case: ")
            TP_predictions_count = []
            for j in range(0,10):
                X, Y = gift.make_td_diff( n , 1 , ml_rounds,data=1)
                if (bit_size==16):
                    P = model.predict(X[:,[i for i in range(0,BLOCK_SIZE,8)]])
                elif (bit_size==128):
                    P = model.predict(X)
                TP_predictions_count.append(len(np.where(np.array(P) > 0.5)[0])/n)
            accuracy1 = min(TP_predictions_count)
            print("TP Accuracy: " + str(accuracy1))
            print("All Random Case: ")
            TP_predictions_count = []
            for j in range(0,10):
                X, Y = gift.make_td_diff( n , 1 , ml_rounds,data=0)
                if (bit_size==16):
                    P = model.predict(X[:,[i for i in range(0,BLOCK_SIZE,8)]])
                elif (bit_size==128):
                    P = model.predict(X)
                TP_predictions_count.append(len(np.where(np.array(P) > 0.5)[0])/n)
            accuracy0 = max(TP_predictions_count)
            print("TP Accuracy: " + str(accuracy0))
            return accuracy1,accuracy0
        
        gift.diff_arr = input_diff
        TP_arr = []
        print("Making Prediction for All Real Data: ")
        for l in trange(0,loop):
            X, Y = gift.make_td_diff( n , 1 , num_rounds,data=1)
            if (bit_size==16):
                P = model.predict(X[:,[i for i in range(0,BLOCK_SIZE,8)]],verbose=0)
            elif (bit_size==128):
                P = model.predict(X,verbose=0)
            TP_predictions_count = len(np.where(np.array(P) > 0.5)[0])
            TP_arr.append(TP_predictions_count)
        print("Making Prediction for All Random Data: ")
        for l in trange(0,loop):
            X, Y = gift.make_td_diff( n , 1 , num_rounds,data=0)
            if (bit_size==16):
                P = model.predict(X[:,[i for i in range(0,BLOCK_SIZE,8)]],verbose=0)
            elif (bit_size==128):
                P = model.predict(X,verbose=0)
            TP_predictions_count = len(np.where(np.array(P) > 0.5)[0])
            TP_arr.append(TP_predictions_count)
        print("Prediction Results (All Real Data): ")
        print(TP_arr[0:loop])
        print("Prediction Results (All Random Data): ")
        print(TP_arr[loop:])
        print("Average TP in All Real Data: " + str(statistics.mean(TP_arr[0:loop])))
        print("Average TP in All Random Data: " + str(statistics.mean(TP_arr[loop:])))
        print("Difference in Average TP Data (Real-Random): " + str(statistics.mean(TP_arr[0:loop])-statistics.mean(TP_arr[loop:])))
        print("Min TP in All Real Data: " + str(min(TP_arr[0:loop])))
        print("Max TP in All Random Data: " + str(max(TP_arr[loop:])))
        print("Difference in Min_Real-Max_Random: " + str(min(TP_arr[0:loop])-max(TP_arr[loop:])))
        cutoff_graph = (min(TP_arr[0:loop]) + max(TP_arr[loop:]))//2
        cutoff_cal = math.ceil(max(2,n*accuracy0 + ( n//class_data*(accuracy1-accuracy0)*(0.5))))
        print("Difference in Accuracy: " + str(accuracy1-accuracy0) + " | Data Difference Expected (based on accuracy): " + str((n//class_data)*(accuracy1-accuracy0)) + " | Cutoff Using Graph (Algo 3): " + str(cutoff_graph)+ " | Cutoff Calculated (Algo 4): " + str(cutoff_cal)+" | Data Used: 2^" +str(int(math.log(n,2))))
        
        xdata = [i+1 for i in range(0,loop)]
        ydata_1 = TP_arr[0:loop]
        ydata_0 = TP_arr[loop:2*loop]
        plt.figure()
        plt.title("Rounds: "+ str(num_rounds) +" | Data Required: 2^" + str(int(math.log(n,2))) )
        plt.xlabel("Experiment No.")
        plt.ylabel("No. of Prediction > 0.5")
        plt.plot(xdata,ydata_1,'o',label = "TP",linestyle=":",color="green")
        plt.plot(xdata,ydata_0,'d', label = "TN",linestyle=":",color="red")
        plt.savefig("images/GIFT128_16_bits_" + str(num_rounds)+"_rounds_data_2_" + str(str(int(math.log(n,2)))) + ".png"  )
        plt.show()
        
        print("\nResults for Cutoff Using Graph (Algo 3):  " + str(cutoff_graph))
        cal_accuracy(TP_arr,cutoff_graph)
        print("\nResults for Cutoff Calculated (Algo 4):  " + str(cutoff_cal))
        return cal_accuracy(TP_arr,cutoff_cal)
     

def cal_accuracy(TP_arr,cutoff):
    TP_Real_count = 0
    TP_Random_count = 0
    for i in range(0,len(TP_arr)):
        if (TP_arr[i] > cutoff):
            if (i< loop):
                TP_Real_count += 1
        else:
            if (i >= loop):
                TP_Random_count += 1
    print("TP_Real Count: " + str(TP_Real_count) + " | TP_Random Count: " + str(TP_Random_count) + " | Accuracy: " + str(TP_Real_count+TP_Random_count)+str("%"))
    return (TP_Real_count+TP_Random_count)*100/(2*loop)

def validation(n,input_diff,input_diff_ML,num_rounds,class_data,ml_rounds,acc,bit_size,C_T,loop):
    print("Input Parameters--> Model Trained on "+str(bit_size)+" output bits of GIFT-128"+" | Input Difference for Data:  " + str("".join([hex(i)[2:].zfill(2) for i in input_diff])).upper()+ " | Input Difference for ML model:  " + str("".join([hex(i)[2:].zfill(2) for i in input_diff_ML])).upper()+ " | Prediction Data: 2^" + str(int(math.log(n,2)))+" | Classical Data: 2^" + str(int(math.log(class_data,2))) + " | Total Number of Rounds: " + str(num_rounds) +  " | ML Number of Rounds: " + str(ml_rounds) + " | Model Accuracy: " + str(acc) + " | Number of Experiments: " + str(2*loop)+"\n")
    
    with tf.device('/CPU:0'):
        gift.diff_arr = input_diff_ML
        if (bit_size==16):
            model = load_model('./saved_model/model_0_GIFT'+str(BLOCK_SIZE)+"_"+ str(bit_size)  +'_bits_'+str(ml_rounds)+'_'+ str("".join([hex(i)[2:].zfill(2) for i in input_diff_ML])).upper() + '_acc_'+str(acc).ljust(4,'0')+'.h5') 
        elif (bit_size==128):
            model = load_model('./saved_model/model_8_GIFT'+str(BLOCK_SIZE)+"_"+ str(bit_size)  +'_bits_'+str(ml_rounds)+'_'+ str("".join([hex(i)[2:].zfill(2) for i in input_diff_ML])).upper() + '_acc_'+str(acc).ljust(4,'0')+'.h5') 
        print(str('./saved_model/model_0_GIFT'+str(BLOCK_SIZE)+"_"+ str(bit_size)  +'_bits_'+str(ml_rounds)+'_'+ str("".join([hex(i)[2:].zfill(2) for i in input_diff_ML])).upper() + '_acc_'+str(acc).ljust(4,'0')+'.h5')+ " Model Loaded!\n")
        gift.diff_arr = input_diff
        TP_arr = []
        print("Making Prediction for All Real Data: ")
        for l in trange(0,loop):
            X, Y = gift.make_td_diff( n , 1 , num_rounds,data=1)
            if (bit_size==16):
                P = model.predict(X[:,[i for i in range(0,BLOCK_SIZE,8)]],verbose=0)
            elif (bit_size==128):
                P = model.predict(X,verbose=0)
            TP_predictions_count = len(np.where(np.array(P) > 0.5)[0])
            TP_arr.append(TP_predictions_count)
        print("Making Prediction for All Random Data: ")
        for l in trange(0,loop):
            X, Y = gift.make_td_diff( n , 1 , num_rounds,data=0)
            if (bit_size==16):
                P = model.predict(X[:,[i for i in range(0,BLOCK_SIZE,8)]],verbose=0)
            elif (bit_size==128):
                P = model.predict(X,verbose=0)
            TP_predictions_count = len(np.where(np.array(P) > 0.5)[0])
            TP_arr.append(TP_predictions_count)
        print("Results for Manual Cutoff (For Validation):  " + str(C_T))
        return cal_accuracy(TP_arr,C_T) 

if __name__ == '__main__':
    num_rounds = 7
    ml_rounds = 7
    class_data = 2**0
    acc_arr = [0,0,0,0,0,.94,.73,.55] # 128 bits - 5/6/7 rounds
    # acc_arr = [0,0,0,0,0,.72,.56,.51] # 16 bits - 5/6/7 rounds
    acc = acc_arr[ml_rounds]
    # input_diff = [0x00,0x02,0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
    # input_diff = [0x00,0x00,0x00,0x00,0x00,0x00,0x0c,0x00,0x00,0x00,0x06,0x00,0x00,0x00,0x00,0x00]
    # input_diff = [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x11,0x00,0x00,0x00,0x00]
    # input_diff = [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xa0,0x00,0xa0,0x00]
    input_diff = [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01]
    input_diff_ML = [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01]
    bit_size = 128
    loop = 50
    validate = True 
    if (validate == False):
        accuracy1,accuracy0 = prediction(n=2 ** 16,input_diff=input_diff,input_diff_ML=input_diff_ML, class_data=class_data,num_rounds=num_rounds,ml_rounds=ml_rounds,acc=acc, accuracy1=0, accuracy0=0,bit_size=bit_size, loop=50) 
        
        for i in range(int(math.log(class_data,2)),25):
            if (i<2):
                continue
            Accuracy = prediction(n=2 ** i,input_diff=input_diff,input_diff_ML=input_diff_ML,class_data = class_data,num_rounds=num_rounds,ml_rounds=ml_rounds,acc=acc, accuracy1=accuracy1, accuracy0=accuracy0, bit_size=bit_size, loop=loop) 
            print("\n+++++++++++++++++++++++++++++++++++++\n")
            if (Accuracy>98):
                break
    else: 
        i = 9 # 2^i data complexity(beta) calculated using prediction function
        C_T = 175 # C_T calculated using prediction function
        for j in range(0,10):  # no. of experiments 
            validation(n=2 ** i,input_diff=input_diff,input_diff_ML=input_diff_ML,class_data = class_data,num_rounds=num_rounds,ml_rounds=ml_rounds,acc=acc,bit_size=bit_size, C_T = C_T, loop=loop) 
            print("\n+++++++++++++++++++++++++++++++++++++\n")

    
    

    


