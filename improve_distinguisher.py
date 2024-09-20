import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import statistics
import math
import os
from tqdm import trange
import matplotlib.pyplot as plt
import gift as gift
import ascon as ascon


if not os.path.exists("./graphs"):
    os.makedirs("./graphs/")
        
def prediction(n,input_diff,input_diff_ML,num_rounds,class_data,ml_rounds,acc,accuracy1,accuracy0,bit_size,loop):
    print("Input Parameters--> Model Trained on "+str(bit_size)+" output bits of "+str(CIPHER_NAME)+" | Input Difference for Data:  " + str("".join([hex(i)[2:].zfill(2) for i in input_diff])).upper()+ " | Input Difference for ML model:  " + str("".join([hex(i)[2:].zfill(2) for i in input_diff_ML])).upper()+ " | Prediction Data: 2^" + str(int(math.log(n,2)))+" | Classical Data: 2^" + str(int(math.log(class_data,2))) + " | Total Number of Rounds: " + str(num_rounds) +  " | ML Number of Rounds: " + str(ml_rounds) + " | Model Accuracy: " + str(acc) + " | TP_Real Accuracy: " + str(accuracy1) + " | TP_Random Accuracy: " + str(accuracy0) + " | Number of Experiments: " + str(2*loop)+"\n")
    if (CIPHER_NAME=="GIFT128"):
        cipher = gift
        acc_precision = 2; 
    elif (CIPHER_NAME=="ASCON320"):
        cipher = ascon
        acc_precision = 16;
    if (bit_size < BLOCK_SIZE):
        if (CIPHER_NAME=="GIFT128"):
            model_index = 0;
            bit_range = range(0,BLOCK_SIZE,8)
        elif (CIPHER_NAME=="ASCON320"):
            model_index = 7;
            bit_range = range(7*BLOCK_SIZE//8,BLOCK_SIZE)
    else:
        model_index = 8
            
    with tf.device('/CPU:0'):
        cipher.diff_arr = input_diff_ML
        model = load_model("./saved_models_paper/model_"+str(model_index)+"_"+str(CIPHER_NAME)+"_"+ str(bit_size)  +'_bits_'+str(ml_rounds)+'_'+ str("".join([hex(i)[2:].zfill(2) for i in input_diff_ML])).upper() + '_acc_'+str(acc).ljust(acc_precision+2,'0')+'.h5') 
        print(str("./saved_models_paper/model_"+str(model_index)+"_"+ str(bit_size)  +'_bits_'+str(ml_rounds)+'_'+ str("".join([hex(i)[2:].zfill(2) for i in input_diff_ML])).upper() + '_acc_'+str(acc).ljust(acc_precision+2,'0')+'.h5')+ " Model Loaded!\n")
        if (accuracy1==0 and accuracy0==0):
            print("Calculating Accuracies for Real and Random Case: ")
            X, Y = cipher.make_td_diff( n , 1 ,ml_rounds,data=2)
            if (bit_size<BLOCK_SIZE):
                loss2, accuracy2 = model.evaluate(X[:,[i for i in bit_range]], Y)
            elif (bit_size==BLOCK_SIZE):
                loss2, accuracy2 = model.evaluate(X,Y)
            print('loss=', loss2)
            print('acc=', accuracy2)
            
            print("All Real Case: ")
            TP_predictions_count = []
            for j in range(0,10):
                X, Y = cipher.make_td_diff( n , 1 , ml_rounds,data=1)
                if (bit_size<BLOCK_SIZE):
                    P = model.predict(X[:,[i for i in bit_range]])
                elif (bit_size==BLOCK_SIZE):
                    P = model.predict(X)
                TP_predictions_count.append(len(np.where(np.array(P) > 0.5)[0])/n)
            accuracy1 = min(TP_predictions_count)
            print("TP Accuracy: " + str(accuracy1))
            print("All Random Case: ")
            TP_predictions_count = []
            for j in range(0,10):
                X, Y = cipher.make_td_diff( n , 1 , ml_rounds,data=0)
                if (bit_size<BLOCK_SIZE):
                    P = model.predict(X[:,[i for i in bit_range]])
                elif (bit_size==BLOCK_SIZE):
                    P = model.predict(X)
                TP_predictions_count.append(len(np.where(np.array(P) > 0.5)[0])/n)
            accuracy0 = max(TP_predictions_count)
            print("TP Accuracy: " + str(accuracy0))
            return accuracy1,accuracy0
        
        cipher.diff_arr = input_diff
        TP_arr = []
        print("Making Prediction for All Real Data: ")
        for l in trange(0,loop):
            X, Y = cipher.make_td_diff( n , 1 , num_rounds,data=1)
            if (bit_size<BLOCK_SIZE):
                P = model.predict(X[:,[i for i in bit_range]],verbose=0)
            elif (bit_size==BLOCK_SIZE):
                P = model.predict(X,verbose=0)
            TP_predictions_count = len(np.where(np.array(P) > 0.5)[0])
            TP_arr.append(TP_predictions_count)
        print("Making Prediction for All Random Data: ")
        for l in trange(0,loop):
            X, Y = cipher.make_td_diff( n , 1 , num_rounds,data=0)
            if (bit_size<BLOCK_SIZE):
                P = model.predict(X[:,[i for i in bit_range]],verbose=0)
            elif (bit_size==BLOCK_SIZE):
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
        plt.savefig("graphs/"+str(CIPHER_NAME)+"_"+str(bit_size)+"_bits_" + str(num_rounds)+"_rounds_data_2_" + str(str(int(math.log(n,2)))) + ".png"  )
        plt.show()
        
        print("\nResults for Cutoff Using Graph (Algo 3):  " + str(cutoff_graph))
        cal_accuracy(TP_arr,cutoff_graph,loop)
        print("\nResults for Cutoff Calculated (Algo 4):  " + str(cutoff_cal))
        return cal_accuracy(TP_arr,cutoff_cal,loop)
     

def cal_accuracy(TP_arr,cutoff,loop):
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
    print("Input Parameters--> Model Trained on "+str(bit_size)+" output bits of "+str(CIPHER_NAME)+" | Input Difference for Data:  " + str("".join([hex(i)[2:].zfill(2) for i in input_diff])).upper()+ " | Input Difference for ML model:  " + str("".join([hex(i)[2:].zfill(2) for i in input_diff_ML])).upper()+ " | Data Required (beta): 2^" + str(int(math.log(n,2)))+" | Classical Data: 2^" + str(int(math.log(class_data,2))) + " | Total Number of Rounds: " + str(num_rounds) +  " | ML Number of Rounds: " + str(ml_rounds) + " | Model Accuracy: " + str(acc) + " | Cutoff (C_T): " + str(C_T)+ " | Number of Experiments: " + str(2*loop)+"\n")
    
    if (CIPHER_NAME=="GIFT128"):
        cipher = gift
        acc_precision = 2; 
    elif (CIPHER_NAME=="ASCON320"):
        cipher = ascon
        acc_precision = 16;
    if (bit_size < BLOCK_SIZE):
        if (CIPHER_NAME=="GIFT128"):
            model_index = 0;
            bit_range = range(0,BLOCK_SIZE,8)
        elif (CIPHER_NAME=="ASCON320"):
            model_index = 7;
            bit_range = range(7*BLOCK_SIZE//8,BLOCK_SIZE)
    else:
        model_index = 8
        
    with tf.device('/CPU:0'):
        cipher.diff_arr = input_diff_ML
        model = load_model("./saved_models_paper/model_"+str(model_index)+"_"+str(CIPHER_NAME)+"_"+ str(bit_size)  +'_bits_'+str(ml_rounds)+'_'+ str("".join([hex(i)[2:].zfill(2) for i in input_diff_ML])).upper() + '_acc_'+str(acc).ljust(acc_precision+2,'0')+'.h5') 
        print(str("./saved_model/model_"+str(model_index)+"_"+ str(bit_size)  +'_bits_'+str(ml_rounds)+'_'+ str("".join([hex(i)[2:].zfill(2) for i in input_diff_ML])).upper() + '_acc_'+str(acc).ljust(acc_precision+2,'0')+'.h5')+ " Model Loaded!\n")
        cipher.diff_arr = input_diff
        TP_arr = []
        print("Making Prediction for All Real Data: ")
        for l in trange(0,loop):
            X, Y = cipher.make_td_diff( n , 1 , num_rounds,data=1)
            if (bit_size<BLOCK_SIZE):
                P = model.predict(X[:,[i for i in bit_range]],verbose=0)
            elif (bit_size==BLOCK_SIZE):
                P = model.predict(X,verbose=0)
            TP_predictions_count = len(np.where(np.array(P) > 0.5)[0])
            TP_arr.append(TP_predictions_count)
        print("Making Prediction for All Random Data: ")
        for l in trange(0,loop):
            X, Y = cipher.make_td_diff( n , 1 , num_rounds,data=0)
            if (bit_size<BLOCK_SIZE):
                P = model.predict(X[:,[i for i in bit_range]],verbose=0)
            elif (bit_size==BLOCK_SIZE):
                P = model.predict(X,verbose=0)
            TP_predictions_count = len(np.where(np.array(P) > 0.5)[0])
            TP_arr.append(TP_predictions_count)
        print("\nResults for Manual Cutoff (For Validation):  " + str(C_T))
        return cal_accuracy(TP_arr,C_T,loop) 

def distinguisher(n_test,num_rounds,ml_round,class_data,acc,input_diff,input_diff_ML,bit_size,loop,validate,beta,C_T):
    
    if (validate == False):
        accuracy1,accuracy0 = prediction(n=n_test,input_diff=input_diff,input_diff_ML=input_diff_ML, class_data=class_data,num_rounds=num_rounds,ml_rounds=ml_rounds,acc=acc, accuracy1=0, accuracy0=0,bit_size=bit_size, loop=50) 
        
        for i in range(int(math.log(class_data,2)),25):
            if (i<2):
                continue
            Accuracy = prediction(n=2 ** i,input_diff=input_diff,input_diff_ML=input_diff_ML,class_data = class_data,num_rounds=num_rounds,ml_rounds=ml_rounds,acc=acc, accuracy1=accuracy1, accuracy0=accuracy0, bit_size=bit_size, loop=loop) 
            print("\n+++++++++++++++++++++++++++++++++++++\n")
            if (Accuracy>98):
                break
    else: 
        for j in range(0,10):  # no. of experiments 
            validation(n=beta,input_diff=input_diff,input_diff_ML=input_diff_ML,class_data = class_data,num_rounds=num_rounds,ml_rounds=ml_rounds,acc=acc,bit_size=bit_size, C_T = C_T, loop=loop) 
            print("\n+++++++++++++++++++++++++++++++++++++\n")

    
if __name__ == '__main__':
    CIPHER_NAME = "GIFT128" # GIFT128 or ASCON
    if (CIPHER_NAME=="GIFT128"):
        BLOCK_SIZE = 128
        num_rounds = 8
        ml_rounds = 6
        class_data = 2**10
        bit_size = 128 # 16 or 128
        if (bit_size==BLOCK_SIZE):
            # acc_arr = [0,0,0,0,0,.94,.73,.55] # 128 bits - 5/6/7 rounds - precision is upto 2 digit so no need to change as per model
            acc_arr = [0,0,0,0,0,.83,.58] # for differential-ML (trained on difference [0x00,0x02,0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00])
        elif(bit_size==16):
            acc_arr = [0,0,0,0,0,.72,.56,.51] # 16 bits - 5/6/7 rounds
        acc = acc_arr[ml_rounds]
        # input_diff = [0x00,0x02,0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
        # input_diff = [0x00,0x00,0x00,0x00,0x00,0x00,0x0c,0x00,0x00,0x00,0x06,0x00,0x00,0x00,0x00,0x00]
        # input_diff = [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x11,0x00,0x00,0x00,0x00]
        # input_diff = [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xa0,0x00,0xa0,0x00]
        # input_diff = [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01]
        # input_diff_ML = [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01]
        input_diff = [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x11,0x00,0x00,0x00,0x00] #[0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xa0,0x00,0xa0,0x00]
        input_diff_ML = [0x00,0x02,0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
        loop = 50
        n_test = 2 ** 16
        validate = True # False for construction and True for validation
        beta = 2 ** 18 # 2^i data complexity(beta) calculated using prediction function; anything if validate is False
        C_T = 112189 # C_T calculated using prediction function; anything if validate is False
        
    elif (CIPHER_NAME=="ASCON320"):
        BLOCK_SIZE = 320
        num_rounds = 4
        ml_rounds = 4
        class_data = 2**0
        bit_size = 40 # 40 or 320
        if (bit_size==BLOCK_SIZE):
            acc_arr = [0,0,0,0,0.5027949810028076] # 320 bits - 4rounds - the accuracy must be according to the model used as precision is too high; as per paper - 0.5027949810028076
        elif(bit_size==40):
            acc_arr = [0,0,0,0,0.5022529959678650] # 40 bits - 4 rounds - the accuracy must be according to the model used as precision is too high; as per paper - 0.5022529959678650
        acc = acc_arr[ml_rounds]
        input_diff = [0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000001]
        input_diff_ML = [0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000001]
        loop = 50
        n_test = 2 ** 19
        validate = True # False for construction and True for validation
        beta = 2 ** 18 # 2^i data complexity(beta) calculated using prediction function; anything if validate is False
        C_T = 125161 # C_T calculated using prediction function; anything if validate is False
    distinguisher(n_test,num_rounds,ml_rounds,class_data,acc,input_diff,input_diff_ML,bit_size,loop,validate,beta,C_T)


