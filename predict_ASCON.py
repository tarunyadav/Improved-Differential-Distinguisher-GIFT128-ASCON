import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import ascon as ascon
import statistics
import math
from tqdm import trange
import matplotlib.pyplot as plt

BLOCK_SIZE = 320
ascon.diff_arr = [0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000001]
def test_prediction(n,diff_arr,num_rounds,class_data,ml_rounds,acc,accuracy1,accuracy0,loop):
    print("Input Parameters--> Training on 128-bits of ASCON320 | BLOCK SIZE: "+ str(BLOCK_SIZE)+ " | Difference:  " + str("".join([hex(i)[2:].zfill(2) for i in diff_arr])).upper()+ " | Prediction Data: 2^" + str(int(math.log(n,2)))+" | Classical Data: 2^" + str(int(math.log(class_data,2))) + " | Total Number of Rounds: " + str(num_rounds) +  " | ML Number of Rounds: " + str(ml_rounds) + " | Model Accuracy: " + str(acc) + " | TP_Real Accuracy: " + str(accuracy1) + " | TP_Random Accuracy: " + str(accuracy0) + " | Number of Experiments: " + str(2*loop)+"\n")
    with tf.device('/CPU:0'):
        
        ascon.diff_arr = [0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000001]
        model = load_model('./saved_model/model_7_'+str(BLOCK_SIZE)+"_"+ str(int(BLOCK_SIZE//8))  +'_bits_'+str(ml_rounds)+'_'+ str("".join([hex(i)[2:].zfill(2) for i in ascon.diff_arr])).upper() + '_acc_'+str(acc).ljust(18,'0')+'.h5') 
        if (acc==1.00):
            acc = 0.99
        # acc = .5024660229682922 # old
        # acc = .5021539926528931  # full 
        if (accuracy1==0 and accuracy0==0):

            ascon.diff_arr = [0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000001]
            n_test = n
            X, Y = ascon.make_td_diff( n_test , 1 , ml_rounds,data=2)
            loss2, accuracy2 = model.evaluate(X[:,[i for i in range(7*BLOCK_SIZE//8,BLOCK_SIZE)]],Y)
            # loss2, accuracy2 = model.evaluate(X,Y)
            print('loss=', loss2)
            print('acc=', accuracy2)
            

            print("All Real Case: ")
            TP_predictions_count = []
            for j in range(0,10):
                X, Y = ascon.make_td_diff( n_test , 1 , ml_rounds,data=1)
                P = model.predict(X[:,[i for i in range(7*BLOCK_SIZE//8,BLOCK_SIZE)]])
                # P = model.predict(X)
                TP_predictions_count.append(len(np.where(np.array(P) > 0.5)[0])/n_test)
            # accuracy1 = (TP_predictions_count/10)/n_test
            accuracy1 = min(TP_predictions_count)
            print("TP Accuracy: " + str(accuracy1))
            # X, Y = ascon.make_td_diff( n , 1 , num_rounds-2,data=1)
            # loss1, accuracy1 = model.evaluate(X[:,[i for i in range(7,BLOCK_SIZE,8)]], Y)
            # loss1, accuracy1 = model.evaluate(X,Y)
            # print('loss=', loss1)
            # print('acc=', accuracy1)
            # print("All Random Case: ")
            TP_predictions_count = []
            for j in range(0,10):
                X, Y = ascon.make_td_diff( n_test , 1 , ml_rounds,data=0)
                P = model.predict(X[:,[i for i in range(7*BLOCK_SIZE//8,BLOCK_SIZE)]])
                # P = model.predict(X)
                TP_predictions_count.append(len(np.where(np.array(P) > 0.5)[0])/n_test)
            # accuracy0 = (TP_predictions_count/10)/n_test
            accuracy0 = max(TP_predictions_count)
            print("TP Accuracy: " + str(accuracy0))
            # X, Y = ascon.make_td_diff( n , 1 , num_rounds-2,data=0)
            # loss0, accuracy0 = model.evaluate(X[:,[i for i in range(7,BLOCK_SIZE,8)]], Y)
            # loss0, accuracy0 = model.evaluate(X,Y)
            # print('loss=', loss0)
            # print('acc=', accuracy0)
            # accuracy0 = 1- accuracy0
            # sys.exit()
            return accuracy1,accuracy0
        TP_arr = []
        ascon.diff_arr = diff_arr
        for l in trange(0,loop):
            X, Y = ascon.make_td_diff( n , 1 , num_rounds,data=1)
            P = model.predict(X[:,[i for i in range(7*BLOCK_SIZE//8,BLOCK_SIZE)]],verbose=0)
            # P = model.predict(X,verbose=0)
            TP_predictions_count = len(np.where(np.array(P) > 0.5)[0])
            # print("Total Real Data: " + str(n) + " | Predicted TP: " + str(TP_predictions_count))
            TP_arr.append(TP_predictions_count)
        for l in trange(0,loop):
            X, Y = ascon.make_td_diff( n , 1 , num_rounds,data=0)
            P = model.predict(X[:,[i for i in range(7*BLOCK_SIZE//8,BLOCK_SIZE)]],verbose=0)
            # P = model.predict(X,verbose=0)
            TP_predictions_count = len(np.where(np.array(P) > 0.5)[0])
            # print("Total Random Data: " + str(n) + " | Predicted TP: " + str(TP_predictions_count))
            TP_arr.append(TP_predictions_count)
        # sys.exit()
        # TP_predictions_count = len(np.where(np.array(TP_arr) >= cutoff)[0])
        print("Prediction Result: ")
        print(TP_arr)
        print("AVG TP in REAL: " + str(statistics.mean(TP_arr[0:loop])))
        print("AVG TP in RANDOM: " + str(statistics.mean(TP_arr[loop:])))
        print("Difference in AVG Data: " + str(statistics.mean(TP_arr[0:loop])-statistics.mean(TP_arr[loop:])))
        print("Min TP in REAL: " + str(min(TP_arr[0:loop])))
        print("Max TP in RANDOM: " + str(max(TP_arr[loop:])))
        print("Difference in Min-Max: " + str(min(TP_arr[0:loop])-max(TP_arr[loop:])))
        # mul_factor = pow(accuracy1-accuracy0,2) if pow(accuracy1-accuracy0,2) > 0.1 else pow(accuracy1-accuracy0,1/3)
        # cutoff = math.ceil(max(2,n*accuracy0+(((n//class_data)*(accuracy1-accuracy0))*acc/2)))
        # cutoff = math.ceil(max(2,n*accuracy0 + ( n*(accuracy1-accuracy0)*(0.5))))
        cutoff = 125161
        print("Difference in Accuracy: " + str(accuracy1-accuracy0) + " | Data Difference Expected: " + str((n//class_data)*(accuracy1-accuracy0)) + " | Cutoff Expected: " + str(cutoff)+" | Data Expected using Calculations: 2^" +str(int(math.log(n,2))))
        TP_Real_count = 0
        TP_Random_count = 0
        
        for i in range(0,len(TP_arr)):
            if (TP_arr[i] > cutoff):
                if (i< loop):
                    TP_Real_count += 1
            else:
                if (i >= loop):
                    TP_Random_count += 1
        print("\nTP_Real Count: " + str(TP_Real_count) + " | TP_Random Count: " + str(TP_Random_count) + " | Accuracy: " + str((TP_Real_count+TP_Random_count)*100/(2*loop))+str("%"))
        xdata = [i+1 for i in range(0,loop)]
        ydata_1 = TP_arr[0:loop]
        ydata_0 = TP_arr[loop:2*loop]
        plt.figure()
        plt.title("Rounds: "+ str(num_rounds) +" | Data Required: 2^" + str(int(math.log(n,2))) )
        plt.xlabel("Experiment No.")
        plt.ylabel("No. of Prediction > 0.5")
        plt.plot(xdata,ydata_1,'o',label = "TP",linestyle=":",color="green")
        plt.plot(xdata,ydata_0,'d', label = "TN",linestyle=":",color="red")
        plt.savefig("images/ASCON320_" + str(num_rounds)+"_rounds_data_2_" + str(str(int(math.log(n,2)))) + ".png"  )
        plt.show()
        # return (n//class_data)*(accuracy1-accuracy0)
        return (TP_Real_count+TP_Random_count)*100/(2*loop)
if __name__ == '__main__':
    num_rounds = 4
    ml_rounds = 4
    acc = 0.5022529959678650 #0.5022529959678650 #0.5027949810028076
    diff_arr = [0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000001]
    accuracy1,accuracy0 = test_prediction(n=2 ** 19,diff_arr=diff_arr,class_data = 2**0,num_rounds=num_rounds,ml_rounds=ml_rounds,acc=acc, accuracy1=0, accuracy0=0, loop=50) 
    # accuracy1 = 0.47846412658691406
    # accuracy0 = 0.4764442443847656
    for i in range(18,19):
        # Accuracy = test_prediction(n=2 ** i,diff_arr=diff_arr,class_data = 2**0,num_rounds=4,ml_rounds=4,acc=0.50, accuracy1=0.04382514953613281, accuracy0=0.04347419738769531, loop=10)  # all 320 bits
        #Accuracy = test_prediction(n=2 ** i,diff_arr=diff_arr,class_data = 2**0,num_rounds=4,ml_rounds=4,acc=0.50, accuracy1=0, accuracy0=0, loop=10)  # 40 bits
        for j in range(0,10):
            Accuracy = test_prediction(n=2 ** i,diff_arr=diff_arr,class_data = 2**0,num_rounds=4,ml_rounds=4,acc=acc, accuracy1=accuracy1, accuracy0=accuracy0, loop=50)  # 40 bits
        print("\n+++++++++++++++++++++++++++++++++++++\n")
        if (Accuracy>98):
            break
        



