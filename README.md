# Improved-Differential-Distinguisher-GIFT128-ASCON
Source code for **ML based Improved Differential Distinguisher with High Accuracy: Application to GIFT-128 and ASCON** 
* eprint - https://eprint.iacr.org/2024/1370.pdf

## Source Code for Improved Differential Distinguisher for GIFT128 and ASCON

### There are four files in this source code.
There are 8 files and 2 folders in this repository.

**Cipher Implementartion:**
* gift.py
* ascon.py

**Training the Differential Distinguisher**
* train_distinguisher.py

**Constructing Improved Differential Distinguisher**
* improve_distinguisher.py

**Trained Model Folder:**
* saved_models_paper
  
**TP and TN Graphs**
* graphs

**Execution Parameters**
* ExecutionParameters.txt

**Results**
* Results_train_distinguisher.txt
* Results_improve_distinguisher.txt

**README.MD**

## Instructions for Execution 
### To Train the Distinguisher 
**Input Parameters are:**
* CIPHER_NAME (GIFT128 or ASCON320)
* bit_size (16/128 for GIFT238 and 40/320 for ASCON320)
* input_diff (input difference for which ML distinguisher is to be trained)

**Execution:**
```python train_distinguisher.py```
* Models will be saved in saved_models folder

*Sample Output of Executions with parameters is stored in ```Results_train_distinguisher.txt```*

### To Contruct the Improved Distinguisher with High Accuracy ###
**Input Parameters are:**
* CIPHER_NAME (GIFT128 or ASCON320)
* num_rounds (total no. of rounds)
* ml_rounds (no. of rounds for which ML distinguishers is trained, it will be same as num_round if no differential trail is added to construct differential-ML distinguisher)
* class_data (data complexity of differential trail in case of Differential-ML distinguisher otherwise 1)
* bit_size (no. of bits used for training/prediction)
* acc (accuracy of trained ML distinguisher)
* input_diff (input difference for the complete disintigusher)
* input_diff_ML (input difference to the ML distinguisher - it will be same as input_diff if no differential trail is added to construct differential-ML distinguisher)
* loop (no. of time the TP and TN experiments are performed)
* validate (False for construction and True for validation)
* beta (data complexity(β) derived from construction phase, required only if validate is True)
* C_T (cutoff(C<sub>T</sub>) derived from construction phase, required only if validate is True)
  
**Execution:**
```python improve_distinguisher.py```
* Model will be used from *saved_models_paper* folder. Change *saved_models_paper* to *saved_models* if new models to be used. 
* Graphs will be stored in *graphs* folder

*Sample Output of Executions with parameters is stored in ```Results_improve_distinguisher.txt```*

## Acknowledgement ##
1. Shen, D., Song, Y., Lu, Y., Long, S., Tian, S.: Neural differential distinguisher for GIFT-128 and ASCON. Journal of Information Security and Applications, vol. 82, (2024) ([https://github.com/agohr/deep_speck](https://github.com/yijSong/ND-GIFT-ASCON))
