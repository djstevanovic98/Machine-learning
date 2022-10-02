# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 23:24:51 2021

@author: dzou
"""

import numpy as np
import pandas as pd

output_path = 'D:/Python Projekti/Momenat/output.csv'
test_path = 'D:/Python Projekti/Momenat/test.csv'

input_path_domaci = 'D:/Python Projekti/Momenat/NewYork.csv'
input_data_1 = pd.read_csv(input_path_domaci, sep=',')

all_x_1 = np.array(input_data_1.loc[:100, 'f(t)'], dtype=np.float64)
       
def napraviUlaze(array,brojUlaza,end):
    matrica = []
    for i in range(0,end):
        for k in range(brojUlaza):
            matrica.append(array[k+i])
    reshaped_matrica = np.reshape(matrica, (-1,brojUlaza))
    finalnaMatrica = -np.ones((len(reshaped_matrica), brojUlaza+1))
    finalnaMatrica[:,1:] = reshaped_matrica
    return finalnaMatrica

TDNN = napraviUlaze(all_x_1, 5, 95)
TDNN_1 = napraviUlaze(all_x_1, 10, 90)
TDNN_2 = napraviUlaze(all_x_1,15, 85)

izlazi = np.array(input_data_1.loc[5:,'f(t)']).transpose()
izlazi_1 = np.array(input_data_1.loc[10:,'f(t)']).transpose()
izlazi_2 = np.array(input_data_1.loc[15:,'f(t)']).transpose()


# Логистички сигмоид
def logsig(x):
    return 1 / (1 + np.exp(-x))

# Извод сигмоида
def logsigder(x):
    s = logsig(x)
    return s * (1 - s)

def calculate_output(w_1, w_2, input_l):
    y_1 = input_l * w_1
    y_1 = logsig(np.sum(y_1, 1))  
    y_1 = np.hstack(([-1], y_1))
    y_2 = logsig(np.sum(y_1 * w_2))
    return y_2

def calculate_output_pravi(w_1, w_2, input_l):
    y_1 = input_l * w_1
    y_1 = logsig(np.sum(y_1, 1))  
    y_1 = np.hstack(([-1], y_1))
    y_2 = np.sum(y_1 * w_2)
    return y_2

# Средња квадратна грешка
def mse(w_1, w_2, inputs, outputs):
    p = len(inputs)
    e = 0
    for i in range(p):
        err = calculate_output(w_1, w_2, inputs[i]) - outputs[i]
    e += 0.5 * err * err
    return e / p

confs = []
confs_1 = []
confs_2 = []

with open(output_path, 'w') as fout:
  # Заглавље... редни број тренинга, улазне тежине, коначне тежине и број епоха
    fout.write('Training MSE Epochs\n')
    eta = 0.1
    epsilon = 0.5e-6
    alpha = 0.8
  
    mse_per_epoch_0 = []
    mse_per_epoch_1 = []
    mse_per_epoch = []
  
    mse_per_training = []
    mse_per_training_1 = []
    mse_per_training_2 = []
  
    for m in range(3):
        #podesavamo parametre ulaza:
        fout.write('Mreza %d\n' % (m+1))
        if m == 0:
            ulazni_sloj= 5 #broj ulaza
            drugi_sloj = 10 #broj neurona
            ulaz = TDNN #postavljamo ulaze koji zavise od mreze
            izlaz = izlazi #isto za izlaze
        elif m == 1:
            ulazni_sloj = 10
            drugi_sloj = 15
            ulaz = TDNN_1
            izlaz = izlazi_1
            mse_per_epoch_0 =mse_per_epoch #cuvamo greske prve mreze
        else:
            ulazni_sloj = 15
            drugi_sloj = 25
            ulaz = TDNN_2
            izlaz = izlazi_2
            mse_per_epoch_1 =mse_per_epoch #cuvamo greske druge mreze  
        
        mse_per_epoch = []
        print('Trening mreze broj: ',m)
        for i in range(1, 4):
            mse_per_epoch.append([])
            w_1 = np.random.uniform(size=(drugi_sloj, ulazni_sloj+1)) #10*6
            w_1_old = []
            w_2 = np.random.uniform(size=(1, drugi_sloj+1))
            w_2_old = []
            epoch = 0
            diff = 1
            error = mse(w_1, w_2, ulaz, izlaz)
            mse_per_epoch[i-1].append(error)
        
            while diff>epsilon:
                prev_error = error
                for k in range(len(ulaz)):
                    # Кораци унапред
                    i_1 = np.sum(ulaz[k] * w_1, 1)
                    y_1 = logsig(i_1)
                    y_11 = np.hstack(([-1], y_1))
                    i_2 = np.sum(y_11 * w_2, 1)
                    y_2 = logsig(i_2)
                    # Кораци уназад
                    delta_2 = (izlaz[k] - y_2) * logsigder(i_2)
                    #cuvamo w_2
                    w_2_old_save = w_2
                    w_2 = w_2 + eta * delta_2 * y_2
                    if len(w_2_old) != 0:
                        w_2 = w_2 + alpha*(w_2_old-w_2_old_save)
                    w_2_old = w_2_old_save
            
                    delta_1 = - np.sum(delta_2 * w_2, 1) * logsigder(i_1)
                    # Пошто је резултат низ, транспонујемо га при додавању да би се сваки елемент доделио одговарајућем неурону
                    w_1_old_save = w_1
                    w_1 = w_1 + (eta * delta_1 * y_1)[np.newaxis].transpose()
                    if len(w_1_old) != 0:    
                        w_1 = w_1 + alpha*(w_1-w_1_old)
                    w_1_old = w_1_old_save
                error = mse(w_1, w_2, ulaz, izlaz) 
                epoch += 1
                diff = abs(error - prev_error)
                mse_per_epoch[i-1].append(error)
                if epoch % 10 == 0:
                    print('Training %d epoch %d mse %.6f diff %f' % (i, epoch, error, diff))
                if epoch == 2000:
                    break;
        
            if(m==0):
                confs.append((w_1, w_2))
                mse_per_training.append(error)
            elif(m==1):
                confs_1.append((w_1, w_2))
                mse_per_training_1.append(error)
            else:
                confs_2.append((w_1, w_2))
                mse_per_training_2.append(error)
            fout.write('%d %f %d\n' % (i, error, epoch))
print('Done.')


#Testiranje uzoraka
test_data = pd.read_csv(test_path, sep=',')

test_data_1 = np.array(test_data.loc[:20, 'f(t)'], dtype='float')
print(test_data_1)

TDNN_td = napraviUlaze(test_data_1, 5, 15)
TDNN_td_1 = napraviUlaze(test_data_1, 10, 10)
TDNN_td_2 = napraviUlaze(test_data_1,15,5)

izlazi_td = np.array(test_data.loc[5:,'f(t)']).transpose()
izlazi_td_1 = np.array(test_data.loc[10:,'f(t)']).transpose()
izlazi_td_2 = np.array(test_data.loc[15:,'f(t)']).transpose()

output_val = []
relative_err = []
for t in range(len(confs)):
  relative_err.append([])
  output_val.append([])

# Исписујемо резултате за сваки тест, за сваку комбинацију тежина са тренинга
output_val_td = []
output_val_td_1 = []
output_val_td_2 = []

mean_relative_err_0 = []
mean_relative_err_1 = []
mean_relative_err_2 = []

for m in range(3):
    if m == 0:
        print('\n------------MREZA 1-------------')
        ulazi_td = TDNN_td
        izlaz_td = izlazi_td
        confs_td = confs
    elif m==1:
        print('\n------------MREZA 2-------------')
        ulazi_td = TDNN_td_1
        izlaz_td = izlazi_td_1
        confs_td = confs_1
    else:
        print('\n------------MREZA 3-------------')
        ulazi_td = TDNN_td_2
        izlaz_td = izlazi_td_2
        confs_td = confs_2
    
    if m==1:
        output_val = []
        relative_err = []
        
        for t in range(len(confs_1)):
            relative_err.append([])
            output_val.append([])
    elif m==2:
        output_val = []
        relative_err = []
        for t in range(len(confs_2)):
            relative_err.append([])
            output_val.append([])
        
    for k in range(len(ulazi_td)):
        x = ulazi_td[k]
        for t in range(len(confs_td)):
            w_1, w_2 = confs_td[t]
            output = calculate_output(w_1, w_2, x)
            relative_err[t].append(abs((output - izlaz_td[k])/izlaz_td[k]))
            output_val[t].append(output)
            print('Primer %d, iz treninga %d vrednost %f' % (k+1, t+1, output, ))
    mean_relative_err = [sum(relative_err[k])/len(relative_err[k]) for k in range(len(relative_err))]
    if m==0:
        output_val_td = output_val
    elif m==1:
        output_val_td_1 = output_val
    else:
        output_val_td_2 = output_val
    print('Relativne greške treninga', mean_relative_err)
    

print('\nNajbolji proces treniranja prve mreze-kandidat je: ', mse_per_training.index(min(mse_per_training)) + 1)
print('Najbolji proces treniranja druge mreze-kandidat je: ', mse_per_training_1.index(min(mse_per_training_1)) + 1)
print('Najbolji proces treniranja trece mreze-kandidat je: ', mse_per_training_2.index(min(mse_per_training_2)) + 1)
print()

#Varijansa
for m in range(3):
    if m==0:
        output_val_td_print = output_val_td
    elif m==1:
        output_val_td_print = output_val_td_1
    else:
        output_val_td_print = output_val_td_2
    
    mean = [sum(output_val_td_print[k])/len(output_val_td_print[k]) for k in range(len(output_val_td_print))]
    stdevsq = [[(output_val_td_print[k][j] - mean[k])**2 for j in range(len(output_val_td_print[k]))] for k in range(len(output_val_td_print))]
    variance = [sum(stdevsq[k]) / len(stdevsq[k]) for k in range(len(stdevsq))]
    print('Varijanse treninga', variance)

for m in range(3):
    if m==0:
        izlazi_varijansa = izlazi_td
    if m==1:
        izlazi_varijansa = izlazi_td_1
    if m==2:
        izlazi_varijansa = izlazi_td_2
        
    meant = sum(izlazi_varijansa)/len(izlazi_varijansa)
    stdevsqt = [(d - meant)**2 for d in izlazi_varijansa]
    variancet = sum(stdevsqt) / len(stdevsqt)
    print('Varijansa testa', (m+1),variancet)



TDNN_krajnji = np.array(input_data_1.loc[95:,'f(t)']).transpose()
TDNN_krajnji_1 = np.array(input_data_1.loc[90:,'f(t)']).transpose()
TDNN_krajnji_2 = np.array(input_data_1.loc[85:,'f(t)']).transpose()

TDNN_krajnji = np.insert(TDNN_krajnji, 0, -1, axis=0).transpose()
TDNN_krajnji_1 = np.insert(TDNN_krajnji_1, 0, -1, axis=0).transpose()
TDNN_krajnji_2 = np.insert(TDNN_krajnji_2, 0, -1, axis=0).transpose()
#proveri sve

output_val_krajnji = []
output_val_krajnji_1 = []
output_val_krajnji_2 = []

najbolji_1 = mse_per_training.index(min(mse_per_training))
najbolji_2 = mse_per_training_1.index(min(mse_per_training_1))
najbolji_3 = mse_per_training_2.index(min(mse_per_training_2))


for m in range(3):
    if(m==0):
        TDNN_krajnji_td = TDNN_krajnji
        confs_td = confs[najbolji_1]
    if(m==1):
        TDNN_krajnji_td = TDNN_krajnji_1
        output_val = []
        for t in range(len(confs_1)):
            output_val.append([])
        confs_td = confs_1[najbolji_2]
    elif(m==2):
        TDNN_krajnji_td = TDNN_krajnji_2
        output_val = []
        for t in range(len(confs_2)):
            output_val.append([])
        confs_td = confs_2[najbolji_2]
            
    for k in range(20):
        x=TDNN_krajnji_td 
        
        for t in range(len(confs_td)):
            w_1, w_2 = confs_td
            output = calculate_output_pravi(w_1, w_2, x)
        if(m==0):
            TDNN_krajnji_td[5] = output
            if(k==19):
                output_val_krajnji.append(output)
        elif(m==1):
            TDNN_krajnji_td[10] = output
            if(k==19):
                output_val_krajnji_1.append(output)
        else:
            TDNN_krajnji_td[15] = output
            if(k==19):
                output_val_krajnji_2.append(output)
        
        TDNN_krajnji_td[0] = -1

print('\nProcena mreze 1: ')
print(output_val_krajnji)
print('\nProcena Mreze 2: ')
print(output_val_krajnji_1)
print('\nProcena Mreze 3')
print(output_val_krajnji_2)


