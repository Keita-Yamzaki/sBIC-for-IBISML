import numpy as np
from MultipleLayerNetwork_1 import NeuralNetwork
from ReducedRankRegression_3 import ReducedRankRegression
from sBIC_function_4 import sBIC_function
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

#データの読み込み
data = pd.read_csv('severe_cases_daily.csv', encoding = 'utf-8')

#DataFrameにする
data = pd.DataFrame(data)
print(data.columns)
print(data.head())

#欠損値があるデータの取り扱い
#全てのデータがNone or Nanであるデータは取り除く
data.dropna(how = 'all')
#縦軸に関して，すべてのデータがNone or Nanである場合はその軸を削除
data.dropna(axis = 1, how = 'all')
#重複削除
data.drop_duplicates()#()のみでもok


# 特徴量とターゲットを分割
Y = pd.concat([data.iloc[:,1]])
X = pd.concat([data.iloc[:,2:7]])#, data.iloc[:,18:]],axis=1)
#Y = pd.concat([data.iloc[:,3], data.iloc[:,5], data.iloc[:,7], data.iloc[:,9], data.iloc[:,10]],axis=1)  #ターゲット
#X = pd.concat([data.iloc[:,1:3], data.iloc[:,4],data.iloc[:,6],data.iloc[:,8],data.iloc[:,10:]],axis=1)


# CSVファイルに書き出す
scaled = pd.concat([X, Y],axis=1)
scaled.to_csv('output.csv', index=True, encoding = 'utf-8')

#sns.pairplot(data)

data_size = 7
#データをnumpyにする
objective_tmp = Y.to_numpy().tolist()#responses
explanatory_tmp = X.to_numpy().tolist()#covariates
objective_tmp = np.array(objective_tmp) 
objective_tmp = np.expand_dims(objective_tmp, axis=1)
explanatory_tmp = np.array(explanatory_tmp)
print(f'exp_tmp.shape,ojb_tmp.shape={explanatory_tmp.shape,objective_tmp.shape}')
print(f'exp,obj ={explanatory_tmp[0], objective_tmp[0]}')

objective = np.zeros((objective_tmp.shape[0] // data_size+1, data_size, objective_tmp.shape[1]))
explanatory = np.zeros((explanatory_tmp.shape[0] // data_size+1, data_size, explanatory_tmp.shape[1]))
for i in range(objective.shape[0]):
    if i == 0:
        explanatory[i] = explanatory_tmp[i:i+data_size]
        objective[i] = objective_tmp[i:i+data_size]
    elif i != 0:
        explanatory[i] = explanatory_tmp[i+data_size:i+2*data_size]
        objective[i] = objective_tmp[i+data_size:i+2*data_size]
    #objective.append(objective_tmp[i:30+i,:])
    #explanatory.append( explanatory_tmp[i:30+i,:])
objective = np.reshape(objective, (-1, objective_tmp.shape[1]*data_size)) 
explanatory = np.reshape(explanatory, (-1, explanatory_tmp.shape[1]*data_size))
#objective,explanatory = np.array(objective), np.array(explanatory)
print(f'exp.shape,ojb.shape={explanatory.shape,objective.shape}')
print(f'exp,obj ={explanatory[0], objective[0]}')

numSamples = objective.shape[0]
n = [10, 20, 30, 50, 100, numSamples]
#n = [50, 100, 200, 300, 500, numSamples]
#n = [numSamples // 100, numSamples // 50, numSamples //30, numSamples // 20, numSamples //10, numSamples // 5, numSamples // 2, numSamples]#データ数を絞るための変数.本当はリストで指定する
numSimulations = 200


for item in n:
    estimated_sBIC = []
    estimated_BIC = []
    estimated_AIC = []
    #batch size,train size
    batch_size = explanatory.shape[0] * item  // numSamples
    #obj_size = 30
    train_size = explanatory.shape[0]

    for i in range(numSimulations):
        #if i % 10 == 0:
        #    print(f'i={i}回目')
        #X, Y = simRR(N = N, M = M, n = item, r = r)
        #X, Y = explanatory[:item,:].T, objective[:item,:].T
        batch_mask = np.random.choice(train_size, batch_size)
        #batch_mask_obj = np.random.choice(train_size, obj_size)
        X = explanatory[batch_mask,:].T
        Y = objective[batch_mask,:].T
        rrr = ReducedRankRegression(X = X, Y = Y, maxRank = min(X.shape[0], Y.shape[0]))

        numModels = rrr.getNumModels()

        class_sBIC = sBIC_function(X = X, Y = Y)
        result = class_sBIC.calc_sBIC()

        estimated_sBIC.append(np.argmax(result['sBIC']))
        estimated_BIC.append(np.argmax(result['BIC']))
        estimated_AIC.append(np.argmax(result['AIC']))

    print(f'es_sBIC = {len(estimated_sBIC)}')
    print(f'es_BIC = {len(estimated_BIC)}')
    print(f'es_AIC = {len(estimated_AIC)}')

    # ユニークな値を取得
    unique_values = [i-1 for i in range(1, numModels+1)]#sorted(set(estimated_sBIC + estimated_BIC))

    # ユニークな値ごとの度数を計算
    sBIC_counts = [estimated_sBIC.count(value) for value in unique_values]
    BIC_counts = [estimated_BIC.count(value) for value in unique_values]
    AIC_counts = [estimated_AIC.count(value) for value in unique_values]
    #print(f'waic={WAIC_counts}')

    # 棒グラフを描画
    width = 0.25
    x = np.arange(0, numModels ,1)
    print(f'len = {len(unique_values)}')
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, sBIC_counts, width, label='sBIC')
    rects2 = ax.bar(x , BIC_counts, width, label='BIC')
    rects3 = ax.bar(x + width, AIC_counts, width, label='AIC')

    ax.set_xlabel('estimated rank')
    #ax.set_ylabel('number of estimated rank by the information criterion')
    ax.set_title(f'n = {item}')
    ax.set_xticks(x)
    ax.set_xticklabels(unique_values)
    ax.set_ylim([0, numSimulations])
    ax.legend(loc = 'upper left')

    plt.show()

#繰り返し回数を指定
num_epochs_lr = 10 #learning coefficientの変更用
num_epochs_nn = min(explanatory.shape[1],objective.shape[1]) #neural networkの層の厚みの変更用
iter_num = 500 #学習回数

#学習
numSimulations_nn = 100
for i in range(num_epochs_lr):
    model_number = []
    model_number_counts = []
    for k in range(numSimulations_nn):
        whole_loss = []
        for j in range(num_epochs_nn):
                #print((i+1,j+1),"回目")
                #ニューラルネットワークの層の厚みの決定
            model_shape = [explanatory.shape[1],1+j,objective.shape[1]] #ここで層の厚みを決める
            #インスタンス
            model = NeuralNetwork(layers = model_shape)

            #optimizer
            optimizer = optim.SGD(model.parameters(), lr = 0.001*(i+1)) #ADAM,RMSprop,SDG? #0.004
            #loss function
            criterion = nn.MSELoss()#nn.L1Loss() #CrossEntropyLoss, BCEWithLogitsLoss,mse,l1とmseを同時に
            
            predictions_list = []
            loss_list = []
                
            #batch size,train size
            batch_size = 32#explanatory.shape[0] // 
            train_size = explanatory.shape[0]

            for k in range(iter_num):
                #処理時間計測
                start_time = time.time()

                #学習用データの取り出し
                batch_mask = np.random.choice(train_size, batch_size)
                explanatory_batch = explanatory[batch_mask,:]
                objective_batch = objective[batch_mask,:]

                #学習データをtensorに変更
                explanatory_batch= torch.tensor(explanatory_batch).float()
                objective_batch = torch.tensor([objective_batch]).float()

                #prediction
                predictions_batch = model.forward(explanatory_batch) 
                    
                #loss
                loss_sum = 0 #lossのリセット
                loss = criterion(predictions_batch, objective_batch)

                #update parameters
                
                optimizer.zero_grad() #勾配のリセット
                loss.backward() #backward
                optimizer.step() #勾配の更新

                # テンソルから計算グラフを切り離す
                detached_loss = loss.detach()
                # NumPy 配列に変換
                detached_loss = detached_loss.numpy()   
            
                #lossの計算と格納
                loss_sum += detached_loss
                loss_list.append(loss_sum)
                #print(loss_sum)

                #処理時間計測
                end_time = time.time()
                elapsed_time = end_time - start_time
                #print(f"実行時間:{elapsed_time} 秒")
                
            whole_loss.append(loss_sum)

            #学習済みのパラメータでの予測(全データ)
            for input_data, target in zip(explanatory, objective):
                #tensorに変更
                input_data = torch.tensor(input_data).float()
                target = torch.tensor([target]).float()

                #prediction
                predictions = model(input_data) 

                #tensorからデータの切り離し
                detached_predictions = predictions.detach() 
                detached_predictions = detached_predictions.numpy()        
                predictions_list.append(detached_predictions)
                

            x_length = np.arange(len(loss_list))
            #plt.plot(x_length, loss_list, label='train loss')
            #plt.xlabel("number of trials")
            #plt.ylabel("loss")
            #ラベルの位置
            #plt.legend(loc='upper right')
            #plt.show()s
            
        print(f'whole_loss = {whole_loss}')
        whole_loss = np.array(whole_loss)
        print(f'whole_loss_index = {np.argmin(whole_loss)+1}')
        model_number.append(np.argmin(whole_loss)+1)

    # ユニークな値を取得
    unique_values = [i-1 for i in range(1, (numModels)+1)]#sorted(set(estimated_sBIC + estimated_BIC))

    # ユニークな値ごとの度数を計算
    model_number_counts = [model_number.count(value) for value in unique_values]
    print(f'model_number_count = {model_number_counts}')
    # 棒グラフを描画
    width = 0.35
    x = np.arange(1, (numModels)+1  ,1)
    print(f'len = {len(unique_values)}')
    fig, ax = plt.subplots()
    rects2 = ax.bar(x , model_number_counts, width)

    ax.set_xlabel('estimated rank')
    #ax.set_ylabel('number of estimated rank by learning')
    #ax.set_title(f'true model(true rank)')
    ax.set_xticks(x)
    ax.set_xticklabels(unique_values)
    ax.set_ylim([0, numSimulations_nn])
    #ax.legend(loc = 'upper right')

    plt.show()