#calculate sBIC

#import libraries
import numpy as np
import math
from ReducedRankRegression_3 import ReducedRankRegression

#define class sBIC_function
class sBIC_function:
    #初期設定
    def __init__(self, X, Y):
        #クラスのインスタンス
        self.model = ReducedRankRegression(X = X, Y = Y, maxRank = min(X.shape[0], Y.shape[0]))
        
        self.numModels = self.model.getNumModels()
        self.topOrder = self.model.getTopOrder()
        self.n = self.model.getNumSamples()
        self.p = self.model.getPrior()

        self.X, self.Y = X, Y

    def calc_sBIC(self):
        #到達可能なノードのリストを作成
        reach = [[] for _ in range(self.numModels)]
           
        for i in self.topOrder:
            #各モデルについて,モデルが依存するモデルのリストを取得
            parents = self.model.parents(i)
            for j in parents:
                #j = 0のとき,ランク0のモデルなので,親モデルなし, j=1も同様
                if j is not None:
                    reach[i] = list({j} | set(reach[j])) #change
        #print(f'reach={reach}')

        #L^\prime(M_i)の計算に必要なもののリストを用意
        #L^\prime_{ij}を対数とったもの,L_{ij}
        logLij = [[None for _ in range(len(self.topOrder))] for _ in range(len(self.topOrder))]
        Lij = [[None for _ in range(len(self.topOrder))] for _ in range(len(self.topOrder))]
        #L^\prime_{ii}を対数とったもの
        logLii = [None] * self.numModels
        #P(Y_n|hat{pi}_i, M_i)を対数とったもの
        logLike = [None] * self.numModels    

        for i in self.topOrder:
            #対数尤度の計算
            logLike[i] = self.model.logLikeMle(i)

            #学習係数の計算
            lr = self.model.learnCoef(i, i)

            #log(L^\prime_{ii})の計算
            logLii[i] = logLike[i] - lr['lambda'] * math.log(self.n)
            
            #log(L^\prime_{ij})の計算
            for j in reach[i]:
                lr = self.model.learnCoef(i, j)
                logLij[i][j] = logLike[i] - lr['lambda'] * math.log(self.n) + (lr['m'] - 1) * math.log(math.log(self.n))

        #正則化項(最大値を用いる)
        logLij_flat = [item for sublist in logLij for item in sublist]
        combined = logLij_flat + logLii
        mn = max(item for item in combined if item != None)

        #正則化項を用いて,対数をもとに戻す(logA-> exp(logA)=A)
        #logL_{ii} -> L_{ii}
        Lii = [math.exp(item - mn) for item in logLii]
        #Lii = [math.exp(item) for item in logLii]
        #logL_{ij} -> L_{ij}
        for i in self.topOrder:
            for index, value in enumerate(logLij[i]):
                if value is None:
                    Lij[i][index] = None
                else:
                    Lij[i][index] = math.exp(value - mn)
                    #Lij[i][index] = math.exp(value)

        #L^\prime(M_i)の計算
        L = [0] * self.numModels
        for i in self.topOrder:
            #i=0のとき,式(3.11)より解はL^\prime_i = L^\prime_{ii}
            if i <= 1:#model == 0 or model == 1:,reach[0],reach[1]のみ[] #change
                L[i] = Lii[i]
            else:
                #L^\prime(M_i)に関する二次方程式の係数
                a = self.p[i]
                b_tmp, c_tmp = 0, 0
                for j in reach[i]:
                    b_tmp += L[j] * self.p[j]
                    c_tmp += L[j] * self.p[j] * Lij[i][j]
                b = b_tmp - Lii[i] * self.p[i]
                c = - c_tmp

                #二次方程式を解く
                L[i] =  1 / (2 * a) * (- b + math.sqrt(b ** 2 - 4 * a * c)) #サンプル数が多いとエラーが起こる

        #print(f'L = {L}')
        if any(value < 0 for value in L):
            print("Negative probabilities found, likely due to rounding",
            "errors, rounding these values to 0.") #change #このエラー出ると次のsBICの計算でエラー出る

            #各要素に対して0未満の部分は0で置き換え
            L = np.maximum(L,np.zeros(len(L)))         

        #sBICの計算
        logL_tmp = np.log(L)
        sBIC = [value + mn for value in logL_tmp]
        #sBIC = [value for value in logL_tmp]

        #BICの計算
        dimensions = self.model.getDimension(self.numModels)
        BIC = [l - d / 2 * math.log(self.n) for l, d in zip(logLike, dimensions)]

        #AICの計算
        AIC = [2 * l / self.n - 2 * d / self.n  for l, d in zip(logLike, dimensions)]
        #print(f'loglike = {-2 * l / math.log(self.n)}'  for l in logLike)
        #print(f'dimensions = {dimensions}')
        #"print(f'AIC={AIC}')
        #TICの計算
        #TIC = [- l / math.log(self.n) + 1 / self.n *  for l in logLike]

        #WBIC = []
        #WAICの計算
        #WAIC = [l / math.log(self.n) - d / (2 * math.log(self.n)) + 3 for l, d in zip(logLike, dimensions)]
        #WAIC = []
        #for i in self.topOrder:
        #    log_likelihoods = []
        #    log_likelihoods = self.model.logLikeMle1(i)
            
            #loglike_flat = log_likelihoods.flatten()#[item for sublist in log_likelihoods for item in sublist]
        #    log_likelihoods_prime = np.array(log_likelihoods)
        #    log_likelihoods_prime = log_likelihoods_prime[np.newaxis, :]
            #log_likelihoods_prime = log_likelihoods.T
            #log_likelihoods = np.reshape(log_likelihoods, (self.numModels,-1))
            #print(f'log_likelihoods={log_likelihoods}')
        #    mn_prime = max(item for item in log_likelihoods if item != None)
            #V_nを求める.
        #    loglike_matrix = np.dot(log_likelihoods_prime.T, log_likelihoods_prime)
        #    V_n = np.mean(np.diag(loglike_matrix)) - (np.mean(log_likelihoods) ** 2)
        #    T_n = 0.5*math.log(2*math.pi *(self.n+2)/(self.n+1)) + (self.n+1)/((self.n+2)*2) *np.mean((log_likelihoods -np.sum(log_likelihoods)/(self.n+1) )**2)  - np.log(np.mean(np.exp(log_likelihoods)))
        #    WAIC.append(T_n + V_n)

        #結果のリスト
        results = {
            "logLike": logLike,
            "sBIC": sBIC,
            "AIC": AIC,
            #"TIC": TIC,
            "BIC": BIC,
            #"WBIC": WBIC,
            "modelPoset": self.model
        }      
    
        return results      