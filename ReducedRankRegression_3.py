#Reduced Rank Regression_3

#import libraries
import numpy as np 
import igraph #モデルの対応付けのため

#parentsとDimensionについては元のプログラム通りでないので注意

#define class:Reduced Rank Regression
class ReducedRankRegression:
    #初期設定
    def __init__(self, X, Y, maxRank):
        #X, Y:input, output
        self.X, self.Y = X, Y

        #numResponses, numCovariates:目的変数の次元, 説明変数の次元
        self.numResponses = self.Y.shape[0] #change
        self.numCovariates = self.X.shape[0] #change

        #考慮するモデル数(実際は最初のモデルは無視するのでmaxRankが考慮するモデル数)
        self.numModels = maxRank + 1

        #各モデルに対する事前確率(一様分布)
        self.prior = np.ones(self.numModels) / self.numModels

        #考慮する最大ランク
        self.maxRank = maxRank

        #各モデルの次元数
        self.dimension = [None] * self.numModels

        #log likelihood
        self.logLikes = [None] * self.numModels

        #MLE(無制約最尤推定量（Unconstrained Maximum Likelihood Estimate),self.numResponses*self.numCovariates行列が入る
        self.unconstrainedMLE = None

        #posetの構築
        if self.numModels == 1:
            # モデルが1つだけの場合は空のグラフを作成
            self.E = np.array([]).reshape(0, 2)
            self.posetAsGraph =igraph.Graph(edges=self.E.tolist(),directed = True)
        else:
            # 複数のモデルがある場合、エッジリストを作成してグラフを構築
            self.E = np.column_stack((np.arange(1, self.numModels-1), np.arange(2, self.numModels)))
            self.posetAsGraph = igraph.Graph(edges=self.E.tolist(), directed=True)

        # トポロジカルソートを使用してモデルの順序を決定
        self.topOrder = np.array(self.posetAsGraph.topological_sorting())
    
    #データの設定
    def setData(self):
        if self.X.shape[1] != self.Y.shape[1]: 
            raise ValueError("Input data XY has incorrect dimensions.")

        #目的変数の次元,説明変数の次元
        X, Y = np.array(self.X), np.array(self.Y)

        return X, Y
    
    #考慮するモデル数を返す    
    def getNumModels(self):
        return self.numModels
    
    #サンプル数を返す
    def getNumSamples(self):
        return self.X.shape[1]
    
    #モデル順序を返す
    def getTopOrder(self):
        return self.topOrder
    
    #事前確率を返す
    def getPrior(self):
        return self.prior
    
    #親モデルを返す #change
    def parents(self, model):
        #モデルの番号が対象外だとエラー
        if model > self.numModels or model < 0:
            raise ValueError("Invalid model number.")
        
        #model == 1のとき,rank=0が親モデルになるのでこれもnoneがよい?
                
        #最初のモデルには親モデルなし
        if model == 0 or model == 1:
            return [None]
        #それ以外のモデルにはランクが一つ小さいモデルを返す
        else:
            return [model - 1]
        
    #モデル番号が有効範囲内か確認する
    def isValidModel(self, model):
        #returnはTrue or Falseで返す
        return 0 <= model <= self.numModels
                  
    #最尤推定量(maximum likelihood estimation,MLE)
    def logLikeMle(self, model):
        if not self.isValidModel(model):
            raise ValueError("Invalid model number.")
        
        #rank
        H = model #change

        _, Yhat, S = self.logLikeMleHeler()

        M, N = self.numCovariates, self.numResponses

        if H < min(M, N):
            ell = - 0.5 * np.sum((self.Y - Yhat) ** 2) - 0.5 * np.sum(S[1][H:len(S[1])+1] ** 2)
        else:
            ell = - 0.5 * np.sum((self.Y - Yhat) ** 2)

        self.logLikes[model] = ell

        return ell

    #MLE計算の補助関数
    def logLikeMleHeler(self):
        if not isinstance(self.unconstrainedMLE, np.ndarray):
            # 無制約最尤推定量が未定義の場合、計算を行う
            self.unconstrainedMLE = np.dot(self.Y, np.dot(self.X.T, np.linalg.pinv(np.dot(self.X, self.X.T))))
            self.Yhat = np.dot(self.unconstrainedMLE, self.X)
            self.S = np.linalg.svd(self.Yhat,  full_matrices=False) #特異値分解   

        return self.unconstrainedMLE, self.Yhat, self.S
    
    #学習係数     
    def learnCoef(self, superModel, subModel):
        M = self.numCovariates
        N = self.numResponses
        H = superModel
        r = subModel

        if r > H:
            return self.learnCoef(H, H)

        # 各ケースに応じた計算
        if (N + r <= M + H) and (M + r <= N + H) and (H + r <= M + N):
            if ((M + H + N + r) % 2) == 0:
                m = 1
                lambda_value = -((H + r) ** 2) - (M ** 2) - (N ** 2) + 2 * (H + r) * (M + N) + 2 * M * N
                lambda_value /= 8
            else:
                m = 2
                lambda_value = -((H + r) ** 2) - (M ** 2) - (N ** 2) + 2 * (H + r) * (M + N) + 2 * M * N + 1
                lambda_value /= 8
        elif M + H < N + r:
            m = 1
            lambda_value = H * M - H * r + N * r
            lambda_value /= 2
        elif N + H < M + r:
            m = 1
            lambda_value = H * N - H * r + M * r
            lambda_value /= 2
        elif M + N < H + r:
            m = 1
            lambda_value = M * N / 2

        return {'lambda': lambda_value, 'm': m}     
    
    #モデルの次元 #change
    def getDimension(self, model):
        if not self.isValidModel(model):
            raise ValueError("Invalid model number.") 
        
        for i in range(model):
            if self.dimension[i] is None:
                lr = self.learnCoef(i, i)
                lambda_i = lr['lambda']
                self.dimension[i] = 2 * lambda_i

        return self.dimension #change #self.dimension[model-1]