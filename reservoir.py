import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from scipy.stats import mode
import pandas as pd
class Reservoir:
    def __init__(self, shape, network = 'random', size = 100, res='tudo', W=[], Win=[], mu = 0, sigma = 0.08, nGrupos=5, distancia = 1):
        self.mu = mu
        self.shape = shape
        self.sigma = sigma
        self.nGrupos = nGrupos
        self.size = size
        self.res = res
        self.iniciaWin()
        self.iniciaW(network)
        self.distancia = distancia

    def aleatorio(self):
        first = True
        W=[]

        while(first or max(abs(np.linalg.eigvals(W)))>=1):
            first = False
            proporc = int((self.size**2)*(0.2))
            W = np.concatenate((np.random.normal(self.mu, self.sigma, proporc),(np.zeros((self.size**2)-proporc))), axis=0)
            np.random.shuffle(W)
            W = W.reshape((self.size,self.size))
            
        for x in range(self.size):
            W[x][x]=0
        #print('Aleatoria')
        return W



    def pequenoMundo(self, taxa = 0.1, ligacoes=0):
        first = True

        if ligacoes == 0:
            ligacoes = int(self.size/(2*self.nGrupos))
        W=[]
        while(first or max(abs(np.linalg.eigvals(W)))>=1):
            first = False
            index = 0
            W = np.zeros((self.size,self.size))
            for x in range(self.size):
                for y in range(-ligacoes, +ligacoes+1):
                    if y==0: 
                        continue
                    W[x][(x+y+self.size)%self.size] = 1
            pos = []
            for x in range(self.size):
                for y in range(self.size):
                    if W[x][y] == 1 and np.random.rand() < taxa :
                        W[x][y] = 0
                        W[x][x] = 1
                        v = np.random.choice(np.transpose(np.argwhere(W[x]==0))[0])

                        W[x][x] = 0
                        W[x][v] = 1

            pos = np.transpose(np.nonzero(W))
            #print(len(pos))
            valores = np.random.normal(self.mu, self.sigma, len(pos))
            for index, p in enumerate(pos):
                W[p[0]][p[1]] = valores[index] 
        if taxa == 0:    
            #print('Regular')
            pass
        if taxa == 0.1:
            #print('Mundo')
            pass
        return W   

    def iniciaWin(self):
        proporc = int((self.size*self.shape[2])*(0.8))
        Win = np.concatenate((np.random.normal(self.mu, self.sigma, proporc),(np.zeros((self.shape[2]*self.size)-proporc))), axis=0)
        np.random.shuffle(Win)
        Win = Win.reshape((self.size,self.shape[2]))    
        #Win = np.concatenate((np.ones(self.size).reshape((self.size,1)), Win), axis=1)

        #print(Win.shape)
        self.Win = np.around(Win,4)

    def iniciaW(self, network):
        if network == 'Regular':
            self.W = np.around(self.pequenoMundo(taxa=0),4)
        elif network == 'Mundo':
            self.W = np.around(self.pequenoMundo(),4)
        else:
            self.W = np.around(self.aleatorio(),4)
    def transform(self, data):
        cont=0
        saida = []
        data = np.asarray(data)
        for dado in data:


            cont+=1
            vetor = np.zeros(self.size).reshape((self.size,1))
            vector = []
            index = 0
            for x in dado:
                valorEscolhido = 0.9
                produtoWin = np.dot(self.Win, x.reshape(len(x),1))
                produtoW = np.dot(self.W, vetor)
                vetor = np.sum(((1-valorEscolhido)*vetor , valorEscolhido*np.tanh(np.sum(( produtoWin, produtoW), axis=0))),axis=0)
                if index % self.distancia == 0:
                    vector.append(copy.copy(vetor))
                index+=1

            if self.res == 'ultima':
                saida.append(copy.copy(vetor))
            elif self.res == 'tudo':
                saida.append(copy.copy(vector))
    
        saida = np.array(saida) 
        #print(saida.shape)
        saida.shape = (saida.shape[0],  saida.shape[1], self.size)
        return saida


    def flatten(self, data):
        data.shape = (data.shape[0], data.shape[1]* data.shape[2])
        return data


    def classifica(self, data, target, test, classificador='KNN'):
        #print(target)
        if classificador == 'KNN':
            model = KNeighborsClassifier(3)
        elif  classificador == 'SVC':
            model = SVC(gamma=1, C=2)
        else:
            model = classificador
        model.fit(data, target) 
        resp = np.array(model.predict(test))
        return resp


    def KFoldOrganiza(self, target, donos, k):
        
        kf = StratifiedKFold(n_splits=k, shuffle=True)
        nomesUnicosDonos = np.unique(donos)
        targetDono = []
        for nome in nomesUnicosDonos:
            targetDono.append(target[donos==nome][0])
        resultado = []
        for train_donos_index, test_donos_index in kf.split(nomesUnicosDonos, targetDono):
            train_index = []
            for trainIndex in train_donos_index:
                train_index.extend(np.arange(len(donos))[donos == nomesUnicosDonos[trainIndex]])

            test_index = []
            for testIndex in test_donos_index:
                test_index.extend(np.arange(len(donos))[donos == nomesUnicosDonos[testIndex]])
            resultado.append((train_index, test_index))

        return resultado
    def classificaKFold(self, data, target, donos, classificador='KNN', k=5, n=3):

        accExec = []
        fscoreExec = []
        for i in range(n):
            accMedia = []
            fScoreMedia = []
            for train_index, test_index in self.KFoldOrganiza(target, donos, k):

                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = target[train_index], target[test_index]
                donos_test = donos[test_index]

                index = np.arange(len(X_train))
                np.random.shuffle(index)
                X_train = X_train[index]
                y_train = y_train[index]


                index = np.arange(len(X_test))
                np.random.shuffle(index)
                X_test = X_test[index]
                y_test = y_test[index]
                donos_test = donos_test[index]

                resultados = self.classifica(X_train, y_train, X_test, classificador)

                acc = self.acc(resultados, y_test, donos_test)
                accMedia.append(acc)
                fScore = self.fScore(resultados, y_test, donos_test)
                fScoreMedia.append(fScore)

            accExec.append(np.mean(accMedia))
            fscoreExec.append(np.mean(fScoreMedia))
            print('acc:', np.mean(accExec), 'fScore: ',np.mean(fscoreExec))
        return (round(np.mean(accExec), 4), round(np.mean(fscoreExec), 4))

    def acc(self, resultados, y_test, donos_test):
    
        newresp, rightResp = self.modeByDono(resultados, y_test, donos_test)
        acc = np.sum(np.ones(len(newresp))[newresp == rightResp])/len(newresp)
        
        return acc

    def fScore(self, resultados, y_test, donos_test):   
        newresp, rightResp = self.modeByDono(resultados, y_test, donos_test)    
        #print(rightResp, newresp) 

        conf = confusion_matrix(newresp, rightResp)
        print(pd.DataFrame(conf,columns = [0, 1]))
        fscore = f1_score(newresp, rightResp, average='macro')
        return fscore

    def modeByDono(self, resp, y_test, donos_test):
        newresp = []
        rightResp = []
        #print(len(newresp), len(rightResp), newresp)
        #resp = np.argmax(resultados, axis=1)
        for dono in np.unique(donos_test):
            #print(resp[donos_test==dono])
            #print( np.array(y_test)[donos_test==dono], resp[donos_test==dono])
            newresp.append(mode(resp[donos_test==dono])[0][0])
            rightResp.append(mode(np.array(y_test)[donos_test==dono])[0][0])
        newresp, rightResp = np.asarray(newresp), np.asarray(rightResp)
        return (newresp, rightResp)

def concatena(a, b):        
    x, y, z = a.shape[0], a.shape[1], a.shape[2]
    a = a.reshape(a.shape[0]*a.shape[1],a.shape[2] )
    b = b.reshape(b.shape[0]*b.shape[1],b.shape[2] )
    newVec = np.hstack((a,b))

    newVec = newVec.reshape(x, y, a.shape[1]+b.shape[1])
    return newVec