This lib creates an echo state model, and realizes a data transformation to a high dimensional space, with a temporal context together.

This is a Recurrent model, but differently from others, the ESN doesn't do classification, so U can use other methods to do this. 

In https://dl.acm.org/doi/pdf/10.1145/3412841.3441983?casa_token=Z0PInSxsm3QAAAAA:6lvkh7XywSk2P1JCyoef1ps-ETUD3nfcGGo2eS7ffbxG9J3F3DluuUwa-Fn7wd4EQHBgVEwrxEShbmU I did a study and show that the best methods are SVM Linear, SVM Gaussian, and KNN.


In this struct, it's also possible to choose the structure used to create the reservoir, which can be:

regular, 
smallWord, 
random


It's also possible to use a parameter to control de reservoir. The parameters are:

shape: Format of input data
network: Enumerate (random, smallword and regular) 
size: Size of network
W=[]: if created an echo state before, it's possible to pass the input matrix.
Win=[]: if created an echo state before, it's possible to pass the reservoir matrix.
mu: Mean of reservoir values
sigma: The reservoir use a normal distribution in values, so this is the sigma of distribution
nGrupos=5: 1/nGroups is the number of neighbors chosen for smallword and regular


Default values:

network = 'random'
size = 100
res='tudo'
W=[]
Win=[]
mu = 0
sigma = 0.08
nGrupos = 5
distancia = 1


How to use:

import reservoir as rv
model = rv.Reservoir(shape, size = 100, network='Regular')          
trainData = model.transform(trainData)



As the reservoir is a recurrent network, the data operate over a list, but it's possible to have more the one list, so the data is a 3D matrix.

data.shape = (x, y, z)
x is the number of lists, y represents the sequential objects, and z is the attributes of each object.


The output will be:
output = (x, y, k) where the k is the reservoir size.


