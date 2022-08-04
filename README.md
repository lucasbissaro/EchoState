This lib creates an echo state model, and realizes a data transformation to a high dimensional space, adding also the temporal context.

This is a Recurrent model, but differently from others, the ESN doesn't do classification, after the transformation you can use any other method.

In https://dl.acm.org/doi/pdf/10.1145/3412841.3441983?casa_token=Z0PInSxsm3QAAAAA:6lvkh7XywSk2P1JCyoef1ps-ETUD3nfcGGo2eS7ffbxG9J3F3DluuUwa-Fn7wd4EQHBgVEwrxEShbmU I did a study and show that the best methods are SVM Linear, SVM Gaussian, and KNN.



In this module, it's possible to choose the structure used to create the reservoir, which can be:

regular, 
smallWord, 
random


It's also possible to use the follow parameters to control the reservoir:

  shape: (i,j,k), Required
        Format of input data
  network: string, Default ='random'
        Enumerate (random, smallword and regular) 
  size: float, Default = 100
        Size of network
  W: array(float), Default = []
        if created an echo state before, it's possible to pass the input matrix.
  Win: array(float), Default = []
        if created an echo state before, it's possible to pass the reservoir matrix.
  mu: float, Default = 0
        Mean of reservoir values
  sigma: float, Default = 0.08
        The reservoir use a normal distribution in values, so this is the sigma of distribution
  nGrupos: float, Default = 5
        1/nGroups is the number of neighbors chosen for smallword and regular
  res: string, Default = 'Tudo', 
      Enumerate ('tudo', 'ultima'). The final result, can be the transformation of reservoir for all layers in the temporal context of the data inputed ('tudo'), or only one result for the all sequence, getting the last transformation ('ultima').
      
  distancia: int, Default = 1
       A midterm between 'tudo' and 'last', the return only have the result to every 'distancia' between layers on the temporal context.
  


How to use:

import reservoir as rv
model = rv.Reservoir(shape, size = 100, network='Regular')          
trainData = model.transform(trainData)



As the reservoir is a recurrent network, the data operate over a list, but it's possible to have more the one list, so the data is a 3D matrix.

data.shape = (x, y, z)
x is the number of lists, y represents the sequential objects, and z is the attributes of each object.


The output will be:
output = (x, y, k) where the k is the reservoir size.


