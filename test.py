import reservoir as rv


shape = (1,2,2)

trainData = [[[2,2],[3,1]]]

model = rv.Reservoir(shape, size = 100, network='Regular')  
        
trainData = model.transform(trainData)
print(trainData)