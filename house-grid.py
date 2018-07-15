import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.preprocessing import text
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.metrics import mean_absolute_error

######################################################################################
#Processing data

print("starting");

g = pandas.read_csv("data/kc-house-data.csv",
                    encoding = "ISO-8859-1")
g["price"] = g["price"]/1000


X = g[["sqft_above","sqft_basement","sqft_lot","sqft_living","floors",
       "bedrooms","yr_built","lat","long","bathrooms"]].values
Y = g["price"].values
zipcodes        = pandas.get_dummies(g["zipcode"]).values
condition       = pandas.get_dummies(g["condition"]).values
grade           = pandas.get_dummies(g["grade"]).values

X = np.concatenate((X,zipcodes),axis=1)
X = np.concatenate((X,condition),axis=1)
X = np.concatenate((X,grade),axis=1)

#######################################################################################
#Building deep network

def neural_model1(init,first_layer_N):
    model = Sequential()
    model.add(Dense(first_layer_N, input_dim=97,init=init, activation='relu'))
    model.add(Dense(5,activation="relu"))
    model.add(Dropout(0.05))
    model.add(Dense(5,activation="relu"))
    model.add(Dense(1))
    sgd = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mae',optimizer=sgd,metrics=["mae"])
    return model


#model.fit(X,Y,nb_epoch=300,verbose=2)

#######################################################################################
model = KerasRegressor(build_fn=neural_model1, nb_epoch = 20, verbose=2)

parameters = {'batch_size':[10,25],'init':['uniform','normal'],'first_layer_N':[10,50,64]}
grid = grid_search.GridSearchCV(model,parameters,n_jobs=-1,scoring="neg_mean_absolute_error")
grid_result = grid.fit(X, Y)


print("Best - : %f using %s" % (grid_result.best_score_, grid_result.best_params_))

for x in (grid_result.grid_scores_):
        print(str(x))
'''
Best - : -169.154745 using {'batch_size': 25, 'first_layer_N': 10, 'init': 'uniform'}
mean: -169.59983, std: 1.15714, params: {'batch_size': 10, 'first_layer_N': 10, 'init': 'uniform'}
mean: -285.36041, std: 167.77881, params: {'batch_size': 10, 'first_layer_N': 10, 'init': 'normal'}
mean: -406.97921, std: 168.34414, params: {'batch_size': 10, 'first_layer_N': 50, 'init': 'uniform'}
mean: -292.43490, std: 175.08486, params: {'batch_size': 10, 'first_layer_N': 50, 'init': 'normal'}
mean: -536.22566, std: 10.87691, params: {'batch_size': 10, 'first_layer_N': 64, 'init': 'uniform'}
mean: -286.98172, std: 175.16964, params: {'batch_size': 10, 'first_layer_N': 64, 'init': 'normal'}
mean: -169.15475, std: 0.69248, params: {'batch_size': 25, 'first_layer_N': 10, 'init': 'uniform'}
mean: -171.64070, std: 3.65385, params: {'batch_size': 25, 'first_layer_N': 10, 'init': 'normal'}
mean: -176.01319, std: 6.01938, params: {'batch_size': 25, 'first_layer_N': 50, 'init': 'uniform'}
mean: -180.99770, std: 13.98602, params: {'batch_size': 25, 'first_layer_N': 50, 'init': 'normal'}
mean: -172.03677, std: 4.94963, params: {'batch_size': 25, 'first_layer_N': 64, 'init': 'uniform'}
mean: -175.08868, std: 5.22316, params: {'batch_size': 25, 'first_layer_N': 64, 'init': 'normal'}
'''
