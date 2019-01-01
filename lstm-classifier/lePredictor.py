#Load Packages
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
#Generate 2 sets of X variables
#LSTMs have unique 3-dimensional input requirements
seq_length=5
X =[[i+j for j in range(seq_length)] for i in range(100)]
X_simple =[[i for i in range(4,104)]]
X =np.array(X)
X_simple=np.array(X_simple)

y =[[ i+(i-1)*.5+(i-2)*.2+(i-3)*.1 for i in range(4,104)]]
y =np.array(y)
X_simple=X_simple.reshape((100,1))
X=X.reshape((100,5,1))
y=y.reshape((100,1))

model = Sequential()

model.add(LSTM(8,input_shape=(5,1),return_sequences=False))#True = many to many
model.add(Dense(2,kernel_initializer='normal',activation='linear'))
model.add(Dense(1,kernel_initializer='normal',activation='linear'))
model.compile(loss='mse',optimizer ='adam',metrics=['accuracy'])
model.fit(X,y,epochs=2000,batch_size=5,validation_split=0.05,verbose=0);

scores = model.evaluate(X,y,verbose=1,batch_size=5)
print('Accurracy: {}'.format(scores[1]))

import matplotlib.pyplot as plt
predict=model.predict(X)
plt.plot(y, predict-y, 'C2')
plt.ylim(ymax = 3, ymin = -3)
plt.show()