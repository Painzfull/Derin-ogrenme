import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import re  

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam




import os
os.chdir("C:\\Users\\metec\\OneDrive\\Masaüstü\\Udemy_DerinOgrenmeyeGiris-master\\TitanicGorsellestirmeveSiniflama")

def preprocess(data):
    data.Cabin.fillna('0', inplace =True)
    data.loc[data.Cabin.str[0] == 'A' , 'Cabin'] = 1
    data.loc[data.Cabin.str[0] == 'B' , 'Cabin'] = 2
    data.loc[data.Cabin.str[0] == 'C' , 'Cabin'] = 3
    data.loc[data.Cabin.str[0] == 'D' , 'Cabin'] = 4
    data.loc[data.Cabin.str[0] == 'E' , 'Cabin'] = 5
    data.loc[data.Cabin.str[0] == 'F' , 'Cabin'] = 6
    data.loc[data.Cabin.str[0] == 'G' , 'Cabin'] = 7
    data.loc[data.Cabin.str[0] == 'T' , 'Cabin'] = 8
        
    data['Sex'].replace('female' , 1, inplace =True)
    data['Sex'].replace('male' , 2, inplace =True)
        
    data['Embarked'].replace('S', 1, inplace=True)
    data['Embarked'].replace('C', 2, inplace=True)
    data['Embarked'].replace('Q', 3, inplace=True)
        
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].median(), inplace=True)
        
    return data 
        

def group_titles(data):
    data['Names'] = data['Name'].map(lambda x: len(re.split(' ', x)))
    data['Title'] = data['Name'].map(lambda x: re.search(', (.+?) ',x).group(1))
    data['Title'].replace('Master.', 0, inplace=True)
    data['Title'].replace('Mr.', 1, inplace=True)
    data['Title'].replace(['Ms.','Mlle.','Miss.'], 2, inplace=True)
    data['Title'].replace(['Mme.','Mrs.'], 3, inplace=True)
    data['Title'].replace(['Donna.', 'Lady.', 'the Countess.', 'Capt.', 'Col', 'Don.', 'Major.', 'Rev.','Sir.', 'Jonkheer.','the'], 4, inplace=True)


def data_subset(data):
    features =['Pclass', 'SibSp', 'Parch','Sex', 'Names', 'Title', 'Age', 'Cabin' ]
    lengh_features = len(features)
    subset=data[features]#.fillna(0)
    return subset, lengh_features 
    
   
#modelin tasarlandığı fonksiyon
def create_model(train_set_size , input_lengh , num_epochs , batch_size):
    model = Sequential()
    model.add(Dense(7, input_dim=input_lengh, activation = 'softplus'))
    model.add(Dense(3,activation='softplus'))
    model.add(Dense(1,activation='softplus'))
    
    lr = .001
    adam0 = Adam(lr = lr)
    
    model.compile(loss ='binary_crossentropy', optimizer = adam0, metrics=['accuracy'])
    filepath ='weights.best.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list =[checkpoint]
    
    history_model = model.fit(X_train[:train_set_size], Y_train[:train_set_size], callbacks=callbacks_list, epochs= num_epochs, batch_size=batch_size, verbose=0)
    return model, history_model

def plots(history):
    loss_history = history.history['loss']
    acc_history = history.history['accuary']
    epochs = [(i + 1) for i in range (num_epochs)]
    
    ax = plt.subplot(211)
    ax.plot(epochs, loss_history, color = 'red')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error Rate\n')
    ax.set_title('Error Rate per Epoch\n')
    
    ax2 = plt.subplot(212)
    ax2.plot(epochs, acc_history, color = 'blue')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy\n')
    ax2.set_title('Accuracy per Epoch\n')
    
    plt.subplots_adjust(hspace=0.8)
    plt.savefig('Accuracy_loss.png')
    plt.close()

def test(batch_size):
    test= pd.read_csv('test.csv',header=0)
    test_ids =test['PassenderId']
    test = preprocess(test)
    group_titles(test)
    testdata, _ = data_subset(test)
    
    X_test = np.array(testdata).astype(float)
    
    output=model.predict (X_test,batch_size=batch_size,verbose = 0)
    output =output.reshape((418,))
    
    colum_1 = np.concatenate((['PassengerId'], test_ids), axis = 0)
    colum_2 = np.concatenate((['Survived'], output), axis = 0)
    
    f = open("output.csv", "w")
    writer = csv.writer(f)
    for i in range(len(colum_1)):
        writer.writerow( [colum_1[i]] + [colum_2[i]])
        f.close()


seed  = 7
np.random.seed(seed)

train = pd.read_csv('train.csv', header=0)

preprocess(train)
group_titles(train)

num_epochs = 100
batch_size = 32

traindata, lengh_features = data_subset(train)

Y_train = np.array(train['Survived']).astype(int)
X_train = np.array(traindata).astype(float)

train_set_size = int(.67 * len(X_train))

model, history_model = create_model(train_set_size , lengh_features, num_epochs, batch_size)

plots(history_model)

X_validation = X_train[train_set_size:]
Y_validation = Y_train[train_set_size:]
    
loss_and_metrics = model.evaluate(X_validation, Y_validation, batch_size=batch_size) 
print("loss_and_metrics")

test(batch_size)
