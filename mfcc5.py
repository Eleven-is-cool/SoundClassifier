import librosa
import keras
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM,Dense,Activation,SimpleRNN,Conv1D,MaxPool1D,Flatten,Reshape,Dropout
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy
from keras.optimizers import RMSprop
from keras.layers import TimeDistributed, Bidirectional
import copy
import matplotlib.pyplot as plt

dogpath =  "/Users/birenjianmo/Desktop/learn/keras/dog"
catpath =  "/Users/birenjianmo/Desktop/learn/keras/cat"

test_dogpath =  "/Users/birenjianmo/Desktop/learn/keras/test/dog"
test_catpath =  "/Users/birenjianmo/Desktop/learn/keras/test/cat"

maxlength = 200
train_x = []
train_y = []

labelDict = {
    "dog":{
        "featureArr":[0,1],
        "frepArea":(450,1800)
    },
    "cat":{
        "featureArr":[1,0],
         "frepArea":(760,1500)
    }
}

def appendFeatureData( filepath, label ):
    y, sr = librosa.load( filepath )
    # 抽取13阶mfcc特征，.T是将shape为(13,None)转置为(None,13)
    split_y = librosa.effects.split( y,top_db=10 )
    for s_y in split_y:
        mfcc_feat = librosa.feature.melspectrogram(y=y[s_y[0]:s_y[1]],n_mels=26,fmin=labelDict[label]["frepArea"][0],fmax=labelDict[label]["frepArea"][1]).T
        if mfcc_feat.shape[0] < maxlength:
            mfcc_feat = np.concatenate((mfcc_feat,np.zeros((maxlength-mfcc_feat.shape[0],26))))
        mfcc_feat = mfcc_feat[:maxlength,:]
        train_x.append( mfcc_feat )
        train_y.append( labelDict[label]["featureArr"] )

def formatToNumpyArray():
    global train_x, train_y
    train_x = np.asarray( train_x )
    train_y = np.asarray( train_y )


def initData( path, label, empty=False, **kw ):
    global train_x, train_y

    for root, dirs, files in os.walk( path ):
        for file in files:
            if os.path.splitext( file )[-1] in [".mp3",".wav"]:
                if empty:
                    train_x = []
                    train_y = []

                filepath = os.path.join( root, file ) 
                appendFeatureData( filepath, label )
                try:
                    kw["func"](kw["arg"], label, filepath)
                except:
                    pass

def _test(model,*args):
    formatToNumpyArray()
    r = model.evaluate( train_x,train_y )
    print( "acc:\t%s\t%s\t%s" % (r[-1], args[0], args[-1]) )

def test( path, label, model ):
    initData(path, label, empty=True, func=_test, arg=model)
    print( path )



def convModel():
    model = Sequential()
    model.add(Conv1D(32,4,input_shape=(( maxlength,26))))
    model.add(MaxPool1D(4))
    model.add(Conv1D(64,8))
    model.add(MaxPool1D(8))
    # model.add(Conv1D(64,16))
    # model.add(MaxPool1D(16))
    model.add(Flatten())
    model.add(Dense(32,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss="categorical_crossentropy",optimizer=RMSprop(lr=0.001),metrics=[categorical_accuracy])
    return model


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('categorical_accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_categorical_accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('categorical_accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_categorical_accuracy'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

def main():
    initData( dogpath, "dog" )
    initData( catpath, "cat" )
    formatToNumpyArray()
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3)
    model = convModel()

    history = LossHistory()
    early_stopping = EarlyStopping(monitor='val_loss',patience=5)
    model.fit(X_train, y_train,callbacks=[history,early_stopping], validation_data=(X_test,y_test), epochs=55, batch_size=64)

    # 测试dog

    print( "#"*20," dog " ,"#"*20 )
    test( test_dogpath, "dog", model )
    print( "#"*20," cat " ,"#"*20 )
    test( test_catpath, "cat", model )

    model.save('mfcc5_model.h5')
    history.loss_plot('epoch')

if __name__ == '__main__':
    main()
    