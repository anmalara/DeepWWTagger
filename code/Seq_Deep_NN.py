# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import matplotlib.pyplot as plt
#import csv
import numpy
#from keras.models import model_from_json
from sklearn.metrics import roc_curve, auc
import numpy
import root_numpy
import pandas

def jetID(jetEta,jetPhf,jetNhf,ncandidates,jetMuf,jetChf,jetChm,jetElf):

    if(abs(jetEta)<=2.4 and jetPhf<0.90 and jetNhf<0.90 and ncandidates>1 and jetMuf<0.80 and jetChf>0 and jetChm>0 and jetElf<0.8):
        return True
    elif(abs(jetEta)>2.4 and abs(jetEta)<2.7 and jetPhf<0.90 and jetNhf<0.90 and ncandidates>1 and jetMuf<0.80):
        return True
    else:
        return False

def FillVectors(filename,mode):

    treename="boostedAK8/events"

    JetInfos=[]
    JetInfosArray=[]
    njet=0

    data = root_numpy.root2array(filename,treename=treename,branches=["jetPt","jetEta","jetChf","jetMuf","jetElf","jetNhf","jetChm","jetPhf","jetTau1","jetTau2","jetTau3","jetMassSoftDrop","ncandidates"])
    data = pandas.DataFrame(data)

    print data.shape

    if(mode=='train'):
        start=0
        end=data.shape[0]/2

    elif(mode=='test'):
        start=data.shape[0]/2
        end=data.shape[0]

    for i in range(start,end):

        if(data.iloc[i]['jetPt']>500.): #pt cut

            if(jetID(data.iloc[i]['jetEta'],data.iloc[i]['jetPhf'],data.iloc[i]['jetNhf'],data.iloc[i]['ncandidates'],data.iloc[i]['jetMuf'],data.iloc[i]['jetChf'],data.iloc[i]['jetChm'],data.iloc[i]['jetElf'])):

                njet+=1
                JetElements=[data.iloc[i]['jetTau1'],data.iloc[i]['jetTau2'],data.iloc[i]['jetTau3'],data.iloc[i]['jetMassSoftDrop']]
                print data.iloc[i]['jetMassSoftDrop']
                JetInfos.append(JetElements)

    JetInfosArray=numpy.asarray(JetInfos)
    JetInfosArray=JetInfosArray.astype(numpy.float)

    JetInfosArray=JetInfosArray.reshape(njet,4)

    return (JetInfosArray, njet)

filename="flatTreeFileHiggs-nano.root"

(HiggsInfosArray,njetHiggs) = FillVectors(filename,'train')

filename="flatTreeFileQCD-nano.root"

(QCDInfosArray,njetQCD) = FillVectors(filename,'train')

JetInfosArray = numpy.concatenate((HiggsInfosArray,QCDInfosArray))

LabelsSignal_array=numpy.ones(njetHiggs)
LabelsBackground_array=numpy.zeros(njetQCD)

Labels_array=numpy.concatenate((LabelsSignal_array,LabelsBackground_array))

#print the features
for j in range(0,JetInfosArray.shape[1]):
    jetInfosQCD=[]
    jetInfosHiggs=[]

    for i in range(0,len(Labels_array)):
        if Labels_array[i]==0.0:
            jetInfosQCD.append(JetInfosArray[i,j])
        elif Labels_array[i]==1.0:
            jetInfosHiggs.append(JetInfosArray[i,j])

    f = plt.figure(j)
    plt.hist(jetInfosQCD,label='QCD',histtype="step",normed=True)
    plt.hist(jetInfosHiggs,label='Higgs',histtype="step",normed=True)
    plt.xlabel('Features')
    plt.ylabel('Entries')
    plt.legend(loc='upper right')
    plt.title('First feature test')
    if j<3:
         plt.axis([0, 1, 0.0001, 99])
    else:
         plt.axis([0, 300, 0.0001, 99])
    plt.grid(True)

    plt.yscale('log')
    f.show()
    f.savefig('feat'+str(j)+'.png')

for f in range(JetInfosArray.shape[1]):
    mean = numpy.mean(JetInfosArray[:,f])
    std = numpy.std(JetInfosArray[:,f])
    JetInfosArray[:,f] = (JetInfosArray[:,f] - mean)/std

# fix random seed for reproducibility
numpy.random.seed(7)

# let's try a deep neural network with the built inputs
model = Sequential()

#keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)   #batch normalization, normally is not tuned further

model.add(Dense(4, input_dim=4, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
model.add(Dense(10, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print 'Training on '+str(njetHiggs)+' signal events and '+str(njetQCD)+' background events'

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# Fit the model (this is the training!)
history = model.fit(JetInfosArray, Labels_array, epochs=100, batch_size=256)

#Plot machinery based on matplotlib
from keras.utils import plot_model
plot_model(model, to_file='model.svg')

#printing the model summary
print(model.summary())

#Testing samples here
filename="flatTreeFileHiggs-nano.root"

(HiggsTestInfosArray,njetHiggsTest) = FillVectors(filename,'test')

filename="flatTreeFileQCD-nano.root"

(QCDTestInfosArray,njetQCDTest) = FillVectors(filename,'test')

JetTestInfosArray = numpy.concatenate((HiggsTestInfosArray,QCDTestInfosArray))

for f in range(JetTestInfosArray.shape[1]):
    mean = numpy.mean(JetTestInfosArray[:,f])
    std = numpy.std(JetTestInfosArray[:,f])
    JetTestInfosArray[:,f] = (JetTestInfosArray[:,f] - mean)/std

LabelsSignal_array=numpy.ones(njetHiggsTest)
LabelsBackground_array=numpy.zeros(njetQCDTest)

LabelsTest_array=numpy.concatenate((LabelsSignal_array,LabelsBackground_array))

print 'Testing on '+str(njetHiggsTest)+' signal events and '+str(njetQCDTest)+' background events'

predictions = model.predict(JetTestInfosArray)
predictions_array = numpy.asarray(predictions)
predictions_array = predictions_array.astype(numpy.float)

print predictions

scores_train = model.evaluate(JetInfosArray,Labels_array,verbose=True)
print("\n TRAIN \n%s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))

scores_test = model.evaluate(JetTestInfosArray,LabelsTest_array,verbose=True)
print("\n TEST \n%s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))

#plot the loss as a function of the epoch
# summarize history for accuracy
f = plt.figure()
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
f.show()
f.savefig('AccuracyVsEpoch.png')

# summarize history for loss
f = plt.figure()
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
f.show()
f.savefig('LossVsEpoch.png')

#ROC curve
f = plt.figure()
fpr, tpr, _ = roc_curve(LabelsTest_array, predictions)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
print('AUC: %f' % roc_auc)
plt.show()
f.show()
f.savefig('ROC.png')
