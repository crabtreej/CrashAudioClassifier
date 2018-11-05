from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle
from sklearn.cluster import KMeans
from sklearn.svm import SVC, LinearSVC
import numpy as np
from pyAudioAnalysis import audioFeatureExtraction as fe
from sklearn.metrics import classification_report
from scipy.stats import norm

def getHistogramsAndMembershipFromKMeans(kmeansCLF, classToClipsMFCCsMap):
    histograms = []
    classMembership = []
    for classID in classToClipsMFCCsMap:
        for clip in classToClipsMFCCsMap[classID]:
            predictedLabels = [0] * 64
            prediction = kmeans.predict(clip)
            for frame in prediction:
                predictedLabels[frame] += 1
            histograms.append(predictedLabels)
            classMembership.append(classID)

    return (histograms, classMembership)
    
def findBestParamForSVClassifier(trainingHistograms, trainingLabels, validationHistograms, validationLabels):

    c_value = '' 
    param_values = ''

    param_grid = [
        {'C': [0.001, 0.01, 1, 10, 100, 500, 1000, 100000, 1000000], 'kernel': ['linear']},
        {'C': [0.001, 0.01, 1, 10, 100, 500, 1000, 100000, 1000000], 'gamma': [3, 5, 20, 100, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
    ]

    clf = GridSearchCV(SVC(max_iter=10000), param_grid, cv=3, scoring='accuracy')

    clf.fit(trainingHistograms, trainingLabels)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")

    y_true, y_pred = validationLabels, clf.predict(validationHistograms)
    print(classification_report(y_true, y_pred))
    print()



    return clf.best_params_

def findBestParamForLinearSVClassifier(trainingHistograms, trainingLabels, validationHistograms, validationLabels):

    c_value = '' 
    #can change this to take input from user if you want to test values yourself by setting c_value = the line below.
    #input('Input a list of C-values (one by one) to test, or blank to see a random sampling from around what I\'ve found to be the best values.\n')

    if c_value == '':
        #i think 0.01 is best, so sample around 0.01 randomly with some values
        c_values = [0.01, 0.011, 0.012, 0.0115, 0.0125, 0.018, 0.0099, 0.009, 0.0095, 0.0085, 0.0080, 0.0075, 0.007, 0.0065, 0.006, 0.005, 0.0055, 0.0051, 0.0049, 0.0045, 0.004, 0.0119, 0.0093] 
    else:
        c_values = [float(c_value)]
        while(c_value != ''):
            c_value = input()
            if c_value != '':
                c_values.append(float(c_value))
    clf = GridSearchCV(SVC(max_iter=10000), {'C': c_values}, cv=3, scoring='accuracy')

    clf.fit(trainingHistograms, trainingLabels)

    #print("Best parameters set found on development set:")
    #print()
    #print(clf.best_params_)
    #print()
    #print("Grid scores on development set:")
    #print()
    #means = clf.cv_results_['mean_test_score']
    #stds = clf.cv_results_['std_test_score']
    #for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #    print("%0.3f (+/-%0.03f) for %r"
    #          % (mean, std * 2, params))
    #print()

    #print("Detailed classification report:")
    #print()
    #print("The model is trained on the full development set.")
    #print("The scores are computed on the full evaluation set.")

    y_true, y_pred = validationLabels, clf.predict(validationHistograms)
    #print(classification_report(y_true, y_pred))
    #print()

    return clf.best_params_['C']

if __name__ == '__main__':

    with open("classToClipsAsFramesOfMFCCsMapA.txt", "rb") as fp:
        classToClipsAsFramesOfMFCCsMapA = pickle.load(fp)
    with open("classToClipsAsFramesOfMFCCsMapB.txt", "rb") as fp:
        classToClipsAsFramesOfMFCCsMapB = pickle.load(fp)
    with open("classToClipsAsFramesOfMFCCsMapC.txt", "rb") as fp:
        classToClipsAsFramesOfMFCCsMapC = pickle.load(fp)
    
    MFCCsMapList = []
    MFCCsMapList.append(classToClipsAsFramesOfMFCCsMapA)
    MFCCsMapList.append(classToClipsAsFramesOfMFCCsMapB)

    comboList = []
    for MFCCsMap in MFCCsMapList:
        for classID in MFCCsMap:
            #there are a certain amount of clips keyed to each class
            for clip in MFCCsMap[classID]:
                #a clip is an array of frames
                for frame in clip:
                    #a frame is an array of 13 mfccs
                    comboList.append(frame)

    # KMeans clustering of combo list
    kmeans = KMeans(n_clusters=64).fit(comboList)

    # KMeans predict on each clip now, get predicted label
    trainingHistograms, trainingLabels = getHistogramsAndMembershipFromKMeans(kmeans, classToClipsAsFramesOfMFCCsMapA)
    tempHist, tempLabels = (getHistogramsAndMembershipFromKMeans(kmeans, classToClipsAsFramesOfMFCCsMapB))
    trainingHistograms.extend(tempHist)
    trainingLabels.extend(tempLabels)

    validationHistograms, validationLabels = getHistogramsAndMembershipFromKMeans(kmeans, classToClipsAsFramesOfMFCCsMapC)

    #now we have training data for an svm
   
    #determined dual=True is twice as fast

    #If you run this code, you'll find that the best parameter for a LinearSVC for this data is
    #0.00999999... = 0.01. I tried values much smaller that that, and up to 10^7 for the C parameter,
    #and 0.01 is the best. This just averages the values found over a large number of iterations in an
    #attempt to remove any variance from it. 
   
    bestC = 0
    bestGamma = 0
    countRBF = 0
    countLinear = 0
    for i in range (0, 20):
        tempDict = findBestParamForSVClassifier(trainingHistograms, trainingLabels, validationHistograms, validationLabels)
        bestC += tempDict['C']
        bestGamma += tempDict['gamma']
        if(tempDict['kernel'] == 'rbf'):
            countRBF += 1
        else:
            countLinear += 1

    bestC = bestC / 20.0
    bestGamma = bestGamma / 20.0
    bestKernel = 'rbf'
    if countRBF < countLinear:
        bestKernel = 'linear'

    print(f'Best C from averaging was: {bestC}')
    print(f'Best Gamma was: {bestGamma}')
    print(f'Best Kernel was: {bestKernel}')


