from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.cluster import KMeans
from sklearn.svm import SVC #, LinearSVC
import numpy as np
#from pyAudioAnalysis import audioFeatureExtraction as fe
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def getHistogramsAndMembershipFromKMeans(kmeansCLF, classToClipsMFCCsMap, numMeans):
    histograms = []
    classMembership = []
    for classID in classToClipsMFCCsMap:
        for clip in classToClipsMFCCsMap[classID]:
            predictedLabels = [0] * numMeans 
            prediction = kmeans.predict(clip)
            for frame in prediction:
                predictedLabels[frame] += 1
            histograms.append(predictedLabels)
            classMembership.append(classID)

    return (histograms, classMembership)
    
def findBestParamForSVClassifier(trainingHistograms, trainingLabels, validationHistograms, validationLabels):

    c_value = '' 
    param_values = ''

    c_values = [0.5, 0.1, 1.0, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5,5.5,6,6.5,7,7.5,8]
    gamma_values = [0.0001, 0.0003, 0.0005, .0007,.0008, .00085, .0009, .00095, .001, .0011, .0012, .0015, .002]
 
    param_grid = [
        {'C': c_values, 'gamma': gamma_values, 'kernel': ['rbf']},
    ]

    clf = GridSearchCV(SVC(max_iter=10000), param_grid, cv=10, scoring='accuracy')
    clf.fit(trainingHistograms, trainingLabels)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    #print("Grid scores on development set:")
    #print()
    #means = clf.cv_results_['mean_test_score']
    #stds = clf.cv_results_['std_test_score']
    #for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #    print("%0.3f (+/-%0.03f) for %r"
    #          % (mean, std * 2, params))
    #print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")

    y_true, y_pred = validationLabels, clf.predict(validationHistograms)
    print(classification_report(y_true, y_pred))
    print()

    recognition_rate = 0.0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            recognition_rate += 1.0

    return (clf.best_params_, recognition_rate / len(y_true))

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
    clf = GridSearchCV(SVC(max_iter=100), {'C': c_values}, cv=3, scoring='accuracy')

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

    suffixes = ["50m", "100m", "300m"]
   
    figNum = 0 
    clustersToBestCsandGammas = { 64 : (1, .002), 128 : (2, 0.0012), 256 : (3.5, 0.0007), 512 : (2, 0.0009), 1024 : (4.5, 0.0005) }
    for lengthSuffix in suffixes:

    #start 
        with open("classToClipsAsFramesOfMFCCsMapA_" + lengthSuffix + ".txt", "rb") as fp:
            classToClipsAsFramesOfMFCCsMapA = pickle.load(fp)
            for classID in classToClipsAsFramesOfMFCCsMapA:
                classToClipsAsFramesOfMFCCsMapA[classID] = classToClipsAsFramesOfMFCCsMapA[classID][:len(classToClipsAsFramesOfMFCCsMapA[classID])//2]
        with open("classToClipsAsFramesOfMFCCsMapB_" + lengthSuffix + ".txt", "rb") as fp:
            classToClipsAsFramesOfMFCCsMapB = pickle.load(fp)
            for classID in classToClipsAsFramesOfMFCCsMapB:
                classToClipsAsFramesOfMFCCsMapB[classID] = classToClipsAsFramesOfMFCCsMapB[classID][:len(classToClipsAsFramesOfMFCCsMapB[classID])//2]
        with open("classToClipsAsFramesOfMFCCsMapC_" + lengthSuffix + ".txt", "rb") as fp:
            classToClipsAsFramesOfMFCCsMapC = pickle.load(fp)
            for classID in classToClipsAsFramesOfMFCCsMapC:
                classToClipsAsFramesOfMFCCsMapC[classID] = classToClipsAsFramesOfMFCCsMapC[classID][:len(classToClipsAsFramesOfMFCCsMapC[classID])//2]
        with open("classToClipsAsFramesOfMFCCsMapD_" + lengthSuffix + ".txt", "rb") as fp:
            classToClipsAsFramesOfMFCCsMapD = pickle.load(fp)
        
        MFCCsMapList = []
        MFCCsMapList.append(classToClipsAsFramesOfMFCCsMapA)
        MFCCsMapList.append(classToClipsAsFramesOfMFCCsMapB)
        MFCCsMapList.append(classToClipsAsFramesOfMFCCsMapC)

        comboList = []
        for MFCCsMap in MFCCsMapList:
            for classID in MFCCsMap:
                #there are a certain amount of clips keyed to each class
                for clip in MFCCsMap[classID]:
                    #a clip is an array of frames
                    for frame in clip:
                        #a frame is an array of 13 mfccs
                        comboList.append(frame)

        kmeansSizes = [64, 128, 256, 512, 1024]
        recRatesForKMeans = []
        bestCForKMeans = []
        bestGammaForKMeans = []

        for ksize in kmeansSizes:
            # KMeans clustering of combo list
            kmeans = KMeans(n_clusters=ksize).fit(comboList)

            # KMeans predict on each clip now, get predicted label
            trainingHistograms, trainingLabels = getHistogramsAndMembershipFromKMeans(kmeans, classToClipsAsFramesOfMFCCsMapA, ksize)
            tempHist, tempLabels = getHistogramsAndMembershipFromKMeans(kmeans, classToClipsAsFramesOfMFCCsMapB, ksize)
            trainingHistograms.extend(tempHist)
            trainingLabels.extend(tempLabels)
            tempHist, tempLabels = getHistogramsAndMembershipFromKMeans(kmeans, classToClipsAsFramesOfMFCCsMapC, ksize)
            trainingHistograms.extend(tempHist)
            trainingLabels.extend(tempLabels)

            testingHistograms, testingLabels = getHistogramsAndMembershipFromKMeans(kmeans, classToClipsAsFramesOfMFCCsMapD, ksize)

            #now we have training data for an svm
        
            #determined dual=True is twice as fast

            #If you run this code, you'll find that the best parameter for a LinearSVC for this data is
            #0.00999999... = 0.01. I tried values much smaller that that, and up to 10^7 for the C parameter,
            #and 0.01 is the best. This just averages the values found over a large number of iterations in an
            #attempt to remove any variance from it. 
        
            #tempDict, recRate = findBestParamForSVClassifier(trainingHistograms, trainingLabels, validationHistograms, validationLabels)
            bestC, bestGamma = clustersToBestCsandGammas[ksize]
            crashSVC = SVC(C=bestC, gamma=bestGamma, kernel='rbf', max_iter=10000)
            crashSVC.fit(trainingHistograms, trainingLabels)
            
            testPredictions = crashSVC.predict(testingHistograms)
            print(f'Printing classification report for cluster {ksize} and frame size {lengthSuffix}:')
            print()
            print(classification_report(testingLabels, testPredictions))
            print()

            recognition_rate = 0.0
            for true, pred in zip(testingLabels, testPredictions):
                if true == pred:
                    recognition_rate += 1.0

            recRatesForKMeans.append(recognition_rate / len(testingLabels))

        plt.figure(figNum)
        tickmarks = np.arange(1, len(kmeansSizes) + 1)
        fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(20,10))
        fig.suptitle(lengthSuffix + 's Frame Length', fontsize=16)
        ax1.plot(tickmarks, recRatesForKMeans)
        ax1.set_xticks(tickmarks)
        ax1.set_xticklabels(kmeansSizes)
        ax1.set_yticks(recRatesForKMeans)
        ax1.set_yticklabels(recRatesForKMeans)
        ax1.set_title('Rec. Rate vs. KMeans Clusters')
        ax1.set_xlabel('# of KMeans Clusters')
        ax1.set_ylabel('Rec. Rate')

        plt.subplots_adjust(hspace=0.5)
        figNum += 1

    plt.show()
    #end
    
