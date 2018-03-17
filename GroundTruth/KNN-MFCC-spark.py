# coding: utf-8

# In[1]:


import pickle
import os.path
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh import qparser
import random
#from nltk.corpus import stopwords
#import nltk
# import pyttsx3
# import speech_recognition as sr
import glob
# import pydub
# import asr.align
# from gtts import gTTS
import os
import soundfile as sf
from python_speech_features import mfcc
from python_speech_features import fbank
# nltk.download('stopwords')
# import lshknn
import numpy as np
import fnmatch
import json
from sklearn.neighbors import NearestNeighbors
from pyspark import SparkContext, SparkConf
import time


# In[2]:


homeDir = '/users/Fangyi'
audioFileLocation = homeDir + '/DevData-AudioBooks/dev-clean/'
queryAudioLocation = homeDir + '/spokenkeywordsearch/GroundTruth/'
encodingLocation = homeDir + '/model_encoding/'


# In[18]:


class indexer():
    def __init__(self):
        self.transDict = pickle.load(open('transDict2', 'r'))
        self.convertedDict = pickle.load(open('speechRecConvertDict2', 'r'))
        self.queryList = pickle.load(open('queryList2', 'r'))
        self.queryConvertedDict = pickle.load(open('queryConvertedDict2', 'r'))
        schema = Schema(audio=TEXT(stored=True), content=TEXT)
        if not os.path.exists("index"):
            os.mkdir("index")
        self.index = create_in("index", schema)
        writer = self.index.writer()
        for a in self.transDict:
            writer.add_document(audio=u""+a, content=u""+self.transDict[a])
        writer.commit()
        self.searcher = self.index.searcher()

    def getMaxSearch(self):
        maxFoundList = [0]*len(self.queryList)
        diffCount = 0
        for i in range(len(self.queryList)):
            if(self.queryList[i] and self.queryConvertedDict[str(i)]):
                query1 = QueryParser("content", self.index.schema, group=qparser.OrGroup).parse(self.queryList[i])
                result1 = self.searcher.search(query1, limit=None)
                list1 = []
                for res in result1:
                    list1.append(res['audio'])
                maxFoundList[i] = len(list1)
        return maxFoundList
    
    def compareResults(self, indices, distances, audioPosnDict, queryPosnDict, logfileName, limitk, dontPrint=False):
        log = open(logfileName, 'a')
        totalQueryCount = 0
        totalTrueResults = 0
        totalMatchingResults = 0
        maxMatchingQuery = ""
        maxMatching = 0
        maxFoundList = self.getMaxSearch()
        precisionList = []
        recallList = []
        i = 0
        totalFrames = len(distances)
        while(i<totalFrames):
            samePos = []
            while(i+1<totalFrames and queryPosnDict[i]==queryPosnDict[i+1]):
                samePos.append(i)
                i += 1
            if(len(samePos)==0):
                samePos = [i]
            totalQueryCount += 1
            log.write("------------------------------------------------")
            query = self.queryList[int(queryPosnDict[i])]
            log.write("Results for query: " + query +"\n")
            query = QueryParser("content", self.index.schema, group=qparser.OrGroup).parse(query)
            results = self.searcher.search(query, limit=limitk)
            list1 = []
            list2 = []
            for res in results:
                list1.append(res['audio'])
            distanceArr = np.array(distances[samePos[0]:samePos[len(samePos)-1]+1]).flatten()
            indiceArr = np.array(indices[samePos[0]:samePos[len(samePos)-1]+1]).flatten()
            distanceArr = (-distanceArr).argsort()[:min(limitk, len(distanceArr))]
            for d in distanceArr:
                list2.append(audioPosnDict[indiceArr[d]])
            comm = len(set(list1).intersection(set(list2)))
            foundCount = len(list2)
            origCount = maxFoundList[int(queryPosnDict[i])]
            if(foundCount>0):
                precisionList.append(float(comm)/float(foundCount))
            else:
                precisionList.append(0.0)
            if(origCount>0):
                recall = float(comm)/float(min(limitk, origCount))
                if(recall == 1.0 or recall == 1):
                    print("Query with recall 1.0: ", query)
                recallList.append(float(comm)/float(min(limitk, origCount)))
            else:
                recallList.append(0)
            totalTrueResults += len(list1)
            if(comm>maxMatching):
                maxMatching = comm
                maxMatchingQuery = query
            totalMatchingResults += comm
            log.write("Matching : {0}".format(comm))
            log.flush()
            i += 1
        if(not dontPrint):
            print("Total queries: ", totalQueryCount)
            print("Total true finds: ", totalTrueResults)
            print("Total matching results: ", totalMatchingResults)
        return precisionList, recallList

# In[19]:


class knn_mfcc():
    def __init__(self, featureType, stride, windowSize):
        self.featureType = featureType
        self.stride = stride
        self.windowSize = windowSize
        self.knnnbrs = []
        
    def initializeFeatures(self, featLocation, kind):
        features = []
        posnDict = dict()
        totalFileCount = 0
        frameCount = 0
            
        if(self.featureType == "mfcc"):
            for root, dirnames, filenames in os.walk(featLocation):
                for filename in fnmatch.filter(filenames, '*.wav'):
                    totalFileCount += 1
                    with open(os.path.join(root, filename), 'rb') as f:
                        data, samplerate = sf.read(f)
                    if(kind=="audio"):
                        mfcc_feat = np.array(mfcc(data, samplerate))
                    else:
                        mfcc_feat = np.array(mfcc(data, samplerate, nfft=600))
                    mfcclen = len(mfcc_feat)
                    i = 0
                    lastLen = 0
                    while(i+self.windowSize<mfcclen):
                        features.append(mfcc_feat[i:i+self.windowSize].flatten())
                        lastLen = len(mfcc_feat[i:i+self.windowSize].flatten())
                        posnDict[frameCount] = filename.split('.')[0]
                        i += self.stride
                        frameCount += 1
#                     if(i<mfcclen):
#                         features.append(mfcc_feat[i:mfcclen].flatten()  [0]*(lastLen-(mfcclen-i)))
                        
        elif(self.featureType == "encoding"):
            if(kind=="audio"):
                fileType = '*.flac.json'
            else:
                fileType = '*.wav.json'
            for root, dirnames, filenames in os.walk(featLocation):
                for filename in fnmatch.filter(filenames, fileType):
                    totalFileCount += 1
                    curr = np.array(json.load(open(root+filename, 'r')))
                    for c in curr:
                        features.append(c)
                        posnDict[frameCount] = filename.split('.')[0]
                        frameCount += 1
        return features, posnDict
    
    def find(self, x):
        assert type(x) == type(np.zeros((1,))), x
        x = np.reshape(x, (-1, self.n_feature)) 
        distances, indices = self.bc_knn.value.kneighbors(x)
        return [distances.tolist(), indices.tolist()]
        
    def merge(self, x, y):
        return [x[0] + y[0], x[1] + y[1]]

    def initKNN(self, k, knnalgorithm, audioFeatures, metric=None):
        train_data = np.array(audioFeatures)
        self.n_feature = train_data.shape[1]
        if(metric is not None):
            self.knn = NearestNeighbors(n_neighbors=k, algorithm=knnalgorithm, metric=metric).fit(train_data)
        else:
            self.knn = NearestNeighbors(n_neighbors=k, algorithm=knnalgorithm).fit(train_data)
        # self.knn = NearestNeighbors(n_neighbors=k, algorithm=knnalgorithm).fit(train_data)
        
    def searchForQuery(self, queryFeatures):
        sc = SparkContext()
        self.bc_knn = sc.broadcast(self.knn)
        
        distances, indices = sc.parallelize(queryFeatures).map(self.find).reduce(self.merge)
        return distances, indices

kList = [1, 5] #, 10, 20, 50, 100, 200]
knnObj = knn_mfcc("mfcc", 10, 30)
audioFeatures, audioPosnDict = knnObj.initializeFeatures(audioFileLocation, "audio")
print(len(audioFeatures))
queryFeatures, queryPosnDict = knnObj.initializeFeatures(queryAudioLocation, "query")
print(len(queryFeatures))
precisionatk = []
recallatk = []
for k in kList:
    knnObj.initKNN(k, "ball_tree", audioFeatures)
    print("Done initKNN")
    start = time.time()
    nnDistances, nnIndices = knnObj.searchForQuery(queryFeatures)
    print('Spend %.4f on KNN search for k = %d' % (time.time() - start, k))
    # print(len(nnDistances))
    print("Done finding nearest neighbors")
    resultchecker = indexer()
    precisionList, recallList = resultchecker.compareResults(nnIndices, nnDistances, audioPosnDict, queryPosnDict, 'mfccFeaturek10.txt', k)
    precisionatk.append(np.mean(np.array(precisionList)))
    recallatk.append(np.mean(np.array(recallList)))

plt.figure()
plt.plot(kList, precisionatk)
plt.xlabel("k")
plt.ylabel("Precision")
plt.title("Precision@k vs. k")
plt.savefig("Precisionatk.png")

plt.figure()
plt.plot(kList, recallatk)
plt.xlabel("k")
plt.ylabel("Recall")
plt.title("Recall@k vs. k")
plt.savefig("Recallatk.png")

plt.figure()
plt.scatter(precisionatk, recallatk)
for i in range(len(kList)):
    xy=(precisionatk[i],recallatk[i])
    plt.annotate(kList[i],xy)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("Recall vs. Precision")
plt.savefig("PrecisionRecallScatterPlot.png")

# kList = [1, 5, 10, 20, 50, 100, 200]
# knnObj = knn_mfcc("encoding", 10, 30)
# audioFeatures, audioPosnDict = knnObj.initializeFeatures(encodingLocation, "audio")
# print(len(audioFeatures))
# queryFeatures, queryPosnDict = knnObj.initializeFeatures(encodingLocation, "query")
# print(len(queryFeatures))
# precisionatk = []
# recallatk = []
# for k in kList:
#     knnObj.initKNN(k, "brute", audioFeatures, metric='cosine')
#     print("Done initKNN")
#     start = time.time()
#     nnDistances, nnIndices = knnObj.searchForQuery(queryFeatures)
#     print('Spend %.4f on KNN search for k = %d' % (time.time() - start, k))
#     # print(len(nnDistances))
#     print("Done finding nearest neighbors")
#     resultchecker = indexer()
#     precisionList, recallList = resultchecker.compareResults(nnIndices, nnDistances, audioPosnDict, queryPosnDict, 'mfccFeaturek10.txt', k)
#     precisionatk.append(np.mean(np.array(precisionList)))
#     recallatk.append(np.mean(np.array(recallList)))

# plt.figure()
# plt.plot(kList, precisionatk)
# plt.xlabel("k")
# plt.ylabel("Precision")
# plt.title("Precision@k vs. k")
# plt.savefig("EncodingPrecisionatk.png")

# plt.figure()
# plt.plot(kList, recallatk)
# plt.xlabel("k")
# plt.ylabel("Recall")
# plt.title("Recall@k vs. k")
# plt.savefig("EncodingRecallatk.png")

# plt.figure()
# plt.scatter(precisionatk, recallatk)
# for i in range(len(kList)):
#     xy=(precisionatk[i],recallatk[i])
#     plt.annotate(kList[i],xy)
# plt.xlabel("Precision")
# plt.ylabel("Recall")
# plt.title("Recall vs. Precision")
# plt.savefig("EncodingPrecisionRecallScatterPlot.png")

