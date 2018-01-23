#classes to generate text similarity values
import torch
import sys, os
import logging
class Sent2Vec(object):
    def __init__(self):
        pass
    def getVec(self, sent):
        ''' get the vector for the given sent str'''
        pass
    def getBatchVec(self, sentList):
        ''' get corresponding vectors for all given sentences'''
        pass

class InferSent2Vec(Sent2Vec):
    def __init__(self, modelPath, glovPath):
        self.modelPath = modelPath
        self.glovPath = glovPath
        self.inferSent = None
        self.inferSent = torch.load(self.modelPath, map_location=lambda storage, loc: storage)
        self.inferSent.set_glove_path(self.glovPath)
        self.inferSent.build_vocab(["I like artificial intelligence"])

    def getVec(self, sent):
        if self.inferSent is None:
            return None
        self.inferSent.update_vocab([sent])
        embeddings = self.inferSent.encode([sent])
        return embeddings[0]

    def getBatchVec(self, sentList):
        if self.inferSent is None:
            return None
        self.inferSent.update_vocab(sentList)
        embeddings = self.inferSent.encode(sentList)
        return embeddings

def convSent2Vec():
    '''
    given the sent list, output the vector representation
    '''
    import numpy as np
    modelPath = sys.argv[1]
    glovPath = sys.argv[2]
    inputFile = sys.argv[3]
    outputFile = sys.argv[4]
    sent2Vec = InferSent2Vec(modelPath, glovPath)
    sentList = open(inputFile, "r").read().splitlines()
    logging.info("read in %d sents", len(sentList))
    sentVecList = sent2Vec.getBatchVec(sentList)
    with open(outputFile, "w") as fd:
        for sent, sentVec in zip(sentList, sentVecList):
            if type(sentVec) is not np.ndarray:
                logging.warning("unknown sent vec for sent %s: %s", sent, sentVec)
                continue
            vecStr = sentVec.dumps()
            vecStr = vecStr.encode("hex")
            fd.write("{}||{}\n".format(sent, vecStr))

def calSimFromVec():
    '''
    given question-answer pairs and text-vec mappings, calculate consine similarity between each q-a pair
    '''
    import numpy as np
    import scipy.spatial.distance as distUtil
    inputVecFile = sys.argv[1]
    inputPairFile = sys.argv[2]
    outputFile = sys.argv[3]
    sentVecDict = {}
    pairDict = {}#key is (q, a), value is freq
    resultDict = {} #key is (q, a), value is freq, cosineSim
    pairList = []
    #read in vec dict
    with open(inputVecFile, "r") as fd:
        for line in fd:
            attrs = line.strip().split("||")
            sent = attrs[0]
            vecStr = attrs[1].decode("hex")
            vecArray = np.loads(vecStr)
            sentVecDict[sent] = vecArray
    logging.info("loaded %d string-vec mappings", len(sentVecDict))
    #read in pair dict
    with open(inputPairFile, "r") as fd:
        for line in fd:
            attrs = line.split("||")
            qText = attrs[0]
            aText = attrs[2]
            freq = int(attrs[4])
            pairDict[(qText, aText)] = freq
            pairList.append((qText, aText))
    #cal cosineSimilarity
    for qaKey in pairDict:
        qText, aText = qaKey
        freq = pairDict[qaKey]
        qVec = sentVecDict[qText] if qText in sentVecDict else None
        aVec = sentVecDict[aText] if aText in sentVecDict else None
        if aVec is None or qVec is None:
            logging.warning("no vec found for aText or qText: %s||%s", aText, qText)
            simValue = None
        else:
            simValue = 1 - distUtil.cosine(qVec, aVec)
        resultDict[qaKey] = (freq, simValue)
    with open(outputFile, "w") as fd:
        for qaKey in pairList:
            freq, simValue = resultDict[qaKey]
            resultList = []
            resultList.extend(list(qaKey))
            resultList.append(freq)
            resultList.append(simValue)
            resultStr = "||".join([str(item) for item in resultList])
            fd.write(resultStr + "\n")
    return True
if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    convSent2Vec()
    #calSimFromVec()
    sys.exit(1)
    import scipy as sci
    modelPath = sys.argv[1]
    glovPath = sys.argv[2]
    sent2Vec = InferSent2Vec(modelPath, glovPath)
    sentList = ["which music do you like", "any one please", "please choose one for me", "please stop", "which music do you want"]
    sentVecList = sent2Vec.getBatchVec(sentList)
    for i in range(len(sentList)):
        for j in range(i + 1, len(sentList)):
            sentI = sentList[i]
            sentJ = sentList[j]
            sentVecI = sentVecList[i]
            sentVecJ = sentVecList[j]
            cosineSim = 1 - sci.spatial.distance.cosine(sentVecI, sentVecJ)
            print(sentI, sentJ, cosineSim)

