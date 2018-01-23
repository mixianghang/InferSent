#!/usr/bin/python
import logging
import os, sys
from collections import defaultdict
def binQASim():
    '''
    group qa pairs in different bins of sent similarity value
    '''
    qaSimFile = sys.argv[1]
    binStepValue = float(sys.argv[2])
    resultDir = sys.argv[3]
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)

    binRange = 1
    minIndex = int(float(binRange) / binStepValue)  + 1
    maxIndex = 0
    resultBinDict = defaultdict(list)
    qaPairNum = 0
    with open(qaSimFile, "r") as fd:
        for line in fd:
            line = line.strip()
            qaPairNum += 1
            attrs = line.split("||")
            simValue = float(attrs[3])
            currIndex = int(simValue / binStepValue)
            if minIndex > currIndex:
                minIndex = currIndex
            if maxIndex < currIndex:
                maxIndex = currIndex
            resultBinDict[currIndex].append(attrs)
    logging.info("read in %d qa pairs,  index range  is %d : %d", qaPairNum, minIndex, maxIndex)
    disFd = open(os.path.join(resultDir, "simDistribution.txt"), "w")
    for currIndex in resultBinDict:
        currSmall = currIndex * binStepValue
        currLarge = currSmall + binStepValue
        currSubDir = os.path.join(resultDir, "bin_{small}_{large}".format(small = currSmall, large = currLarge))
        if not os.path.exists(currSubDir):
            os.makedirs(currSubDir)
        currQaList = resultBinDict[currIndex]
        disFd.write("{small}||{large}||{pairNum}||{pairPercent:.2f}%\n".format(small = currSmall, large = currLarge, pairNum = len(currQaList), pairPercent = float(100 * len(currQaList)) / qaPairNum))
        listSortByFreq = sorted(currQaList, key = lambda item : float(item[2]))
        listSortBySim = sorted(currQaList, key = lambda item : float(item[3]))
        resultFileList = ["qa_pair_sorted-freq.txt", "qa_pair_sorted-sim.txt"]
        resultListList = [listSortByFreq, listSortBySim]
        for resultList, resultFileName in zip(resultListList, resultFileList):
            with open(os.path.join(currSubDir, resultFileName), "w") as fd:
                for item in resultList:
                    fd.write("||".join(item) + "\n")
    disFd.close()

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    binQASim()
