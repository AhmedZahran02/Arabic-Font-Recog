import numpy as np
import matplotlib.pyplot as plt
import cv2
from ImageLoader import *

class WordCutter:
    @staticmethod
    def extractWords(line):
        gaps = []
        gapLength = []
        words = []
        gaps,gapLength = WordCutter.generateGaps(line)
        gaps,gapLength = WordCutter.filterGaps(gaps,gapLength)
        
        for i in range(len(gaps) - 1):
            word = line[0:line.shape[0],gaps[i]:gaps[i+1]]
            words.append(word)
        
        for gap in gaps:    
            cv2.line(line, (gap, 0), (gap, line.shape[1]), (255, 255, 255), 1)
        ImageLoader.print(line)
        return words
    
    @staticmethod
    def generateGaps(line):
        gaps = []
        gapLength = []
        verticalHist = np.sum(line, axis=0)
        flag = 0
        i = 0
        while i+1 < len(verticalHist) and verticalHist[i+1] == 0 and verticalHist[i] == 0:
            i += 1
        while i < len(verticalHist):
            if verticalHist[i] == 0 and flag == 0:
                gapLength.append(i - (gaps[-1] if gaps else i))
                gaps.append(i)
                flag =1
            elif verticalHist[i] != 0 and flag ==1:
                flag=0
            i += 1
        
        tempLine = line.copy()
        for gap in gaps:    
            cv2.line(tempLine, (gap, 0), (gap, tempLine.shape[1]), (255, 255, 255), 1)
        ImageLoader.print(tempLine)
        return gaps,gapLength
    
    @staticmethod
    def filterGaps(gaps,gapLength):
        IQR = np.percentile(gapLength, 75) - np.percentile(gapLength, 25)
        print(gaps)
        print(gapLength)
        print(IQR)
        i = 0
        while i < len(gapLength):
            if gapLength[i] != 0 and gapLength[i] < IQR:
                del gaps[i]
                del gapLength[i]
            else:
                i += 1

        mean_value = np.mean(gapLength)

        i = 0
        while i < len(gapLength):
            if gapLength[i] < mean_value:
                del gaps[i]
                del gapLength[i]
            else:
                i += 1
        
        print(gaps)
        return gaps,gapLength