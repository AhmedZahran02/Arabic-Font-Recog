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
        gaps = WordCutter.generateGaps(line)
        gapLength = WordCutter.sanatizeData(line,gaps[1:-1])
        filteredGaps = WordCutter.filterGaps(gaps[1:-1],gapLength)
        filteredGaps.append(gaps[-1])
        filteredGaps = [gaps[0]] + filteredGaps
        print(filteredGaps)
        for i in range(len(filteredGaps) - 1):
            word = line[0:line.shape[0],filteredGaps[i]:filteredGaps[i+1]]
            words.append(word)
        
        for gap in filteredGaps:    
            cv2.line(line, (gap, 0), (gap, line.shape[1]), (255, 255, 255), 1)
        ImageLoader.print(line)
        return words
    
    @staticmethod
    def generateGaps(line):
        gaps = []
        verticalHist = np.sum(line, axis=0)
        minOvershot = np.mean(verticalHist) * 0.5
        flag = 0
        i = 0
        while i < len(verticalHist) and verticalHist[i] == 0:
            i += 1
        while i < len(verticalHist):
            if verticalHist[i] == 0 and flag == 0:
                gaps.append(i)
                flag =1
            elif verticalHist[i] > minOvershot and flag ==1:
                flag=0
            i += 1
        startGap = 0
        i = 0
        while i < len(verticalHist) and verticalHist[i] == 0:
            startGap = i
            i+=1
            
        gaps = [startGap] + gaps
        tempLine = line.copy()
        for gap in gaps:    
            cv2.line(tempLine, (gap, 0), (gap, tempLine.shape[1]), (255, 255, 255), 1)
        ImageLoader.print(tempLine)
              
        return gaps
    
    @staticmethod
    def sanatizeData(line,gaps):
        gapLength = []

        horizontalHist = np.sum(line, axis=1)
        baseLineIndex = np.argmax(horizontalHist)
        
        for gap in gaps:
            i = gap
            rightHalfGapLength = 0
            while i < line.shape[1] and line[baseLineIndex][i] == 0:
                rightHalfGapLength+=1
                i+=1
            
            i = gap - 1
            leftHalfGapLength = 0
            while i > 0 and line[baseLineIndex][i] == 0:
                leftHalfGapLength+=1
                i-=1
            
            gapLength.append(rightHalfGapLength+leftHalfGapLength)     
        return gapLength
    
    @staticmethod
    def filterGaps(gaps,gapLength):
        IQR = np.percentile(gapLength, 75) - np.percentile(gapLength, 25)
        meanValue = np.mean(gapLength)
        print(IQR,meanValue)
        
        i = 0
        while i < len(gapLength):
            if gapLength[i] != 0 and gapLength[i] < (0.3 * IQR + 0.7 * meanValue):
                del gaps[i]
                del gapLength[i]
            else:
                i += 1
        
        return gaps