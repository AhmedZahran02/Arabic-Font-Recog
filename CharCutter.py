import cv2
import numpy as np
from SeparationRegion import *
from statistics import mode
from skimage.morphology import skeletonize


class CharCutter:
    @staticmethod
    def extractCharacters(line):
        thinned_lines = CharCutter.ThinningLineGeneral(line)
        #print(thinned_lines)
        #smoothing line
        BaseLine = []
        MaxTransitions = []
        SeparationRegions = []
        words = []
        for wordIndex,word in enumerate(thinned_lines):
            BaseLine.append(CharCutter.BaseLineDetection(word))
            MaxTransitions.append(CharCutter.MaximumTransitions(word,BaseLine[-1]))
            SeparationRegions.append(CharCutter.CutPointIdentification(word,MaxTransitions[-1]))
            # SeparationRegions = CharCutter.filterRegions()
            for i in range(0,len(SeparationRegions[wordIndex])):
                cv2.line(word,(SeparationRegions[wordIndex][i].CutIndex,0),(SeparationRegions[wordIndex][i].CutIndex,word.shape[1]),(255, 255, 255), 1)
            words.append(word)   
        return words
    
    @staticmethod
    def BaseLineDetection(word):
        #print(word)
        HorizontalHist = np.sum(word, axis=1)
        BaseLineIndex = 0
        MaximumSum = HorizontalHist[0] 
        #print(HorizontalHist)
        #idx = 1
        for idx in range (1,len(HorizontalHist)):
            if HorizontalHist[idx] > MaximumSum:
                MaximumSum = HorizontalHist[idx]
                BaseLineIndex = idx
        return BaseLineIndex
    
    @staticmethod
    def MaximumTransitions(word,baseLineIndex):
        MaxTransitions = 0
        MaxTransitionsIndex = baseLineIndex
        idx = baseLineIndex
        while idx >= 0:
            CurrentTransitions = 0
            Flag = 0
            for j in range(len(word[0])):
                if word[idx][j] >= 1 and Flag == 0:
                    CurrentTransitions+=1
                    Flag = 1
                elif word[idx][j] == 0 and Flag == 1:
                    Flag = 0
            if CurrentTransitions >= MaxTransitions:
                MaxTransitions = CurrentTransitions
                MaxTransitionsIndex = idx
            #print(CurrentTransitions)
            idx-=1
        return MaxTransitionsIndex 
    
    @staticmethod
    def CutPointIdentification(word,MaxTransitionsIndex):
        kernel = np.ones((2,2),np.uint8)
        # ll = np.array(word)
        # print(ll)
        opening_image = cv2.morphologyEx(np.array(word).astype('uint8'),cv2.MORPH_OPEN,kernel)
        VerticalHist = np.sum(word, axis=0)
        MostPixelsVertical = mode(VerticalHist)
        idx = 0
        Flag = 0
        SeparationRegions = []
        while idx < len(word[0]):
            if word[MaxTransitionsIndex][idx] >= 1 and Flag == 0:
                SR = SeparationRegion(0,0,0)
                SeparationRegions.append(SR)
                SeparationRegions[-1].EndIndex = idx
                Flag = 1
                # print(SeparationRegions[-1].EndIndex)
            elif word[MaxTransitionsIndex][idx] == 0 and Flag == 1:
                SeparationRegions[-1].StartIndex = idx
                MidIndex = int((SeparationRegions[-1].StartIndex + SeparationRegions[-1].EndIndex) / 2)
                ValidCut = []
                # print(MidIndex)
                if VerticalHist[MidIndex] == MostPixelsVertical:
                    SeparationRegions[-1].CutIndex = MidIndex
                else:
                    for k in range(SeparationRegions[-1].EndIndex,SeparationRegions[-1].StartIndex):
                        if VerticalHist[k]  == 0 or (VerticalHist[k] <= MostPixelsVertical and VerticalHist[k] <= SeparationRegions[-1].EndIndex) or (VerticalHist[k] <= MostPixelsVertical and k < SeparationRegions[-1].StartIndex and k > MidIndex):
                            ValidCut.append(k)
                    if len(ValidCut) > 0:
                        SeparationRegions[-1].CutIndex = min(ValidCut,key=lambda x:abs(x-MidIndex))
                    else:
                        SeparationRegions[-1].CutIndex = MidIndex
                Flag = 0
            idx+=1
        return SeparationRegions
    
    @staticmethod
    def ThinningLineGeneral(line):
        ThinnedLines = []
        for word in line:
            ThinnedLines.append(CharCutter.ThinningLine(word))
        return ThinnedLines
    
    @staticmethod
    def ThinningLine(word):
        kernel = np.ones((2,2),np.uint8)
        thinned_line = cv2.erode(np.array(word).astype('uint8'),kernel,iterations=2)
        return thinned_line