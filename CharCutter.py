import cv2
import numpy as np
from SeparationRegion import *
from statistics import mode
from skimage.morphology import skeletonize
from ImageLoader import *

class CharCutter:
    @staticmethod
    def extractCharacters(line):
        thinned_lines = CharCutter.ThinningLineGeneral(line)
        #print(thinned_lines)
        #smoothing line
        BaseLine = []
        MaxTransitions = []
        SeparationRegions = []
        ValidSeparationRegions = []
        words = []
        SeparatedCharacters = []
        for wordIndex,word in enumerate(thinned_lines):
            BaseLine.append(CharCutter.BaseLineDetection(word))
            MaxTransitions.append(CharCutter.MaximumTransitions(word,BaseLine[-1]))
            print(MaxTransitions)
            SeparationRegions.append(CharCutter.CutPointIdentification(word,MaxTransitions[-1]))
            MostFrequentPixelsVertical = CharCutter.MostFrequentPixelsV(word)
            ValidSeparationRegions.append(CharCutter.SeparationRegionsFiltration(word,SeparationRegions[-1],BaseLine[-1],MaxTransitions[-1],MostFrequentPixelsVertical))
            print("Valid Separation regions for word " + str(wordIndex) + " : ")
            print("S length: " + str(len(SeparationRegions[-1])))
            print("V length: " + str(len(ValidSeparationRegions[-1])))
            prevCutIndex = 0
            for i in range(0,len(ValidSeparationRegions[-1])):
                SeparatedCharacters.append(word[:,prevCutIndex:int(ValidSeparationRegions[-1][i].CutIndex+1)])
                prevCutIndex = ValidSeparationRegions[-1][i].CutIndex
            words.append(word)   
            # for i in range(0,len(ValidSeparationRegions[wordIndex])):
            #     cv2.line(word,(ValidSeparationRegions[wordIndex][i].CutIndex,0),(ValidSeparationRegions[wordIndex][i].CutIndex,word.shape[1]),(255, 255, 255), 1)
        return SeparatedCharacters
    
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
    def MostFrequentPixelsV(word):
        VerticalHist = np.sum(word, axis=0) / 255
        VerticalHist = VerticalHist.astype(int)
        #print(VerticalHist)
        MostFrequentPixelsVertical = mode(VerticalHist)
        if MostFrequentPixelsVertical == 0:
            count_dict = {}
            for num in VerticalHist:
                count_dict[num] = count_dict.get(num, 0) + 1
            sorted_counts = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
            MostFrequentPixelsVertical = sorted_counts[1][0]
        return MostFrequentPixelsVertical

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
        print(MaxTransitions)
        return MaxTransitionsIndex 
    
    @staticmethod
    def CutPointIdentification(word,MaxTransitionsIndex):
        kernel = np.ones((2,2),np.uint8)
        # ll = np.array(word)
        # print(ll)
        opening_image = cv2.morphologyEx(np.array(word).astype('uint8'),cv2.MORPH_OPEN,kernel)
        #ImageLoader.print(opening_image)
        #print(VerticalHist)
        MostFrequentPixelsVertical = CharCutter.MostFrequentPixelsV(word)
        VerticalHist = np.sum(word, axis=0) / 255
        VerticalHist = VerticalHist.astype(int)
        idx = 0
        Flag = 0
        SeparationRegions = []
        while idx < len(word[0]):
            if word[MaxTransitionsIndex][idx] == 0 and Flag == 1:
                SR = SeparationRegion(0,0,0)
                SeparationRegions.append(SR)
                SeparationRegions[-1].EndIndex = idx
                Flag = 0
                Flag2 = 1
                # print(SeparationRegions[-1].EndIndex)
            elif word[MaxTransitionsIndex][idx] >= 1 and Flag == 0:
                if len(SeparationRegions) > 0:
                    SeparationRegions[-1].StartIndex = idx
                    MidIndex = int((SeparationRegions[-1].StartIndex + SeparationRegions[-1].EndIndex) / 2)
                    ValidCut = []
                    # print(MidIndex)
                    if VerticalHist[MidIndex] == MostFrequentPixelsVertical  or VerticalHist[MidIndex]  == 0:
                        SeparationRegions[-1].CutIndex = MidIndex
                    else:
                        for k in range(SeparationRegions[-1].EndIndex,SeparationRegions[-1].StartIndex):
                            if VerticalHist[k]  == 0 or (VerticalHist[k] <= MostFrequentPixelsVertical ) or (VerticalHist[k] <= MostFrequentPixelsVertical and k < SeparationRegions[-1].StartIndex and k > MidIndex):
                                ValidCut.append(k)
                        if len(ValidCut) > 0:
                            SeparationRegions[-1].CutIndex = min(ValidCut,key=lambda x:abs(x-MidIndex))
                        else:
                            SeparationRegions[-1].CutIndex = MidIndex
                Flag = 1
            idx+=1
        #print(len(SeparationRegions))
        return SeparationRegions

    @staticmethod
    def are_pixels_connected(image, MaxTransitionsIndex):    
        _, temp_image, _, _ = cv2.floodFill(image.copy(),None,(0,MaxTransitionsIndex),255)
        return (temp_image[MaxTransitionsIndex,-1] == 255)


    @staticmethod
    def SeparationRegionsFiltration(word,SeparationRegions,baseLineIndex,MaxTransitionsIndex,MostFrequentPixelsVertical):
        i = 0
        ValidSeparationRegions = []
        VerticalHist = np.sum(word, axis=0) / 255
        VerticalHist = VerticalHist.astype(int)
        print(len(SeparationRegions))
        while i < len(SeparationRegions) and SeparationRegions[i].StartIndex > SeparationRegions[i].EndIndex:
            end = SeparationRegions[i].EndIndex - 1
            start = SeparationRegions[i].StartIndex
            cut = SeparationRegions[i].CutIndex
            if VerticalHist[SeparationRegions[i].CutIndex] == 0:
                ValidSeparationRegions.append(SeparationRegions[i])
            else:
                #ImageLoader.print(word[:,end-1:start])
                print(i)
                print("End: " + str(end))
                print("Start: " + str(start))
                cut_word = word[:,end:start+1]
                print("End Pixel: " + str(cut_word[MaxTransitionsIndex][0]))
                print("Start Pixel: " + str(cut_word[MaxTransitionsIndex][-1]))
                if cut_word[MaxTransitionsIndex][0] >= 1 and cut_word[MaxTransitionsIndex][-1] >=1 and CharCutter.are_pixels_connected(cut_word,MaxTransitionsIndex) == False:
                    ValidSeparationRegions.append(SeparationRegions[i])
                elif VerticalHist[cut] <= MostFrequentPixelsVertical:
                    ValidSeparationRegions.append(SeparationRegions[i])
            i+=1
        return ValidSeparationRegions
    
    @staticmethod
    def ThinningLineGeneral(line):
        ThinnedLines = []
        for word in line:
            ThinnedLines.append(CharCutter.ThinningLine(word))
        return ThinnedLines
    
    @staticmethod
    def ThinningLine(word):
        kernel = np.ones((2,2),np.uint8)
        thinned_line = cv2.erode(np.array(word).astype('uint8'),kernel,iterations=1)
        return thinned_line