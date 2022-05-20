#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dmytrenko.o
"""

#import sys
#stdOutput = open("outlog.log", "w")
#sys.stderr = stdOutput
#sys.stdout = stdOutput

def predict_emotion(text, model, predictLimit):
    try:
        predict = model.predict(text, k = 2)
        if (predict[0][0] == '__label__pos') and (predict[1][0] >= predictLimit):
            emotion = "Good"
        elif (predict[0][0] == '__label__neg') and (predict[1][0] >= predictLimit):
            emotion = "Bad"
        else:
            emotion = "None"
        prediction = dict()
        prediction["pos"] = float(predict[1][0])
        prediction["neg"] = float(predict[1][1])
        prediction["em"] = emotion
    except:
        emotion = "None"
        print ("¯\_(ツ)_/¯ Unexpectable Error while emotion predicting!")  
        pass
    return prediction
