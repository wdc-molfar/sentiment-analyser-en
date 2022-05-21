#!/usr/bin/env python3 -W ignore::DeprecationWarning


# -*- coding: utf-8 -*-
"""
Created on Thu May 12 18:43:37 2022

@author: dmytrenko.o
"""

import json
import io
import sys, os
import warnings
import traceback
import fasttext

stdOutput = open("outlog.log", "w")
sys.stderr = stdOutput
sys.stdout = stdOutput

#load model
model = fasttext.load_model(os.path.join(os.getcwd()+'/models/en.ftz'))

sys.stdout = sys.__stdout__
sys.stderr = sys.__stdout__

warnings.filterwarnings("ignore", message=r"\[W033\]", category=UserWarning)
input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

if __name__=='__main__':
    
    input_json = None
    for line in input_stream:
        
        # read json from stdin
        input_json = json.loads(line)
        try:
            output = input_json.copy()
            text = input_json["service"]["scraper"]["message"]["text"]
        except BaseException as ex:
            ex_type, ex_value, ex_traceback = sys.exc_info()            

            output = {"error": ''}           
            output['error'] += "Exception type : %s; \n" % ex_type.__name__
            output['error'] += "Exception message : %s\n" %ex_value
            output['error'] += "Exception traceback : %s\n" %"".join(traceback.TracebackException.from_exception(ex).format())
        
        try:
            predict = model.predict(text, k = 2)
            if (predict[0][0] == '__label__pos') and (predict[1][0] >= 0.9):
                emotion = "Good"
            elif (predict[0][0] == '__label__neg') and (predict[1][0] >= 0.9):
                emotion = "Bad"
            else:
                emotion = "None"
            prediction = dict()
            prediction["pos"] = float(predict[1][0])
            prediction["neg"] = float(predict[1][1])
            prediction["em"] = emotion
            output["service"]["sentimentanalyser"] = str(prediction)
        except BaseException as ex:
             ex_type, ex_value, ex_traceback = sys.exc_info()            
                       
             output['error'] += "Exception type : %s; \n" % ex_type.__name__
             output['error'] += "Exception message : %s\n" %ex_value
             output['error'] += "Exception traceback : %s\n" %"".join(traceback.TracebackException.from_exception(ex).format())
         
        
        output_json = json.dumps(output, ensure_ascii=False).encode('utf-8')
        sys.stdout.buffer.write(output_json)
        print ()
        