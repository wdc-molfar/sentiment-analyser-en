#!/usr/bin/env python3
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


#stdOutput = open("outlog.log", "w")
#sys.stderr = stdOutput
#sys.stdout = stdOutput

#from __modules__ import packagesInstaller
#packages = ['os', 'fasttext', 'time']
#packagesInstaller.setup_packeges(packages)

from __modules__ import configLoader, modelsLoader, sentimentAnalyser

#load languages by default
langModels = configLoader.load_default_languages(os.getcwd())
#load models by default
models = modelsLoader.load_models(os.getcwd(), langModels)
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
            lang = input_json["service"]["scraper"]["message"]["lang"]
        except BaseException as ex:
            ex_type, ex_value, ex_traceback = sys.exc_info()            

            output = {"error": ''}           
            output['error'] += "Exception type : %s; \n" % ex_type.__name__
            output['error'] += "Exception message : %s\n" %ex_value
            output['error'] += "Exception traceback : %s\n" %"".join(traceback.TracebackException.from_exception(ex).format())
        
        if lang == "en":
            try:
                prediction = sentimentAnalyser.predict_emotion(text, models[lang], configLoader.default_value(os.getcwd(), "predictLimit"))
                
                output["service"]["sentimentanalyser"] = str(prediction)
            except BaseException as ex:
                 ex_type, ex_value, ex_traceback = sys.exc_info()            
                 
                 output = {"error": ''}           
                 output['error'] += "Exception type : %s; \n" % ex_type.__name__
                 output['error'] += "Exception message : %s\n" %ex_value
                 output['error'] += "Exception traceback : %s\n" %"".join(traceback.TracebackException.from_exception(ex).format())
             
            
            output_json = json.dumps(output, ensure_ascii=False).encode('utf-8')
            sys.stdout.buffer.write(output_json)
            print ()
        