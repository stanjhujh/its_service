# TickDirectionPrediction(saved 31Mar2025).py

import pickle
import sys
import json
import os
import pandas as pd
import numpy as np
import xgboost as xgb # XGBoost is an opensource library
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
import pickle
warnings.filterwarnings("ignore")

import datetime
import traceback

import socket
import atexit

import socketserver

MODEL_FILE_EXT = '_model.ubj'
META_FILE_SUFFIX = '_meta.pkl'

# both resturn Z:\fitted_models
# logFile.write("sys.path[0]: "+sys.path[0]+'\n'+" os.path.dirname(os.path.realpath(__file__))): "+os.path.dirname(os.path.realpath(__file__))+'\n')
# logFile.flush()


# log file to store the errors
#logFile = open ('Z:\\logExitRules.txt',"wt")
logFile = open (sys.path[0][0]+':\\logExitRules.txt',"wt") # get the drive letter



    
def calculate_red_yellow_green_probabilities(trade_type_and_designation,ticks_since_trade_entry,predictor_data,column_names, directory):

    try:
        # Load the appropriate model
        model_filename = trade_type_and_designation + "_" + str(ticks_since_trade_entry) + MODEL_FILE_EXT
        model_filename = os.path.join(directory, model_filename)
        
        if False :
            logFile.write("model_filename: "+model_filename+'\n')
            logFile.flush()

        meta_filename = trade_type_and_designation + "_" + str(ticks_since_trade_entry) + META_FILE_SUFFIX
        meta_filename = os.path.join(directory, meta_filename)
        # print(model_filename + " " + meta_filename)

        # Check if the model file exists
        if not os.path.isfile(model_filename):
            logFile.write('Model file '+model_filename+' cannot be found in directory ' + directory+' ' +str(datetime.datetime.now())+'\n')
            logFile.flush()
            raise Exception("Model folder cannot be found:" + directory)

        # Load the model
        this_model = xgb.XGBClassifier()
        this_model.load_model(model_filename)

        # Put the data into a dataframe
        X_pred = pd.DataFrame([predictor_data],columns=column_names)

        # Make a prediction
        probs = this_model.predict_proba(X_pred)[0]
        
        # Load the meta data
        with open(meta_filename, 'rb') as f:
            this_labels_dict = pickle.load(f)

        # Using the meta data, map the probabilities to a 3-column array
        # (It may be fewer than 3 initially if some trade types didn't exist in the training data)
        ryg_probs = [0.0]*3
        
        for key, value in this_labels_dict.items():
            if value == -1:
                ryg_probs[0] = probs[key]
            elif value == 0:
                ryg_probs[1] = probs[key]
            else: # value == 1
                ryg_probs[2] = probs[key]
        
        # Convert numpy.float32 to standard Python float
        ryg_probs = [float(prob) for prob in ryg_probs]
#    except:
    except Exception as e:
        logFile.write('Error: '+traceback.format_exc()+' ' +str(datetime.datetime.now())+ '\n') 
        logFile.flush()
        # If anything has gone wrong then return all zeros
        ryg_probs = [0.0, 0.0, 0.0]

    return [ryg_probs]

def xgmodel(json_data, column_names, trade_type, trade_designation, bars_since_trade, directory):
    
    df = pd.DataFrame(json_data)
    logFile.write("xgmodel:pd.DataFrame(json_data): "+str(df)+'\n')
    logFile.flush()    
    
    column_names = column_names
    # logFile.write("\n xgmodel:column_names: "+str(column_names)+'\n')
    # logFile.flush()    
    # logFile.write("\n xgmodel:trade_type: "+str(trade_type)+' trade_type='+trade_type+'\n')
    # logFile.flush()    

    # logFile.write("\n xgmodel:df['TradeType'] "+str(df["TradeType"])+'\n')
    # logFile.flush() 
    # logFile.write("\n xgmodel:df['TradeDesignation'] "+str(df["TradeDesignation"])+ 'trade_designation='+trade_designation+'\n')
    # logFile.flush() 
    # logFile.write("\n xgmodel:df['BarsSinceTradeEntry'] "+str(df["BarsSinceTradeEntry"])+' bars_since_trade='+str(bars_since_trade)+'\n')
    # logFile.flush() 
    
    this_model_row = \
                         (df["TradeType"] == trade_type) \
                         & (df["TradeDesignation"] == trade_designation) \
                         & (df["BarsSinceTradeEntry"] == bars_since_trade)
    # logFile.write('\n df["TradeType"] == trade_type:' +str(df["TradeType"] == trade_type)+'\n')
    # logFile.write('\n df["TradeDesignation"] == TradeDesignation ? ' +str(df["TradeDesignation"] == trade_designation)+'\n')
    # logFile.write('\n df["BarsSinceTradeEntry"] == BarsSinceTradeEntry ? ' +str(df["BarsSinceTradeEntry"] == bars_since_trade)+'\n') # False !?
    # logFile.write('\n\t df["BarsSinceTradeEntry"] > BarsSinceTradeEntry ? ' +str(df["BarsSinceTradeEntry"] > bars_since_trade)+'\n') # 
    # logFile.write('\n\t df["BarsSinceTradeEntry"] < BarsSinceTradeEntry ? ' +str(df["BarsSinceTradeEntry"] < bars_since_trade)+'\n') # 
    # logFile.flush()    
    # logFile.write("\n this_model_row: "+str(this_model_row)+'\n')
    # logFile.flush()    

    this_X_test = df.loc[this_model_row,column_names]  # Empty DataFrame
    # logFile.write("\n this_X_test: "+str(this_X_test)+'\n')
    # logFile.flush()    

    if this_X_test.empty :
        print('Exception in xgmodel(json_data,column_names,', trade_type, trade_designation, bars_since_trade, directory,'): Empty DataFrame')
        logFile.write('\n Exception in xgmodel(json_data,column_names,'+ trade_type + trade_designation +str( bars_since_trade)+ directory+'): Empty DataFrame'+'\n')
        logFile.flush() 
        probabilities = [] # <class 'list'>
        probabilities.append([0.0, 0.0, 0.0])
        #return probabilities
        return str(probabilities)               #  'list' object has no attribute 'encode'' occurs when trying to use encode on a list instead of a string. 
                                                # To fix this, use list comprehension to call encode on each string in the list. This creates a new list with the encoded strings.
    
    trade_type_and_designation = trade_type + "_" + trade_designation
    ticks_since_trade_entry = bars_since_trade
    
    predictor_data = this_X_test.iloc[0][column_names].values.tolist()  # this_X_test.iloc[0]:  IndexError: single positional indexer is out-of-bounds
                                                                        # one of the DataFrames doesn't have the number of rows or columns you expect it to have. 
                                                                        # iloc[0] : first row, which does not exist since this_X_test = Empty DataFrame
    logFile.write("\n predictor_data: "+str(predictor_data)+'\n')
    logFile.flush()    

    probabilities = calculate_red_yellow_green_probabilities(trade_type_and_designation,ticks_since_trade_entry,predictor_data,column_names,directory)
    logFile.write("\n probabilities: "+str(probabilities)+str(type(probabilities[0]))+'\n')
    logFile.flush()    
   
#    return probabilities[0]
    return str(probabilities[0])
    
def cleanup() :     
    HOST = sys.argv[1]  # '127.0.0.1' 
    PORT = sys.argv[2]  # 20011    
    
    # this free_port can be used when accessed from cmd telnet  https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    with socketserver.TCPServer(("localhost", 0), None) as s:
        free_port = s.server_address[1]
    #    PORT = free_port
    
    logFile.write('Script num. of arguments '+ str(len(sys.argv))+ ' '+str(datetime.datetime.now())+'\n')
    logFile.write('sys.argv[0] '+str( sys.argv[0])+'\n')
    logFile.write('HOST '+ str(sys.argv[1])+'\n')
    logFile.write('PORT '+ sys.argv[2]+ '----->'+PORT)
    logFile.flush()
    
    print('Script num. of arguments ', len(sys.argv), datetime.datetime.now())
    print('sys.argv[0] ', sys.argv[0])
    print('HOST ', sys.argv[1])
    print('PORT ', sys.argv[2], '----->',PORT)

    logFile.write('datetime.datetime.now(): '+ str(datetime.datetime.now())+'\n')
        
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((HOST, int(PORT)))  # OSError: [WinError 10048] Only one usage of each socket address (protocol/network address/port) is normally permitted
                            # running same code with same port #, i.e. port alreday in use; use a different port 
    except Exception:
        logFile.write('Error: '+traceback.format_exc()+' '+str( datetime.datetime.now()))
        #logFile.write('\nFix: try a different port from '+str(PORT) +'\n')
        logFile.flush()      
        #print('\nFix: try a different port from '+str(PORT) +'\n')
        print('Error: '+traceback.format_exc()+' '+str( datetime.datetime.now()))
        quit()

    backlog = 10        # max. length of the pending connections queue
    s.listen(backlog) # places  a socket in a listening state; retrieve MaxConnections, system specific: 2147483647
    
    logFile.write('Server listening '+str(datetime.datetime.now())+'\n')
    logFile.flush()
    print('Exit rules Pyhon server is listening')
    
    # continously waiting for TCP client
    while True:    
        try:
            conn, addrs = s.accept()
            print('\tconn='+str(conn)+'\n\t adrs='+str(addrs)+'\n')
            package_bytes = conn.recv(1024) # as bytes
            package_str = package_bytes.decode() # 
            
            print(' received '+str(len(package_str))+': '+package_str+'\n')
            logFile.write(' received '+str(len(package_str))+': '+package_str+' ' +str( datetime.datetime.now())+'\n')
            logFile.flush()
                           
            if not package_bytes: break

            message = package_str  
            
            items_bytes = package_bytes.split(b' ') # json_file_path asset, Long/Short, PV1,  barNumber ; b tands for byte-typem otherwise TypeError: a bytes-like object is required, not 'str'
            # message  = calculate_entry_probabilities(items_bytes[0].decode("utf-8") , items_bytes[1].decode("utf-8") , items_bytes[2].decode("utf-8") )  # this is something like '0.1 0.5 0.4' (no appostroph)
            json_file_path = items_bytes[0].decode("utf-8")  # dataPath
            # logFile.write('json_file_path: '+ json_file_path+'\n')
            # logFile.flush()
            with open(json_file_path, 'r') as file:
                json_data = json.load(file)           
        
            column_file_path = items_bytes[1].decode("utf-8")  # columnsPath
            # logFile.write('column_file_path: '+ column_file_path+'\n')
            # logFile.flush()
            with open(column_file_path, 'r') as file:
                column_names = json.load(file)
        
            trade_type = items_bytes[2].decode("utf-8")
            # logFile.write('trade_type: '+ trade_type+'\n')
            # logFile.flush()
            trade_destination = items_bytes[3].decode("utf-8")
            # logFile.write('trade_destination: '+ trade_destination+'\n')
            # logFile.flush()
            bars_since_trade = int(items_bytes[4].decode("utf-8"))
            # logFile.write('bars_since_trade: '+ str(bars_since_trade)+'\n')
            # logFile.flush()
            directory = items_bytes[5].decode("utf-8")          # modelsPath +assetName
            # logFile.write('directory: '+ directory+'\n')
            # logFile.flush()
    
            message = xgmodel(json_data, column_names, trade_type, trade_destination, bars_since_trade, directory)
            print(json.dumps(message))
            
            conn.sendall(message.encode())    
            
        except Exception:
            print('json_data: ',json_data,'\n', 'column_names: ',column_names,'\n', 'trade_type: ',trade_type,'\n',' trade_destination: ',trade_destination,'\n','bars_since_trade: ','\n','directory: ',directory)

            print('Ignored: '+traceback.format_exc(),'\n',str( datetime.datetime.now()))
            logFile.write('Ignored: '+traceback.format_exc()+'\n'+str( datetime.datetime.now()))
            logFile.flush()
            
             
            pass # quit() ; keep Python.exe opened
    # end while  ----------------------------------------
                
    print('Out from while loop , this shoud not happen \n'+str( datetime.datetime.now()))
    logFile.write('Out from while loop , this shoud not happen'+'\n'+str( datetime.datetime.now()))
    logFile.flush()
        
    # if __name__ == "__main__":
        # json_file_path = sys.argv[1]
        # with open(json_file_path, 'r') as file:
            # json_data = json.load(file)
        
        # column_file_path = sys.argv[2]
        # with open(column_file_path, 'r') as file:
            # column_names = json.load(file)

        # trade_type = sys.argv[3]
        # trade_destination = sys.argv[4]
        # bars_since_trade = int(sys.argv[5])
        # directory = sys.argv[6]

        # c1 = xgmodel(json_data, column_names, trade_type, trade_destination, bars_since_trade, directory)
        # print(json.dumps(c1))
        
atexit.register(cleanup)        