# Python program to get the real-time 
# currency exchange rate 
from __future__ import print_function 
from array import *
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

USD_INR = 0
INR_USD = 0
USD_CAD = 0
CAD_USD = 0
INR_CAD = 0
CAD_INR = 0

# Function to get real time currency exchange  
def RealTimeCurrencyExchangeRate(from_currency, to_currency, api_key) : 
      # importing required libraries 
      import requests
      import json 
      import os
    # base_url variable store base url  

      base_url = "https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE"
    # main_url variable store complete url 
      main_url = base_url + "&from_currency=" + from_currency +  "&to_currency=" + to_currency + "&apikey=" + api_key 
  
    # get method of requests module  
   
      req_ob = requests.get(main_url) 
  
    # json method return json format 
    # data into python dictionary data type. 
      
    # result contains list of nested dictionaries 
      result = req_ob.json() 
  
      print("\n After parsing : \n Realtime Currency Exchange Rate for", 
          result["Realtime Currency Exchange Rate"] 
                ["2. From_Currency Name"], 'to', 
          result["Realtime Currency Exchange Rate"] 
                ["4. To_Currency Name"], 'is', 
          result["Realtime Currency Exchange Rate"] 
                ['5. Exchange Rate'], to_currency) 

      return result["Realtime Currency Exchange Rate"]['5. Exchange Rate']

# USD = 1, CAD = 2, INR = 3, If Current and Currency contain same number on same index, then deposit half of the money.
def nonParallelConversion () :
      current = []
      for i in range(10):
          current.append(random.randrange(1, 4, 1))
      print("Current currencies: ", current)
    
      new = []
      for i in range(10):
          new.append(random.randrange(1, 4, 1))
      print("Currencies converting to: ", new)

      currency = []
      for i in range(10):
          currency.append(random.randrange(1, 101, 1))
      print("Current currency values: ", currency)

      for i in range(len(current)):
            if current[i] == 1:
                  if new[i] == 2:
                        currency[i] *= USD_CAD
                  elif new[i] == 3:
                        currency[i] *= USD_INR
                  else:
                        currency[i] /= 2.0
            elif current[i] == 2:
                  if new[i] == 1:
                        currency[i] *= CAD_USD
            # Took out CAD_INR
                  else:
                        currency[i] /= 2.0
            elif current[i] == 3:
                  if new[i] == 1:
                        currency[i] *= INR_USD
                  elif new[i] == 2:
                        currency[i] *= INR_CAD 
                  else:
                        currency[i] /= 2.0          
      print("New currency values: ", currency)

# USD = 0, CAD = 1, INR = 2, If Current and new contain same number on same index, then deposit half of the money.
mod = SourceModule("""
      __global__ void parallel(float* a, int curr[], int n[], float USDCAD, float USDINR, float CADUSD, float INRUSD, float INRCAD){
            int idx = (blockIdx.x * blockDim.x) +  threadIdx.x;
            if(curr[idx] == 0){
                  if(n[idx] == 1){
                        a[idx] *= USDCAD;
                  }
                  else if (n[idx] == 2){
                        a[idx] *= USDINR;
                  }   
                  else{
                        a[idx] /= 2.0;
                  }
            }
            else if (curr[idx] == 1){
                  if (n[idx] == 0){
                        a[idx] *= CADUSD;
                  }
                  else{
                        a[idx] /= 2.0;
                  }
            }
            else if (curr[idx] == 2){
                  if (n[idx] == 0){
                        a[idx] *= INRUSD;
                  }
                  else if (n[idx] == 1){
                        a[idx] *= INRCAD;
                  } 
                  else{
                        a[idx] /= 2.0;  
                  }
            }  
      }
""")

def parallelConversion() :
      
      current = np.random.randint(3, size = 10)
      current = current.round()

      new = np.random.randint(3, size = 10)
      new = new.round()
      
      currency = (np.random.rand(1,10)) * 1000
      
      print(current)
      print(new)
      print(currency)

      current = current.astype(int)
      current_gpu = cuda.mem_alloc(current.nbytes)
      cuda.memcpy_htod(current_gpu, current)

      new = new.astype(int)
      new_gpu = cuda.mem_alloc(new.nbytes)
      cuda.memcpy_htod(new_gpu, new)

      currency = currency.astype(np.float32)
      currency_gpu = cuda.mem_alloc(currency.nbytes)
      cuda.memcpy_htod(currency_gpu, currency)


      func = mod.get_function("parallel")
      func(currency_gpu, current, new, np.float32(USD_CAD), np.float32(USD_INR), np.float32(CAD_USD), np.float32(INR_USD), np.float32(INR_CAD), block = (10, 1, 1))
      #, np.float32(CAD_USD), np.float32(INR_USD), np.float32(INR_CAD),
      new_currency = np.empty_like(currency)
      cuda.memcpy_dtoh(new_currency, currency_gpu)
      print(new_currency)

  
# Driver code 
if __name__ == "__main__" : 
      import time 
      import random

      api_key = "9HLN6GDOFH69DXEK"
      # currency code 

      from_currency = "USD"
      to_currency = "INR"
      USD_INR = float(RealTimeCurrencyExchangeRate(from_currency, to_currency, api_key))

      from_currency = "INR"
      to_currency = "USD"
      INR_USD = float(RealTimeCurrencyExchangeRate(from_currency, to_currency, api_key))

      from_currency = "CAD"
      to_currency = "USD"
      CAD_USD = float(RealTimeCurrencyExchangeRate(from_currency, to_currency, api_key))

      from_currency = "USD"
      to_currency = "CAD"
      USD_CAD = float(RealTimeCurrencyExchangeRate(from_currency, to_currency, api_key))

      from_currency = "INR"
      to_currency = "CAD"
      INR_CAD = float(RealTimeCurrencyExchangeRate(from_currency, to_currency, api_key))

      startTime = time.time()
      
      parallelConversion() 
      #nonParallelConversion() 

      
      endTime = time.time()

      total = endTime - startTime

      
      print("Total time: ", total)
