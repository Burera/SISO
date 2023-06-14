# -- coding: utf-8 --
"""
Created on Fri Apr 21 13:50:40 2023

@author: USER
"""

# -- coding: utf-8 --
"""
Created on Fri Apr 21 11:06:59 2023

@author: USER
"""

import random
import numpy as np
import pdb
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from math import *

#M=2
num_of_rx_atn = 2
M = 4
k = int(np.log2(M))
Nt = 2
Nr = 2
SER_arr =[]
BER_arr = []
real_arr = []
imag_arr = []
symbol_tx_arr = []
symbol_rx_arr = []

for Eb_N0_dB in range (0,50):
    print('******************************')
    print('Eb_N0_dB : ', Eb_N0_dB)
    Eb_N0_ratio = 10**(Eb_N0_dB/10)
    Es_N0_ratio = Eb_N0_ratio*log2(M)
    sigma = sqrt(1/(2*Es_N0_ratio)) #Standard Deviation of Gaussian distribution
    Eb_N0_dB += 1
    error = 0
    num_of_sym = 0
    num_of_bit = 0

    while(error < 80):
      # Generate random binary data
      data_en = np.random.randint(2, size=(Nt, k)) 
      #print(data_en)      
      # Map binary data to QPSK symbols
      s = np.zeros((Nt,1), dtype=np.complex64)
     
     
      for i in range(Nt):
          # Convert binary to decimal
          dec = data_en[i, 0]*2 + data_en[i, 1]
          #print(dec)
          # Map decimal to QPSK symbol using specified mapping
          if dec == 0:
              symbol_real = cos(pi/4)
              symbol_imag = sin(pi/4)
          elif dec == 1:
              symbol_real = cos(3*pi/4)
              symbol_imag = sin(3*pi/4)
          elif dec == 3:
              symbol_real = cos(5*pi/4)
              symbol_imag = sin(5*pi/4)
          elif dec == 2:
              symbol_real = cos(7*pi/4)
              symbol_imag = sin(7*pi/4)
             
          s[i] = symbol_real + 1j*symbol_imag
     
     
      #print(s ,  "transmitted signal")
    #-----------------------------------------------------------------------#
    
    # Generate channel matrix ----------------------------------------------#
      H = (np.random.randn(Nr, Nt) + 1j*np.random.randn(Nr, Nt))*np.sqrt(0.5)
      H_angle = np.angle(H) + 2 * np.pi # channel phase in radian
      phase = np.array([np.sum(H_angle[:, 0]), np.sum(H_angle[:, 1])]) # sum channel phase in col
      q = np.cos(phase) + 1j * np.sin(-phase)
      # Convert q into a 1x1 array (row)
      q_row = np.array([q])
    
    # Transpose q_row
      q_transpose = q_row.T
      Zp = s * q_transpose
 
      #print(q_transpose , "q")
      #print(Zp , "Zp")
    #-----------------------------------------------------------------------#
    # Generate AWGN noise
      n = ((np.random.randn(1, Nr) + 1j * np.random.randn(1, Nr)) * sigma).T

      #print(n , "noise")
    #-----------------------------------------------------------------------#
    # Received signal
      y = np.dot(H, Zp) + n   #without noise
      #print(y , "Y")
    #-----------------------------------------------------------------------# 
    
    # Construct a received channel matrix
      Hp = np.array([[H[0,0]*q[0], H[0,1]*q[1]],[H[1,0]*q[0],H[1,1]*q[1]]])
    #print(Hp)
    # Detected signal ------------------------------------------------------#
    #F = np.linalg.inv(Hp)
    #h_inv_y = h_inv * y
      F = np.dot(np.linalg.inv(np.dot(Hp.T, Hp)), Hp.T) # ZF detection
      x = np.dot(F, y)
      #print(x , "detected signal")
    
      #Demodulation
      symbol_rx_1 = x[0]
      symbol_rx_2 = x[1]
     
    # initialize an empty list to store the symbols
      symbols = []
     
      # determine the symbols and append them to the list
      if symbol_rx_1.imag >= 0:
          symbols.append(0)
      else:
          symbols.append(1)
     
      if symbol_rx_1.real >= 0:
          symbols.append(0)
      else:
          symbols.append(1)
     
      if symbol_rx_2.imag >= 0:
          symbols.append(0)
      else:
          symbols.append(1)
     
      if symbol_rx_2.real >= 0:
          symbols.append(0)
      else:
          symbols.append(1)
     
      # convert the list into a 2D array using numpy
      symbol_array = np.array(symbols).reshape(2,2)
      #print(symbol_array)
     

     
       
           
      num_of_bit += 2*log2(M)
    
      for i in range(Nt):
        tx_bits = data_en[i]
        rx_bits = symbol_array[i]
        for j in range(k):
            if tx_bits[j] != rx_bits[j]:
             error += 1
            
             #print(error)
     
            
     
            
    BER = error / num_of_bit
    print("Bit error rate:", BER)
     
    BER_arr.append(BER)
    
    print(BER_arr , "Berawwa")