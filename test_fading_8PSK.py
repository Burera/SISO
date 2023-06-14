import random
import numpy as np
import pdb
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from math import *

M=8
Eb_N0_dB = 0
# threshold = 0
num_of_sym = 0
BER_arr = []
SER_arr = []
symbol_cx = complex(0,0)
symbol_real = symbol_cx.real
symbol_imag = symbol_cx.imag
# real_arr = []
# imag_arr = []
symbol_tx_arr = []
symbol_rx_arr = []

for Eb_N0_dB in range (0,31):
    print('******************************************************************************')
    print('Eb_N0_dB : ', Eb_N0_dB)
    # Eb_N0_ratio = 10**(Eb_N0_dB*2/10)
    Eb_N0_ratio = 10**(Eb_N0_dB/10)
    Es_N0_ratio = Eb_N0_ratio * log2(M)
    sigma = sqrt(1/(2*Es_N0_ratio)) #Standard Deviation of Gaussian distribution
    # Eb_N0_dB += 1
    error = 0
    num_of_sym = 0
    
    while(error<100):
#1. generating signal (symbol) 000 001 011 010 110 111 101 100
        for i in range (0,3):
            symbol_tx = random.randrange(2) # 0, 1
            symbol_tx_arr.insert(i,symbol_tx)
                
                
#2. modulation 000 001 011 010 110 111 101 100
        #000
        if (symbol_tx_arr[0] == 0 and symbol_tx_arr[1] == 0 and symbol_tx_arr[2] == 0):
            symbol_real = cos(pi*1/M)
            symbol_imag = sin(pi*1/M)
        #001
        elif(symbol_tx_arr[0] == 0 and symbol_tx_arr[1] == 0 and symbol_tx_arr[2] == 1):
            symbol_real = cos(pi*3/M)
            symbol_imag = sin(pi*3/M)
        #011
        elif(symbol_tx_arr[0] == 0 and symbol_tx_arr[1] == 1 and symbol_tx_arr[2] == 1):
            symbol_real = cos(pi*5/M)
            symbol_imag = sin(pi*5/M)
        #010
        elif(symbol_tx_arr[0] == 0 and symbol_tx_arr[1] == 1 and symbol_tx_arr[2] == 0):
            symbol_real = cos(pi*7/M)
            symbol_imag = sin(pi*7/M)
        #110
        elif(symbol_tx_arr[0] == 1 and symbol_tx_arr[1] == 1 and symbol_tx_arr[2] == 0):
            symbol_real = cos(pi*9/M)
            symbol_imag = sin(pi*9/M)
        #111
        elif(symbol_tx_arr[0] == 1 and symbol_tx_arr[1] == 1 and symbol_tx_arr[2] == 1):
            symbol_real = cos(pi*11/M)
            symbol_imag = sin(pi*11/M)
        #101
        elif(symbol_tx_arr[0] == 1 and symbol_tx_arr[1] == 0 and symbol_tx_arr[2] == 1):
            symbol_real = cos(pi*13/M)
            symbol_imag = sin(pi*13/M)
        #100
        elif(symbol_tx_arr[0] == 1 and symbol_tx_arr[1] == 0 and symbol_tx_arr[2] == 0):
            symbol_real = cos(pi*15/M)
            symbol_imag = sin(pi*15/M)
        symbol = complex(symbol_real, symbol_imag)
        
        AWGN_r = np.random.normal(0, sigma)
        AWGN = complex(AWGN_r, AWGN_r)
            
        # rayleigh_fading
        sigma_r = sqrt(0.5)
        x = np.random.normal(0, sigma_r)
        y = np.random.normal(0, sigma_r)
        # x = np.random.rayleigh()
        # y = np.random.rayleigh()
        h = complex(x, y)
        
        ## y = x*h + n
        x_h = symbol*h + AWGN  # yy = complex(symbol_real*x - symbol_imag*y + AWGN_r, symbol_real*y + symbol_imag*x + AWGN_i)

        ## y * h* = x * h * h* + n * h* = h* *(x*h + n)
        h_conj = h.conjugate()
        y_h_conj = x_h*h_conj  
        # y_h_conj = x_h.conjugate()*h_conj
        
        ## (y * h*)/\\h\\^2
        abs_square = pow(abs(h), 2)
        detection = y_h_conj/abs_square
        
        # real_arr.append(detection.real)
        # imag_arr.append(detection.imag)
        
#4. phase = atan(imag_tx/real_tx)
        phase = atan2(detection.imag, detection.real)
        
#5. demodulation 
        #000    
        if (phase > 0 and phase < pi/4):
            symbol_rx_arr.insert(0,0)
            symbol_rx_arr.insert(1,0)
            symbol_rx_arr.insert(2,0)
            #001
        elif(phase > pi/4 and phase < pi/2):
            symbol_rx_arr.insert(0,0)
            symbol_rx_arr.insert(1,0)
            symbol_rx_arr.insert(2,1)
            #011
        elif(phase > pi/2 and phase < pi*3/4):
            symbol_rx_arr.insert(0,0)
            symbol_rx_arr.insert(1,1)
            symbol_rx_arr.insert(2,1)
            #010
        elif(phase > pi*3/4 and phase < pi):
            symbol_rx_arr.insert(0,0)
            symbol_rx_arr.insert(1,1)
            symbol_rx_arr.insert(2,0)
            #110
        elif(phase > -pi and phase < -pi*3/4):
            symbol_rx_arr.insert(0,1)
            symbol_rx_arr.insert(1,1)
            symbol_rx_arr.insert(2,0)
            #111
        elif(phase > -pi*3/4 and phase < -pi/2):
            symbol_rx_arr.insert(0,1)
            symbol_rx_arr.insert(1,1)
            symbol_rx_arr.insert(2,1)
            #101
        elif(phase > -pi/2 and phase < -pi/4):
            symbol_rx_arr.insert(0,1)
            symbol_rx_arr.insert(1,0)
            symbol_rx_arr.insert(2,1)
            #100
        elif(phase > -pi/4 and phase < 0):
            symbol_rx_arr.insert(0,1)
            symbol_rx_arr.insert(1,0)
            symbol_rx_arr.insert(2,0)
            
    #6. information sinc
        for j in range (0,3):
            if (symbol_rx_arr[j] != symbol_tx_arr[j]) :
                error += 1
                
                
        num_of_sym += 1
        symbol_rx_arr=[] 
        symbol_tx_arr = []
        
        
    SER = error/num_of_sym
    SER_arr.append(SER)
    BER = SER/log2(M) #M=8
    BER_arr.append(BER)
    print(SER_arr)
    print(BER_arr)

# plt.subplot(2,1,2)
plt.plot(BER_arr, 'bo-')
plt.xlim([0, 30])
plt.yscale('log')
plt.xlabel('Eb/N0[dB]')
plt.ylabel('BER')
plt.title('8PSK Simulation_complex')
plt.grid(True)
plt.show()

