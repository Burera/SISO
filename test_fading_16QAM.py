import random
import numpy as np
import pdb
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from math import *

M=16
A= 1/sqrt(10)
Eb_N0_dB = 0
num_of_sym = 0
BER_arr = []
SER_arr = []
symbol_cx = complex(0,0)
symbol_real = symbol_cx.real
symbol_imag = symbol_cx.imag
symbol_tx_arr = []
symbol_rx_arr = []


for Eb_N0_dB in range (0,31):
    print('**************************************************************************************')
    print('Eb_N0_dB : ', Eb_N0_dB)
    # Eb_N0_ratio = 10**(Eb_N0_dB*2/10)
    Eb_N0_ratio = 10**(Eb_N0_dB/10)
    Es_N0_ratio = Eb_N0_ratio * log2(M)
    sigma = sqrt(1/(2*Es_N0_ratio)) #Standard Deviation of Gaussian distribution
    # Eb_N0_dB += 1
    error = 0
    num_of_sym = 0
    
    while(error<100):
        #1. generating signal (symbol) 0000 0001 0011 0010 0110 0111 0101 0100 1100 1101 1111 1110 1010 1011 1001 1000
        for i in range (0,4):
            symbol_tx = random.randrange(2) # 0, 1
            symbol_tx_arr.insert(i,symbol_tx)
            
         #2. modulation 0000 0001 0011 0010 0110 0111 0101 0100 1100 1101 1111 1110 1010 1011 1001 1000

                                    
        #0000 -3 3 
        #0001 -3 1             
        if (symbol_tx_arr[0] == 0 and symbol_tx_arr[1] == 0 and symbol_tx_arr[2] == 0):
            if(symbol_tx_arr[3]==0):
                symbol_real = -3*A
                symbol_imag = 3*A
            elif(symbol_tx_arr[3]==1):
                symbol_real = -3*A
                symbol_imag = A
                
        #0011 -3 -1
        #0010 -3 -3
        if (symbol_tx_arr[0] == 0 and symbol_tx_arr[1] == 0 and symbol_tx_arr[2] == 1):
            if(symbol_tx_arr[3]==1):
                symbol_real = -3*A
                symbol_imag = -A 
            elif(symbol_tx_arr[3]==0):
                symbol_real = -3*A
                symbol_imag = -3*A 

        #0110 -1 -3
        #0111 -1 -1
        if (symbol_tx_arr[0] == 0 and symbol_tx_arr[1] == 1 and symbol_tx_arr[2] == 1):
            if(symbol_tx_arr[3]==0):
                symbol_real = -A
                symbol_imag = -3*A 
            elif(symbol_tx_arr[3]==1):
                symbol_real = -A
                symbol_imag = -A 
        #0101 -1 1  
        #0100 -1 3              
        if (symbol_tx_arr[0] == 0 and symbol_tx_arr[1] == 1 and symbol_tx_arr[2] == 0):
            if(symbol_tx_arr[3]==1):
                symbol_real = -A
                symbol_imag = A 
            elif(symbol_tx_arr[3]==0):
                symbol_real = -A
                symbol_imag = 3*A 


        #1100 1 3
        #1101 1 1
        if (symbol_tx_arr[0] == 1 and symbol_tx_arr[1] == 1 and symbol_tx_arr[2] == 0):
            if(symbol_tx_arr[3]==0):
                symbol_real = A
                symbol_imag = 3*A 
            elif(symbol_tx_arr[3]==1):
                symbol_real = A
                symbol_imag = A 

        #1111 1 -1
        #1110 1 -3
        if (symbol_tx_arr[0] == 1 and symbol_tx_arr[1] == 1 and symbol_tx_arr[2] == 1):
            if(symbol_tx_arr[3]==1):
                symbol_real = A
                symbol_imag = -A 
            elif(symbol_tx_arr[3]==0):
                symbol_real = A
                symbol_imag = -3*A 
                

        #1010 3 -3
        #1011 3 -1
        if (symbol_tx_arr[0] == 1 and symbol_tx_arr[1] == 0 and symbol_tx_arr[2] == 1):
            if(symbol_tx_arr[3]==0):
                symbol_real = 3*A
                symbol_imag = -3*A
            elif(symbol_tx_arr[3]==1):
                symbol_real = 3*A
                symbol_imag = -A 
                
        #1001 3 1
        #1000 3 3
        if (symbol_tx_arr[0] == 1 and symbol_tx_arr[1] == 0 and symbol_tx_arr[2] == 0 ):
            if(symbol_tx_arr[3]==1):
                symbol_real = 3*A
                symbol_imag = A 
            elif(symbol_tx_arr[3]==0):
                symbol_real = 3*A
                symbol_imag = 3*A
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
        #Quadrant 1 1100 1101 1000 1001
        if (phase > 0 and phase < pi/2 and detection.real < 2*A):
                    symbol_rx_arr.insert(0,1)
                    symbol_rx_arr.insert(1,1)
                    symbol_rx_arr.insert(2,0)
                    if(detection.imag > 2*A ):
                        symbol_rx_arr.insert(3,0)
                    elif(detection.imag < 2*A):
                        symbol_rx_arr.insert(3,1)
                        
        elif(phase > 0 and phase < pi/2 and detection.real > 2*A):
                    symbol_rx_arr.insert(0,1)
                    symbol_rx_arr.insert(1,0)
                    symbol_rx_arr.insert(2,0)
                    if(detection.imag > 2*A ):
                        symbol_rx_arr.insert(3,0)
                    elif(detection.imag < 2*A):
                        symbol_rx_arr.insert(3,1)
                        
        #Quadrant 2 0000 0001 0101 0100
        elif(phase > pi/2 and phase < pi and detection.real > -2*A ):
                    symbol_rx_arr.insert(0,0)
                    symbol_rx_arr.insert(1,1)
                    symbol_rx_arr.insert(2,0)
                    if(detection.imag > 2*A ):
                        symbol_rx_arr.insert(3,0)
                    elif(detection.imag < 2*A):
                        symbol_rx_arr.insert(3,1)
                        
        elif(phase > pi/2 and phase < pi and detection.real < -2*A):
                    symbol_rx_arr.insert(0,0)
                    symbol_rx_arr.insert(1,0)
                    symbol_rx_arr.insert(2,0)
                    if(detection.imag > 2*A ):
                        symbol_rx_arr.insert(3,0)
                    elif(detection.imag < 2*A):
                        symbol_rx_arr.insert(3,1)
                        
        #Quadrant 3 0011 0010 0110 0111
        elif(phase > -pi and phase < -pi/2 and detection.real > -2*A):
                    symbol_rx_arr.insert(0,0)
                    symbol_rx_arr.insert(1,1)
                    symbol_rx_arr.insert(2,1)
                    if(detection.imag < -2*A):
                        symbol_rx_arr.insert(3,0)
                    elif(detection.imag > -2*A):
                        symbol_rx_arr.insert(3,1)
            
        elif(phase > -pi and phase < -pi/2 and detection.real < -2*A):
                    symbol_rx_arr.insert(0,0)
                    symbol_rx_arr.insert(1,0)
                    symbol_rx_arr.insert(2,1)
                    if( detection.imag < -2*A):
                        symbol_rx_arr.insert(3,0)
                    elif(detection.imag > -2*A):
                        symbol_rx_arr.insert(3,1)
                        
        #Quadrant 4 1111 1110 1010 1011
        elif(phase > -pi/2 and phase < 0 and detection.real < 2*A):
                    symbol_rx_arr.insert(0,1)
                    symbol_rx_arr.insert(1,1)
                    symbol_rx_arr.insert(2,1)
                    if( detection.imag < -2*A):
                        symbol_rx_arr.insert(3,0)
                    elif(detection.imag > -2*A ):
                        symbol_rx_arr.insert(3,1)
                        
        elif(phase > -pi/2 and phase < 0 and detection.real > 2*A ):
                    symbol_rx_arr.insert(0,1)
                    symbol_rx_arr.insert(1,0)
                    symbol_rx_arr.insert(2,1)
                    if(detection.imag < -2*A):
                        symbol_rx_arr.insert(3,0)
                    elif(detection.imag > -2*A):
                        symbol_rx_arr.insert(3,1)
        # pdb.set_trace()
        
        
        #6. information sinc
        for j in range (0,4):
            if (symbol_rx_arr[j] != symbol_tx_arr[j]) :
                error += 1
                
        # pdb.set_trace()
                
        num_of_sym += 1
        symbol_rx_arr=[] 
        symbol_tx_arr = []
        
    SER = error/num_of_sym
    SER_arr.append(SER)
    BER = SER/log2(M) #M=16
    BER_arr.append(BER)
    print(SER_arr)
    print(BER_arr)
    
# plt.subplot(2,1,2)
plt.plot(BER_arr, 'bo-')
plt.xlim([0, 30])
plt.yscale('log')
plt.xlabel('Eb/N0[dB]')
plt.ylabel('BER')
plt.title('16QAM Simulation_complex')
plt.grid(True)
plt.show()

