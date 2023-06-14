import random
import numpy as np
import pdb
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from math import *


M=2
Eb_N0_dB = 0
threshold = 0
num_of_sym = 0
BER_arr = []
symbol_cx = complex(0,0)
symbol_real = symbol_cx.real
symbol_imag = symbol_cx.imag
real_arr = []
imag_arr = []
symbol_tx_arr = []
symbol_rx_arr = []




for Eb_N0_dB in range (0,28):
    print('===============================================================================================')
    print('Eb_N0_dB : ', Eb_N0_dB)
    Eb_N0_ratio = 10**(Eb_N0_dB/10)
    sigma = sqrt(1/(2*Eb_N0_ratio)) #Standard Deviation of Gaussian distribution
    # Eb_N0_dB += 1
    error = 0
    num_of_sym = 0
   
    while(error < 100):
#1. x = 0 or not 1 create [0] [1]
        # 1. generating singal(symbol)
        # for i in range(0,1):
        symbol_tx = random.randrange(2) #0,1
        symbol_tx_arr.insert(0, symbol_tx)
            
#2. modulation 0 -> -1, 1 -> 1
        if (symbol_tx_arr[0] == 1):
            # symbol = cos(0) + sin(0)*1j  #1-> 1 + 0j
            symbol_real = cos(0)
            symbol_imag = sin(0)
        elif(symbol_tx_arr[0] == 0):
            # symbol = cos(pi) + sin(pi)*1j #0 -> -1 + 0j
            symbol_real = cos(pi)
            symbol_imag = 0
        # print("modulated symbol : ", symbol_tx)
        symbol = complex(symbol_real, symbol_imag)
        
        
#3. Multiply Rayleigh Channels and Add Noise
        # 3. generating noise in complex form
        AWGN_r = np.random.normal(0, sigma) #random value of a normal distribution that mean = 0, std = sigma
        # AWGN_i = np.random.normal(0, sigma)
        # symbol_real_wn = symbol_real + AWGN_r
        # symbol_imag_wn = symbol_imag + AWGN_i
        AWGN = complex(AWGN_r, AWGN_r)
        
        # rayleigh_fading
        sigma_r = sqrt(0.5)
        x = np.random.normal(0, sigma_r)
        y = np.random.normal(0, sigma_r)
        # x = np.random.rayleigh()
        # y = np.random.rayleigh()
        h = complex(x, y)
        # ray = sqrt(pow(x,2)+pow(y,2))
        # h = complex(ray, ray)

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
        
#5. demodulation 페이즈가 어디에 해당하는지에 따라서, 오른쪽 평면 -> 1 왼쪽 평면 -> 0

        if(detection.real > 0):
            symbol_rx_arr.insert(0,1)
        elif(detection.real < 0):
            symbol_rx_arr.insert(0,0)
        # if (phase >= -pi/2 and phase <= pi/2 and real_tx > 0) :
        # if (phase > -pi/2 and phase < pi/2):
        #     symbol_rx_arr.insert(0,1)
        # elif((phase > pi/2 and phase <= pi)or(phase >= -pi and phase < -pi/2)) :
        #     symbol_rx_arr.insert(0,0)
        # print('received symbol :', symbol_r)
        
#6. information sinc 처음 만들어낸 신호 != demodulation 신호 -> error += 1
        if (symbol_rx_arr[0] != symbol_tx_arr[0]):
            error += 1
        # pdb.set_trace()
        num_of_sym += 1
        
        # symbol_rx_arr=[]
        # symbol_tx_arr=[]
        

    SER = error/num_of_sym   
    BER = SER/log2(M)
    BER_arr.append(BER)
    print(BER_arr)
    for j in BER_arr:
        data = "%lf \n" %BER
    f.write(data)
    

f.close()