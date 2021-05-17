from numpy import *
from numpy.random import rand
from pylab import pcolor, show, colorbar, xticks, yticks
from pylab import *
from mpl_toolkits.mplot3d import axes3d
from matplotlib import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math

from tqdm import tqdm
import time


"""　メモ　20201220

共焦点光学系のOTFを計算するためのスクリプト。
「Phase control and measurement in digital microscopy」 Matthew Raphael Arnison および
「A 3D vectorial optical transfer function suitable for arbitrary pupil functions」
を参考にしている。 
 
実際のOTFの計算は240行目あたりからスタート

"""

#==========================================================================================
#　Arnisonの文献　式(3.9)について
def a(m, n):
    l = ((m**2) + (n**2))**(1/2)
    if l > 1.0:
        a_1 = np.nan
        a_2 = np.nan
        a_3 = np.nan
    else:    
        s_plus = (1 - (l**2))**(1/2)
        a_1 = (((m**2)*s_plus) + n**2)/(l**2)
        a_2 = ((-1.0)*m*n*(1 - s_plus))/(l**2)
        a_3 = -1.0*m  
        
    return a_1, a_2, a_3

#　Arnisonの文献　式(3.9)について 以上。
#==========================================================================================

#==========================================================================================
# Arnisonの文献　式(3.10)について
def S(m, n, condition):
    if condition == 0:
        Apodisation = 1.0
    
    elif condition == 1:
        l = ((m**2) + (n**2))**(1/2)
        if l > 1.0:
            #Apodisation = np.nan
            Apodisation = 0.0
        else:    
            root_s_plus = ((1 - (l**2))**(1/2))**(1/2)
        Apodisation = root_s_plus
        
    return Apodisation

def T(m,n):
    T_result = 1
    
    return T_result

def P_plus(m, n):
    a_ = a(m, n)
    
    S_ = S(m, n, 1)
    
    T_ = T(m, n)

    
    P_plus_result_0 = a_[0] * S_ * T_
    P_plus_result_1 = a_[1] * S_ * T_
    P_plus_result_2 = a_[2] * S_ * T_
    
    return P_plus_result_0, P_plus_result_1, P_plus_result_2
    
# Arnisonの文献　式(3.10)について 以上。
#==========================================================================================

#==========================================================================================
# Arnisonの文献　式(3.20)について
def r0_K0B(m, n, s, Beta):
    K = ((m)**(2)+(n)**(2)+(s)**(2))**(1/2)
    r0_Buffer = (1-(((K)**2)/4))
    if r0_Buffer < 0:
        r0_0 = 0.0
        r0_1 = 0.0
        r0_2 = 0.0        
    else:
        r0 = (1-(((K)**2)/4))**(1/2)
        r0_0 = 0.0
        r0_1 = r0*(np.sin(Beta))
        r0_2 = (-1.0)*r0*(np.cos(Beta))
        
    return r0_0, r0_1, r0_2 

# Arnisonの文献　式(3.20)について 以上。
#==========================================================================================

#==========================================================================================
# Arnisonの文献　式(3.24)について
def r0_KB(m, n, s, Beta):
    K = ((m)**(2)+(n)**(2)+(s)**(2))**(1/2)
    r0_Buffer = (1-(((K)**2)/4)) #r0の計算に進むか否かの判断要素
    l = ((m**2) + (n**2))**(1/2)
    if r0_Buffer < 0:
        r0_0 = 0.0
        r0_1 = 0.0
        r0_2 = 0.0

    else:
        r0 = (1-(((K)**2)/4))**(1/2)
        r0_0 = (r0*(1/(l*K)))*(m*s*(np.cos(Beta)) - n*K*(np.sin(Beta)))
        r0_1 = (r0*(1/(l*K)))*(n*s*(np.cos(Beta)) + m*K*(np.sin(Beta)))
        r0_2 = (-1.0)*r0*(l/K)*(np.cos(Beta))
                
    return r0_0, r0_1, r0_2 

# Arnisonの文献　式(3.24)について 以上。
#==========================================================================================

#==========================================================================================
# Arnisonの文献　式(3.32)について
def Beta1_KAlpha(m, n, s, Alpha):
    K = ((m)**(2)+(n)**(2)+(s)**(2))**(1/2)
    r0_Buffer = (1-(((K)**2)/4)) #r0の計算に進むか否かの判断要素
    l = ((m**2) + (n**2))**(1/2)


    if r0_Buffer < 0:
        Beta = 0.0
    else:
        r0 = (1-(((K)**2)/4))**(1/2)
        Param0 = (K/(l*r0))*((((np.abs(s))/2)+np.cos(Alpha)))
        Param1 = (K/(l*r0))*(np.abs(((np.abs(s))/2)+np.cos(Alpha)))
        Param2 = np.real(Param0)
        
        if Param1 <= 1:
            Beta = np.arccos(Param0)
        elif Param2 > 1:
            Beta = 0.0
        elif Param2 < -1:
            Beta = np.pi
        else:
            Beta = 0
        
    
    
    return Beta

# Arnisonの文献　式(3.32)について 以上。
#==========================================================================================

#==========================================================================================
# Arnisonの文献　式(3.27)について
def N_Alpha_function(Alpha):

    cos = (np.cos(Alpha))**2
    sin = (np.sin(Alpha))**4
    N_Alpha = ((1/4)*(3+cos)*sin)**2
    
    return N_Alpha

# Arnisonの文献　式(3.27)について 以上。
#==========================================================================================

#==========================================================================================
# Arnisonの文献　式(3.28)の積分の中身について　Primitive　⇐　原始関数
def Primitive_CK(m, n, s, Beta):


    r0_KBeta = r0_KB(m, n, s, Beta)
    r0_0 = r0_KBeta[0]
    r0_1 = r0_KBeta[1]
    r0_2 = r0_KBeta[2]
    
    P_plus_ = P_plus((r0_0 + (1/2)*m), (r0_1 + (1/2)*n))
    P_plus_Ast = P_plus((r0_0 - (1/2)*m), (r0_1 - (1/2)*n))

    
    P_inner_product = (P_plus_[0] * P_plus_Ast[0]) + (P_plus_[1] * P_plus_Ast[1])
    
    
    
    return P_inner_product

# Arnisonの文献　式(3.28)の積分の中身について　以上。
#==========================================================================================













#2020/12/20================================================================================
# 式3-28の計算
#==========================================================================================

# パラメーター入力

Alpha = 1.2   #　開口の角度を定義する。
k_step = int(100) # k空間をいくつに分割するか？
BetaStep = 0.01 #　βを数値積分する際の微小のβ
N_alpha__ = N_Alpha_function(Alpha) # 式 3-27

Wave = 2.0

# パラメーター入力　以上。


#===========================================================================================

m_axis = np.array(np.linspace(-1.0*Wave, Wave, k_step)) #ｍ軸の波数空間を規定
n_axis = np.array(np.linspace(-1.0*Wave, Wave, k_step)) #ｎ軸の波数空間を規定
s_axis = np.array(np.linspace(-1.0*Wave, Wave, k_step)) #ｓ軸の波数空間を規定
CK_space = np.zeros((int(m_axis.shape[0]), int(n_axis.shape[0]), int(n_axis.shape[0])), dtype = float)  #式3-28の左辺の空間を規定



for m_index in tqdm(range(m_axis.shape[0])):
    #print(m_index, 'of', m_axis.shape[0])
    for n_index in range(m_axis.shape[0]):
        for s_index in range(m_axis.shape[0]):
            m = m_axis[m_index]
            n = n_axis[n_index]
            s = s_axis[s_index]

            Normalize = (((m)**(2)+(n)**(2)+(s)**(2))**(1/2)) * N_alpha__   #　規格化定数を計算
                        
            Beta1 = Beta1_KAlpha(m, n, s, Alpha) # Beta　の最大角
            Beta1_minus = -1.0 * Beta1           # Beta　の最小角
            
            BetaVector = np.linspace(Beta1_minus, Beta1, int(2*Beta1/BetaStep))
            Integral_Buffer1 = np.zeros((int(2*Beta1/BetaStep), 1), dtype = float)
            
            if Beta1 == 0:
                CK_space[m_index, n_index, s_index] = 0
            else:                    
                for Beta_index in range(int(2*Beta1/BetaStep)):
                    #print(Beta_index, ' of ', int(2*Beta1/BetaStep))
                    Beta = BetaVector[Beta_index] 
                    Primitive_CK_ = Primitive_CK(m, n, s, Beta)
                    Integral_Buffer1[Beta_index, 0] = Primitive_CK_
                Sum_Integral_Buffer1 = np.sum(Integral_Buffer1)
                
                CK_space[m_index, n_index, s_index] = Sum_Integral_Buffer1 / (Normalize)
                
            #print(CK_space[m_index,n_index,s_index])
            




plt.figure()
plt.imshow(CK_space[:,:, int(k_step/2)])
plt.colorbar ()
plt.title('OTF s= 0')
plt.savefig('s=0_Linear.png',format = 'png', dpi=500)

plt.figure()
plt.imshow(CK_space[:,int(k_step/2),:])
plt.colorbar () 
plt.title('OTF n= 0')
plt.savefig('n=0_Linear.png',format = 'png', dpi=500)

plt.figure()
plt.imshow(CK_space[int(k_step/2),:,:])
plt.colorbar ()
plt.title('OTF m= 0') 
plt.savefig('m=0_Linear.png',format = 'png', dpi=500)

plt.figure()
plt.imshow(np.log10(CK_space[:,:, int(k_step/2)]))
plt.colorbar ()
plt.title('OTF Log10( s= 0 )') 
plt.savefig('s=0_log.png',format = 'png', dpi=500)

plt.figure()
plt.imshow(np.log10(CK_space[:,int(k_step/2),:]))
plt.colorbar ()
plt.title('OTF Log10( n= 0 )') 
plt.savefig('n=0_log.png',format = 'png', dpi=500)

plt.figure()
plt.imshow(np.log10(CK_space[int(k_step/2),:,:]))
plt.colorbar ()
plt.title('OTF Log10( n= 0 )') 
plt.savefig('m=0_log.png',format = 'png', dpi=500)

plt.show()




#==========================================================================================
# 式3-28の計算 以上。
#==========================================================================================


















