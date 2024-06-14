import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftshift
import numpy.fft as nf

import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import copy



def f(x):
    return np.cos(10*x)+np.sin(3*x)*np.cos(3*x)

def fft_tran(T,sr):
    complex_ary = nf.fft(sr)
    y_ = nf.ifft(complex_ary).real
    fft_freq = nf.fftfreq(y_.size, T[1] - T[0])
    fft_pow = np.abs(complex_ary)  # 复数的摸-Y轴
    return fft_freq, fft_pow


filename = r'E:\project\project2023\samm\example\c100_\1\cha\last_sg_sg_chu_norm_norm.txt'#lastog_og_norm



f = open(filename, 'r')
lines = f.readlines()

sv = []
a_var_ = []
a_v = []
data = []
for i in range(len(lines)):
    if i < 78001 and i % 8 == 0:
        data.append(float(lines[i].strip()))

np_data = np.array(data)

flag = 120
for i in range(len(np_data)-flag):


        sort_data = np.sort(np_data[i:i+flag]) #  np_data[i:i+flag]
        print(sort_data)

        vars_ = []
        for j in range(3):
            vars_.append(np.var(sort_data[j*int(flag/3) : (j+1)*int(flag/3)]))
        var_ = np.var(np_data[i:i+flag])

        print('vars_', vars_)
        print('var_', var_)
        print('sum(vars_)/var_', sum(vars_)/len(vars_)/var_)

        a_var_.append(var_)
        a_v.append(sum(vars_)/len(vars_))
        sv.append(sort_data[-1]- sort_data[0])# append(sum(vars_)/len(vars_)/var_)np.mean(np_data[i:i+flag])



sd_= range(len(sv))


fig,ax=plt.subplots(3,1)
plt.subplot(311)
plt.plot(sd_, sv, label='sv')
plt.subplot(312)
plt.plot(sd_, a_var_, label='var_')
plt.subplot(313)
plt.plot(sd_, a_v, label='m_v')
plt.show()







'''


sd_= range(len(data))
plt.plot(sd_, data)

plt.show()

sample_freq=1           # 采样频率
sample_interval=1/sample_freq   # 采样间隔

sample_data = []
sample_data_sum = 0
for i in range(len(data)):
    if i % sample_interval == 0:
        sample_data.append(data[i])
        sample_data_sum += data[i]

N=len(sample_data)

sample_data_mean = sample_data_sum / len(sample_data)

for i in range(len(sample_data)):
    sample_data[i] = sample_data[i] - sample_data_mean



fft_data=fft(sample_data)# np.log()
print('fft_y: ', fft_data/N)


# 在python的计算方式中，fft结果的直接取模和真实信号的幅值不一样。
# 对于非直流量的频率，直接取模幅值会扩大N/2倍， 所以需要除了N乘以2。
# 对于直流量的频率(0Hz)，直接取模幅值会扩大N倍，所以需要除了N。
fft_amp0 = np.array(np.abs(fft_data)/N*2)   # 用于计算双边谱
fft_amp0[0]=0.5*fft_amp0[0]
N_2 = int(N/2)
fft_amp1 = fft_amp0[0:N_2]  # 单边谱

ii = range(0, int(N))


list1 = np.array(range(0, int(N/2)))

freq1 = sample_freq*list1/N        # 单边谱的频率轴

fig,ax=plt.subplots(2,1)
plt.subplot(211)
plt.grid(ls='--')
#plt.plot(x,data)
#plt.xlabel('x')
#plt.ylabel('f(x)')
print(freq1)
print(fft_amp1)
plt.plot(ii, fft_amp0)
plt.ylim(0, 0.01)
plt.xlabel('frequency')
plt.ylabel('Amplitude')

plt.subplot(212)
#plt.plot(fft_x,fft_y) is also ok here
plt.plot(ii, sample_data)
plt.grid(ls='--')
plt.xlabel(r'batch')
plt.ylabel('sam_change_norm')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)


plt.show()

'''


