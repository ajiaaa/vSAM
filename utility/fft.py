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

data = []
for i in range(len(lines)):
    if i > 1150 and i < 1200:
        data.append(float(lines[i].strip()))
f.close()

# plt.hist(data)
data.sort()



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

'''
t = np.linspace(0,100,N)
l1,l2 = fft_tran(t,sample_data)
plt.plot(l1[l1 > 0], l2[l1 > 0]/2,'-',lw=2)
plt.show()
plt.savefig('figure.png', bbox_inches='tight', dpi=256)
'''


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
import numpy as np
from scipy.fftpack import fft,fftshift
import matplotlib.pyplot as plt


N = 1024                        # 采样点数
sample_freq=120                 # 采样频率 120 Hz, 大于两倍的最高频率
sample_interval=1/sample_freq   # 采样间隔
signal_len=N*sample_interval    # 信号长度
t=np.arange(0,signal_len,sample_interval)

signal = 5 + 2 * np.sin(2 * np.pi * 20 * t) + 3 * np.sin(2 * np.pi * 30 * t) + 4 * np.sin(2 * np.pi * 40 * t)  # 采集的信号

fft_data = fft(signal)

# 在python的计算方式中，fft结果的直接取模和真实信号的幅值不一样。
# 对于非直流量的频率，直接取模幅值会扩大N/2倍， 所以需要除了N乘以2。
# 对于直流量的频率(0Hz)，直接取模幅值会扩大N倍，所以需要除了N。
fft_amp0 = np.array(np.abs(fft_data)/N*2)   # 用于计算双边谱
fft_amp0[0]=0.5*fft_amp0[0]
N_2 = int(N/2)
fft_amp1 = fft_amp0[0:N_2]  # 单边谱
fft_amp0_shift = fftshift(fft_amp0)    # 使用fftshift将信号的零频移动到中间

# 计算频谱的频率轴
list0 = np.array(range(0, N))
list1 = np.array(range(0, int(N/2)))
list0_shift = np.array(range(0, N))
freq0 = sample_freq*list0/N        # 双边谱的频率轴
freq1 = sample_freq*list1/N        # 单边谱的频率轴
freq0_shift=sample_freq*list0_shift/N-sample_freq/2  # 零频移动后的频率轴

# 绘制结果
plt.figure()
# 原信号
plt.subplot(221)
plt.plot(t, signal)
plt.title(' Original signal')
plt.xlabel('t (s)')
plt.ylabel(' Amplitude ')
# 双边谱
plt.subplot(222)
plt.plot(freq0, fft_amp0)
plt.title(' spectrum two-sided')
plt.ylim(0, 6)
plt.xlabel('frequency  (Hz)')
plt.ylabel(' Amplitude ')
# 单边谱
plt.subplot(223)
plt.plot(freq1, fft_amp1)
plt.title(' spectrum single-sided')
plt.ylim(0, 6)
plt.xlabel('frequency  (Hz)')
plt.ylabel(' Amplitude ')
# 移动零频后的双边谱
plt.subplot(224)
plt.plot(freq0_shift, fft_amp0_shift)
plt.title(' spectrum two-sided shifted')
plt.xlabel('frequency  (Hz)')
plt.ylabel(' Amplitude ')
plt.ylim(0, 6)

plt.show()



'''

'''
#number of sample points

#sample spacing
T=10
x=np.linspace(0,T,N)
#DFT
fft_y=np.abs(fft.fft(f(x)))
fft_y/=max(fft_y)
#frequency list
fft_x=fft.fftfreq(N,T/N)

fig,ax=plt.subplots(2,1)
plt.subplot(211)
plt.grid(ls='--')
plt.plot(x,f(x))
plt.xlabel('x')
plt.ylabel('f(x)')

plt.subplot(212)
#plt.plot(fft_x,fft_y) is also ok here
plt.plot(fft.fftshift(fft_x),fft.fftshift(fft_y))
plt.grid(ls='--')
plt.xlabel(r'frequency/$2\pi$')
plt.ylabel('relative intensity')
plt.show()

'''