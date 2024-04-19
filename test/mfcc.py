import pyaudio
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 配置参数
FORMAT = pyaudio.paInt16  # 采样位数
CHANNELS = 1  # 声道数
RATE = 44100  # 采样率
CHUNK = 1024  # 缓冲区大小
N_MFCC = 13  # MFCC系数数量

# 初始化PyAudio
p = pyaudio.PyAudio()

# 打开音频流
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# 实时处理音频
while True:
    data = stream.read(CHUNK)
    samples = np.frombuffer(data, dtype=np.int16)

    # 提取MFCCs
    mfccs = librosa.feature.mfcc(y=samples.astype(float), sr=RATE, n_mfcc=N_MFCC)

    # 显示MFCCs
    # plt.figure()
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

# 停止音频流和PyAudio
stream.stop_stream()
stream.close()
p.terminate()


import numpy as np
import matplotlib.pyplot as plt
import math
import time

 
fig=plt.figure()
ax=fig.add_subplot(1,1,1)

ax.set_xlabel('Time')
ax.set_ylabel('cos(t)')
ax.set_title('')

line = None
plt.grid(True) #添加网格
plt.ion()  #interactive mode on
obsX = []
obsY = []

t0 = time.time()

while True:
    t = time.time()-t0
    obsX.append(t)
    obsY.append(math.cos(2*math.pi*1*t))

    if line is None:
        line = ax.plot(obsX,obsY,'-g',marker='*')[0]

    line.set_xdata(obsX)
    line.set_ydata(obsY)

    ax.set_xlim([t-10,t+1])
    ax.set_ylim([-1,1])

    plt.pause(0.01)

