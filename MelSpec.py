import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 生成一个1秒的音频信号
duration = 1.0  # 秒
sr = 44100  # 采样率
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
audio_signal = 0.5 * np.sin(2 * np.pi * 220 * t)  # 220 Hz 正弦波

# 计算梅尔谱
n_fft = 2048  # FFT大小
hop_length = 512  # 帧移大小
n_mels = 40  # 梅尔滤波器数

# 使用librosa计算梅尔谱
mel_spectrogram = librosa.feature.melspectrogram(y=audio_signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

# 转换为对数梅尔谱
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

# 可视化梅尔滤波器组
mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_filter_bank, sr=sr, x_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Filter Bank')
plt.tight_layout()
plt.show()

# 可视化梅尔谱
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_mel_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', fmax=sr/2)
plt.colorbar(format='%+2.0f dB')
plt.title('Log-Mel Spectrogram')
plt.tight_layout()
plt.show()