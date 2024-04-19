import pyaudio
import wave

# 设置录音参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

audio = pyaudio.PyAudio()

# 打开音频流
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

print("Recording...")

frames = []

# 录制音频
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording finished.")

# 关闭音频流
stream.stop_stream()
stream.close()
audio.terminate()

# 保存录制的音频为WAV文件
with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))