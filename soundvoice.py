import os
import sounddevice
from scipy.io.wavfile import write
from scipy.fft import fft, fftfreq
import time
import numpy as np
import matplotlib.pyplot as plt
from my_func import *

fs = 44100
output_directory = "/Users/User/Desktop/soundras/inputsound/"  # เปลี่ยนเส้นทางไปยังเส้นทางที่คุณต้องการ
duration_per_recording = 5  # ระยะเวลาการบันทึกในแต่ละครั้ง (วินาที)
recording_number = 1  # เริ่มต้นที่ 1
delay_between_recordings = 10  # หน่วงเวลาระหว่างการบันทึก (วินาที)

# ตรวจสอบว่าเส้นทางที่ระบุให้เก็บไฟล์เสียงมีอยู่หรือไม่ ถ้าไม่มีให้สร้างเส้นทางนี้
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

#while True:
output_filename = os.path.join(output_directory, f"sound{recording_number}.wav")  # สร้างชื่อไฟล์เสียงตามลำดับ
print(f"Recording {recording_number}...\n")
recorded_voice = sounddevice.rec(int(duration_per_recording * fs), samplerate=fs, channels=1)
sounddevice.wait()
print(type(recorded_voice))
print(recorded_voice.shape)

    # write(output_filename, fs, recorded_voice)
    # print(f"Finished recording {recording_number}...\nPlease check your output file at {output_filename}")
    # recording_number += 1
    # time.sleep(delay_between_recordings)

# FFT
T = 1/fs
N = 60000
# Make data for time-domain plot
x = np.linspace(0, N*T, N)
y = recorded_voice
# Make data for frequency-domain plot
x = np.linspace(0.0, N*T, N, endpoint=False)
yf = fft(y)
xf = fftfreq(N, T)[:N//2]
yf  = 2.0/N * np.abs(yf[0:N//2])
# Check dimension for stereo or mono
p = len(yf.shape)
if p == 2:          # If stereo, choose only one set of data
    yf  = yf[:,0]
# Normalize amplitude
yf = yf/np.max(yf)
yf = yf.tolist()
# Plot the frequency domain representation
plt.figure()
plt.plot(xf, yf)


W1 = np.loadtxt('W1.csv', delimiter=',')
W2 = np.loadtxt('W2.csv', delimiter=',')
W3 = np.loadtxt('W3.csv', delimiter=',')
W4 = np.loadtxt('W4.csv', delimiter=',')


# validating model
Y = DNN(W1,W2,W3,W4,yf)
print(Y)


