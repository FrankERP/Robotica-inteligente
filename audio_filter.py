import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from playsound import playsound

# Read audio file
def audio_filter():
    #Play audio file
    playsound("Grabación1.wav")

    # Read audio file
    sampFreq, sound = wavfile.read("Grabación1.wav")
    print(sound.dtype, sampFreq)

    #normalize audio to be between -1 to 1
    sound = sound / (2.**15)

    #just take one channel
    sound = sound[:,0]
    
    #Measure duration of audio file
    length_in_s = sound.shape[0] / sampFreq
    print("Duration: ", length_in_s, "s")

    #Plot audio file
    plt.figure(figsize=(12, 5))
    plt.plot(sound[:],'r')
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.show()

    #Time vector
    time = np.arange(sound.shape[0])/sound.shape[0]*length_in_s
    plt.plot(time, sound[:], color='r')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

    # add noise
    yerr = (0.005*np.sin(2*np.pi*6000*time) + 0.008*np.sin(2*np.pi*8000*time) + 0.006*np.sin(2*np.pi*2500*time))
    signal = sound + yerr

    #zoom in
    plt.plot(time[6000:7000], signal[6000:7000], color='r')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

    #Fourier transform
    fft_spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.size, d=1./sampFreq)
    print("Fourier spectrum: ", fft_spectrum)
    fft_spectrum_abs = np.abs(fft_spectrum)


    plt.plot(freq, fft_spectrum_abs, color='r')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.show()

    for i,f in enumerate(freq):
        if f > 5900 and f < 6100:
            fft_spectrum[i] = 0.0
        if f > 7900 and f < 8100:
            fft_spectrum[i] = 0.0
        if f > 2400 and f < 2600:
            fft_spectrum[i] = 0.0

    fft_spectrum_abs = np.abs(fft_spectrum)
    plt.plot(freq, fft_spectrum_abs, color='r')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Filtered Fourier spectrum")
    plt.show()
    
    noiseless_signal = np.fft.irfft(fft_spectrum)
    #Audio plot
    plt.plot(time, noiseless_signal, color='r')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

    wavfile.write("Noisy_Audio.wav", sampFreq, signal)
    wavfile.write("Noiseless_Audio.wav", sampFreq, noiseless_signal)
    playsound("Noisy_Audio.wav")
    playsound("Noiseless_Audio.wav")

if __name__ == "__main__":
    audio_filter()
