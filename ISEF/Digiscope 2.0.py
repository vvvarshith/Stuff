import threading
import tkinter as tk
from tkinter import ttk
import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tkinter import *
from tkinter.font import Font
class VoiceRecorder:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('800x480')
        self.root.title("Respiratory Disease Classification")
        Font_12 = Font(family="Helvetica",size= 10)
        self.my_notebook = ttk.Notebook(self.root)
        self.my_notebook.pack()
        self.landing_page = Frame(self.my_notebook, width=1500, height=750)
        self.help = Frame(self.my_notebook, width=1500, height=750)
        self.classify = Frame(self.my_notebook, width=1500, height=750)
        self.mel_spectrogram = Frame(self.my_notebook, width=1500, height=750)
        self.landing_page.pack(fill="both", expand=1)
        self.classify.pack(fill="both", expand=1)
        self.mel_spectrogram.pack(fill="both", expand=1)
        self.help.pack(fill="both", expand=1)
        self.my_notebook.add(self.landing_page, text="Start")
        self.my_notebook.add(self.help, text="Help")
        self.my_notebook.add(self.classify, text="Classify")
        self.my_notebook.add(self.mel_spectrogram, text="Mel_Spectogram")
        self.button_1 = tk.Button(self.classify, text="Record", height=1, width=7, command=self.click_handler, font=Font_12)
        self.button_1.pack(anchor="s",side="left")
        self.button = tk.Button(self.classify, text="Classify", command=self.classification, height=1, width=7, font=Font_12)
        self.button.pack(anchor="s", side="right")
        self.plot_mel_spectrogram = tk.Button(self.mel_spectrogram, text="Mel Spectogram", command=self.graph, height=1, width=15, font= Font_12)
        self.plot_mel_spectrogram.pack(side="bottom")
        self.label = tk.Label(self.classify)
        self.label.pack()
        self.recording = False
        self.root.mainloop()
    def click_handler(self):
        if self.recording:
            self.recording = False
            self.button_1.config(fg="black")
        else:
            self.recording = True
            self.button_1.config(fg="red")
            threading.Thread(target=self.record).start()

    def record(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16,
                            channels= 1,
                            rate=16000,
                            input= True,
                            frames_per_buffer= 3200
                            )
        frames = []
        while self.recording:
            data = stream.read(3600)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        sound_file = wave.open("Sample.wav", 'wb')
        sound_file.setnchannels(1)
        sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(16000)
        sound_file.writeframes(b"".join(frames))
        sound_file.close()

    def graph(self):
        scale_file = "Sample.wav"

        sound, sample_rate = librosa.load(scale_file)

        mel_spectrogram = (librosa.feature.melspectrogram(sound, sr=sample_rate))

        fig = plt.figure(figsize=(11, 10))
        librosa.display.specshow(mel_spectrogram,
                                 x_axis="time",
                                 y_axis="mel",
                                 sr=sample_rate)
        plt.colorbar(format="%+2.f")
        plt.show()
        canvas = FigureCanvasTkAgg(fig,
                                   master=self.mel_spectrogram)
        canvas.draw()

        canvas.get_tk_widget().pack()

        toolbar = NavigationToolbar2Tk(canvas,
                                       self.mel_spectrogram)
        toolbar.update()

        canvas.get_tk_widget().pack(anchor='s', side='left')

    def classification(self):

        scale_file = '101_2_URTI.wav'

        # Attributing Array to a Label
        classes = ["(0)COPD", "(1)Healthy", "(2)URTI", "(3)Bronchiectasis", "(4)Pneumonia", "(5)Bronchiolitis"]
        le = LabelEncoder()
        le.fit(classes)
        le.inverse_transform([2])

        # Pre-Processing
        sound, sample_rate = librosa.load(scale_file)
        stft = np.abs(librosa.stft(sound))
        mfccs = np.mean(librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=40),
                        axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate),
                         axis=1)
        mel = np.mean(librosa.feature.melspectrogram(sound, sr=sample_rate), axis=1)  # Mel-scaled spectrogram
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate), axis=1)  # Spectral contrast
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound), sr=sample_rate),
                          axis=1)
        features = np.concatenate((mfccs, chroma, mel, contrast, tonnetz))
        images = []
        images.append(features)
        x_train = np.array(images)
        x_train_1 = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Model Predictions
        interpreter = tf.lite.Interpreter(model_path="digiscope_96perecent_0_15los.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        input_data = np.array(x_train_1, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Post-Processing
        pred = output_data
        pred_class = le.inverse_transform([np.argmax(pred)])[0]
        pred_percent = np.max(pred)

        z = np.array(pred)

        array_2 = z[0]

        fig = plt.figure(figsize=(10, 5))

        # Declare the plot you want to create
        plt.bar(classes, array_2)

        # Set the title of your plot
        plt.title("Prediction Probabilities",fontsize=12)

        # Give label for x and y of your plot
        plt.xlabel("Class Names",fontsize=12)
        plt.ylabel("Score",fontsize=12)
        plt.xticks(fontsize=7)

        plt.show()


        canvas = FigureCanvasTkAgg(fig,
                                   master=self.classify)
        canvas.draw()

        canvas.get_tk_widget().pack()

        toolbar = NavigationToolbar2Tk(canvas, self.classify)
        toolbar.update()

        canvas.get_tk_widget().pack()

        pred_class = le.inverse_transform([np.argmax(pred)])[0]

        pred_percent = np.max(pred)

        text = "We have predicated that Diagnosis is:", pred_class, "with a", pred_percent, "confidence."



VoiceRecorder()


