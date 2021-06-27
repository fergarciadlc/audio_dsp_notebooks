#!/usr/bin/env python
# coding: utf-8

# # Impulse Response for Audio
# ## Convolution of audio signals
# By: <b>Fernando Garcia</b>
# 
# https://github.com/fergarciadlc
# 
# Free Reverb Impulse Responses by Voxengo - [click here](https://www.voxengo.com/impulses/)

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import librosa.display
import numpy as np


# In[2]:


def display_and_play_audio(audio_array, sampling_frequency, plot_title, figsize=(14,5)):
    plt.figure(figsize=figsize)
    librosa.display.waveplot(audio_array, sampling_frequency)
    plt.title(plot_title)
    ipd.display(ipd.Audio(audio_array, rate=sampling_frequency))


# In[3]:


audio_filename = "audio/guitar-c-major-scale.wav"
audio_sample, sr_audio = librosa.load(audio_filename)


# In[4]:


ir_filename = "IR/IMreverbs/Greek 7 Echo Hall.wav"
impulse_response, sr_ir = librosa.load(ir_filename)


# In[5]:


sr_audio, sr_ir


# In[6]:


display_and_play_audio(audio_sample, sr_audio, audio_filename.split("/")[-1])
display_and_play_audio(impulse_response, sr_ir, ir_filename.split("/")[-1])


# ### Discrete, linear convolution of two one-dimensional sequences.
# 
# $$
# (a\ast b)[n] = \sum_{m=-\infty}^\infty a[m]b[n-m]
# $$
# 
# 1. [Numpy documentation](https://numpy.org/doc/stable/reference/generated/numpy.convolve.html)
# 2. [Convolution](https://en.wikipedia.org/wiki/Convolution)

# In[7]:


convolved_signal = np.convolve(audio_sample, impulse_response)
print("Signal convolved!")
display_and_play_audio(convolved_signal, sr_audio, "Convolved Signal")


# # Example with voice and artificial Impulse Response

# In[8]:


audio_filename = "audio/voz_fernando.wav"
audio_sample, sr_audio = librosa.load(audio_filename)
ir_filename = "IR/cereal_ir.wav"
impulse_response, sr_ir = librosa.load(ir_filename)

display_and_play_audio(audio_sample, sr_audio, audio_filename.split("/")[-1])
display_and_play_audio(impulse_response, sr_ir, ir_filename.split("/")[-1])


# In[9]:


convolved_signal = np.convolve(audio_sample, impulse_response)
print("Signal convolved!")
display_and_play_audio(convolved_signal, sr_audio, "Convolved Signal")

