# https://aihub.cloud.google.com/u/0/p/products%2F186e4836-0280-4bf0-9ca5-006eed0265fb

############################################################################
# Shor, Joel & Jansen, Aren & Maor, Ronnie & Lang, Oran & Quitry,          #
# Felix & Tagliasacchi, Marco & Tuval, Omry & Shavitt, Ira & Emanuel,      #
# Dotan & Haviv, Yinnon. (2020). Towards Learning a Universal Non-Semantic #
# Representation of Speech.                                                #
############################################################################


import os
import numpy as np
import tensorflow_hub as hub
import moviepy.editor as mp
import scipy.io.wavfile
import resampy
import tensorflow as tf
from glob import glob
import json
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


TRILL_PATH = 'https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/1'
TRILL_DISTILLED_PATH = 'https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/1'


def get_audio_sample(video_path, sr=16000):
    """

    Get audio sample from video

    Parameters:
        video_path (str): path of the video for audio sample extraction
        sr (int): the sample rate of audio (default 16000)

    Returns:
        wav_data (numpy.ndarray): the array of extracted audio sample

    """

    # open the video and convert to temporary wav file
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile("temp.wav", fps=sr, verbose=False)

    # read and extract wav file to numpy array and delete temporary wav file
    sample_rate, wav_data = scipy.io.wavfile.read("temp.wav")
    os.remove("temp.wav")

    # convert the wave data to mono channel
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)

    # Resample to the rate assumed by TRILL
    if sample_rate != sr:
        wav_data = resampy.resample(wav_data, sample_rate, sr)

    # normalize the wave data
    wav_data = wav_data / np.sum(wav_data ** 2) ** 0.5

    return wav_data


def audio_tensor_extraction(video_path, distrilled=True, sr=16000):
    """
    Convert video into audio features with TRILL

    Parameters:
        video_path (str): path of the video for audio sample extraction
        sr (int): the sample rate of audio (default 16000)
        distrilled (bool): option for distilled TRILL (default True)

    Returns:
        emb (numpy.ndarray): the audio embedding feature tensor extracted

    """

    # get numpy audio sample from video
    sample = get_audio_sample(video_path)

    # load TRILL weights from tfhub
    if distrilled:
        module = hub.load(TRILL_DISTILLED_PATH)
    else:
        module = hub.load(TRILL_PATH)

    # extract audio features with TRILL and return embedding tensor
    emb = module(samples=sample, sample_rate=sr)['embedding']
    emb = np.mean(emb.numpy(), axis=0)
    return emb.tolist()


if __name__ == '__main__':
    exts = ['mp4']
    audio_dict = {}
    for ext in exts:
        for path in glob(os.path.join('data', f'*.{ext}')):
            file_name = os.path.split(path)[-1].split('.')[0]
            aud = audio_tensor_extraction(path)
            audio_dict[file_name] = aud
    json.dump(audio_dict, open('features/audio.json', 'w'))
