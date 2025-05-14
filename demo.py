import pandas as pd
from drasdic.models.model import get_model
from drasdic.inference.interface import InferenceInterface
from drasdic.inference.inference_utils import load_audio

config_fp = 'weights/args.yaml'
audio_fp = '20160907_Twin1_marmoset1.wav'
st_fp = '20160907_Twin1_marmoset1.txt'
pos_label = 'Phee'

"""
Example usage, one thirty-second support prompt
"""
interface = InferenceInterface(config_fp)

# Load audios
sr = 16000
audio = load_audio(audio_fp, target_sr=sr)
support_dur_sec = 30
support_audio = audio[:int(support_dur_sec*sr)]
query_audio = audio[int(support_dur_sec*sr):]

# support selection table
selection_table = pd.read_csv(st_fp, sep='\t')
support_st = selection_table[selection_table["Begin Time (s)"] <= support_dur_sec]
query_st = selection_table[selection_table["End Time (s)"] > support_dur_sec]

# Prompt model and infer selection table
print("Inference using single support")
interface.load_support(support_audio, support_st, pos_label=pos_label)
predicted_st = interface.predict(query_audio, query_starttime=support_dur_sec, batch_size=1)
print(predicted_st)


"""
Example usage, three support prompt taken from first five minutes
"""
# interface = InferenceInterface(config_fp)
interface.subsample_support_clips(0) # Equivalent: Reset interface prompts without reloading model weights

# Load audios
sr = 16000
audio = load_audio(audio_fp, target_sr=sr)
support_dur_sec = 300
support_audio = audio[:int(support_dur_sec*sr)]
query_audio = audio[int(support_dur_sec*sr):]

# support selection table
selection_table = pd.read_csv(st_fp, sep='\t')
support_st = selection_table[selection_table["Begin Time (s)"] <= support_dur_sec]
query_st = selection_table[selection_table["End Time (s)"] > support_dur_sec]

# Prompt model and infer selection table
print("Inference using multiple support")
interface.load_support_long(support_audio, support_st, pos_label=pos_label)
interface.subsample_support_clips(3)
predicted_st = interface.predict(query_audio, query_starttime=support_dur_sec, batch_size=1)
print(predicted_st)