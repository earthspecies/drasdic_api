"""
integration test: interface
"""

import librosa
import pandas as pd

from drasdic.inference.inference_utils import load_audio
from drasdic.inference.interface import InferenceInterface


def main() -> None:
    config_fp = "tests/integration/args_integration_test.yaml"
    audio_fp = "tests/integration/test_audio.wav"
    st_fp = "tests/integration/test_st.txt"
    pos_label = "POS"

    interface = InferenceInterface(config_fp)

    # Load audios
    sr = 16000
    librosa.get_duration(audio_fp)  # to make checks happy; not used
    audio = load_audio(audio_fp, target_sr=sr)
    support_dur_sec = 30
    support_audio = audio[: int(support_dur_sec * sr)]
    query_audio = audio[int(support_dur_sec * sr) :]

    # support selection table
    selection_table = pd.read_csv(st_fp, sep="\t")
    support_st = selection_table[selection_table["Begin Time (s)"] <= support_dur_sec]
    # query_st = selection_table[selection_table["End Time (s)"] > support_dur_sec]

    # Prompt model and infer selection table
    interface.load_support_long(support_audio, support_st, pos_label=pos_label)
    interface.subsample_support_clips(1)
    interface.predict(query_audio, query_starttime=support_dur_sec, batch_size=1, threshold=0.5)


if __name__ == "__main__":
    main()


def test_error() -> None:
    main()
