"""Small inference feature helpers for silence and frame counts.

The former evaluation-expression harness is outside the shipped boundary.

Reads: extractor_base.py
"""

from .extractor_base import *
import librosa
import numpy as np

class BlankMusic(ExtractorBase):
    def get_feature_class(self):
        return io.MusicIO

    def extract(self,entry,**kwargs):
        time=60.0 # seconds
        if('time' in kwargs):
            time=kwargs['time']
        return np.zeros((int(np.ceil(time*entry.prop.sr))))


class FrameCount(ExtractorBase):
    def get_feature_class(self):
        return io.IntegerIO

    def extract(self,entry,**kwargs):
        # self.require(entry.prop.hop_length)
        return entry.dict[kwargs['source']].get(entry).shape[0]
