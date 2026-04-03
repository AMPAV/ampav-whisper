import whisper
import torch
from pathlib import Path
from ampav.core.gpu import get_devices



def detect_language(audiofile: Path, modelname: str) -> dict:
    model = whisper.load_model(modelname)
    model.to()
    audio = whisper.load_audio(audiofile)
    audio = whisper.pad_or_trim(audio)





def hello():
    print("Hello from ampav.whisper.transcribe, ya jerk")
    print(get_devices())
    
