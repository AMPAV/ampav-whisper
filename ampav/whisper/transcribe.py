import whisper
import torch
from pathlib import Path
from ampav.core.gpu import get_best_device, get_devices
from ampav.core.schema import Metadata, TranscriptOutput, WordData, ToolInformation
from time import time

def detect_language(audiofile: Path, modelname: str, device: str=None) -> dict:
    if device is None:
        device = get_best_device()
    model = whisper.load_model(modelname).to(device)    
    audio = whisper.pad_or_trim(whisper.load_audio(audiofile))
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    _, probs = model.detect_language(mel)
    return probs


def transcribe_file(audiofile: Path, modelname: str, 
                    language: str | None=None, device: str=None) -> Metadata:
    if device is None:
        device = get_best_device()
    model = whisper.load_model(modelname).to(device)    
    audio = whisper.pad_or_trim(whisper.load_audio(audiofile))
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    
    if language is None:
        _, probs = model.detect_language(mel)
        language = max(probs, key=probs.get)

    start = time()
    result = model.transcribe(audio, language=language, word_timestamps=True)
    end = time()
 
    transcript = TranscriptOutput()
    transcript.text = result['text']
    for s in result['segments']:        
        transcript.words.extend([WordData(word=x['word'], start=float(x['start']), end=float(x['end'])) for x in s['words']])
    md = Metadata(info=ToolInformation(name="whisper", version="???",                                       
                                       parameters={'language': language, 'model': modelname,
                                                   'device': device, 'source': str(audiofile)},
                                       start_time=start, end_time=end
                                       ),
                  output=transcript)
    return md




def hello():
    print("Hello from ampav.whisper.transcribe, ya jerk")
    print(get_devices())
    
