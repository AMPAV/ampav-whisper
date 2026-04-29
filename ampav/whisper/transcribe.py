import whisper
import torch
from pathlib import Path
from ampav.core.file_formats.webvtt import paragraphs_to_webvtt
from ampav.core.logging import LOG_FORMAT
from ampav.core.schema import ToolOutput, Transcript, WordSegment, ParagraphSegment, AVMetadata
from ampav.core.gpu import ForceComputeDevice
from ampav.core.schema.transcript import remove_overlapping_words, words_to_paragraphs
from time import time
import logging
import argparse
from ampav.core.media import load_and_resample_audio_file, ChunkedAudio


def detect_language(audiofile: Path, modelname: str, device: str=None) -> dict:
    if device is None:
        device="cuda" if torch.cuda.is_available() else "cpu"    
    model = whisper.load_model(modelname).to(device)    
    _, _, adata = load_and_resample_audio_file(audiofile, 0, 16000, 1)
    audio = whisper.pad_or_trim(adata)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    _, probs = model.detect_language(mel)
    return probs


def transcribe_full_file(audiofile: Path, modelname: str, 
                         language: str | None=None, device: str=None) -> ToolOutput:
    """
    Docstring for transcribe_file
    
    :param audiofile: Description
    :type audiofile: Path
    :param modelname: Description
    :type modelname: str
    :param language: Description
    :type language: str | None
    :param device: Description
    :type device: str
    :return: Description
    :rtype: ToolOutput
    """
    # get the duration of the media file.
    av = AVMetadata.from_file(audiofile)

    # create our output structure
    output = ToolOutput(tool_name="whisper",                        
                        parameters={"model": modelname,
                                    "language": language,
                                    "device": device,
                                    "content_source": str(audiofile),                                    
                                    })

    # set the logging to log into our output structure
    output.setup_logging(ignore=['numba.'])

    # get the device if we need to
    if device is None:
        device="cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Detected device {device}")
        output.parameters['device'] = device

    with ForceComputeDevice(device):
        model = whisper.load_model(modelname).to(device)    
        _, _, adata = load_and_resample_audio_file(audiofile, 0, 16000, 1)        
        if language is None:
            audio = whisper.pad_or_trim(adata)
            mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
            _, probs = model.detect_language(mel)
            language = max(probs, key=probs.get)
            choices = {k: round(v * 100, 2) for k, v in probs.items() if v * 100 > 1}
            logging.info(f"Detected language {language}.  Reasonable choices: {choices}")
            output.parameters['language'] = language  # update this.

        output.start_time = time()
        result = model.transcribe(audio, language=language, word_timestamps=True)
        output.end_time = time()
    # build the transcript structure
    xscript = Transcript(text=result['text'].strip(),
                         media_duration=av.duration)
    for s in result['segments']:        
        xscript.paragraphs.append(ParagraphSegment(start_time=s['start'],
                                                   end_time = s['end'],
                                                   text=s['text'].strip()))
        for w in s['words']:
            xscript.words.append(WordSegment.from_str(w['word'],
                                                      start_time=w['start'],
                                                      end_time=w['end'],
                                                      tool_specific={'probability': float(w['probability'])}))
    output.output = xscript
    logging.info(f"Finished transcript, {len(xscript.paragraphs)} paragraphs, {len(xscript.words)} words.")

    return output


def transcribe_chunked_file(audiofile: Path, modelname: str, 
                    language: str | None=None, device: str=None) -> ToolOutput:
    """
    Docstring for transcribe_file
    
    :param audiofile: Description
    :type audiofile: Path
    :param modelname: Description
    :type modelname: str
    :param language: Description
    :type language: str | None
    :param device: Description
    :type device: str
    :return: Description
    :rtype: ToolOutput
    """
    # get the duration of the media file.
    av = AVMetadata.from_file(audiofile)

    # create our output structure
    output = ToolOutput(tool_name="whisper",                        
                        parameters={"model": modelname,
                                    "language": language,
                                    "device": device,
                                    "content_source": str(audiofile),                                    
                                    })

    # set the logging to log into our output structure
    output.setup_logging(ignore=['numba.'])

    # get the device if we need to
    if device is None:
        device="cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Detected device {device}")
        output.parameters['device'] = device

    with ForceComputeDevice(device):
        model = whisper.load_model(modelname).to(device)  
        if language is None:
            # detect the language by looking the first 30 seconds            
            _, audio = next(ChunkedAudio(audiofile, 0, 16000, 1).get_chunks(30,0))
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
            _, probs = model.detect_language(mel)
            language = max(probs, key=probs.get)
            choices = {k: round(v * 100, 2) for k, v in probs.items() if v * 100 > 1}
            logging.info(f"Detected language {language}.  Reasonable choices: {choices}")
            output.parameters['language'] = language  # update this.
            del audio, mel

        # on some audio whisper will just drop the end of some audio (especially
        # the gettysburg address example) so let's chunk the data in 60 second
        # bits and then piece them back together      
        output.start_time = time()
        words = []
        with ChunkedAudio(audiofile, 0, 16000, 1) as chunked_audio:
            for offsets, chunk in chunked_audio.get_chunks(60, 10):
                start_timestamp = offsets[0] 
                overlap_length = offsets[1]                 
                #print(offsets, len(chunk))
                for segment in whisper.transcribe(model, chunk,
                                                  word_timestamps=True, 
                                                  language=language)['segments']:
                    for word in segment['words']:
                        words.append(WordSegment.from_str(word['word'], 
                                                          start_time=float(word['start'] + (start_timestamp - overlap_length)),
                                                          end_time=float(word['end'] + (start_timestamp - overlap_length)),
                                                          tool_specific={'probability': float(word['probability'])})) 
                            
    
    words = remove_overlapping_words(words, lambda x: x.tool_specific['probability'])

    # build the transcript structure
    xscript = Transcript(words=words,                         
                         media_duration=av.duration)
    xscript.remove_overlapping_words(tiebreaker=lambda x: x.tool_specific['probability'])    
    output.end_time = time()
    output.output = xscript
    logging.info(f"Finished transcript, {len(xscript.paragraphs)} paragraphs, {len(xscript.words)} words.")

    return output


def cli_whisper_transcribe():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File to transcribe using whisper")
    parser.add_argument("--model", type=str, default="medium", help="Model to use")
    parser.add_argument("--language", type=str, default=None, help="Audio Language")
    parser.add_argument("--device", type=str, default=None, help="Device to use (default: best)")
    parser.add_argument("--debug", action="store_true", help="Enable debugging")
    parser.add_argument("--webvtt", action="store_true", help="Dump webvtt instead of yaml")
    parser.add_argument("--chunked", action="store_true", help="Chunk the file manually")
    args = parser.parse_args()
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)
    
    if args.chunked:
        xscript = transcribe_chunked_file(args.file, modelname=args.model, language=args.language,
                                device=args.device)
    else:
        xscript = transcribe_full_file(args.file, modelname=args.model, language=args.language,
                                device=args.device)
        
    if args.webvtt:
        print(paragraphs_to_webvtt(xscript.output.paragraphs))
    else:
        print(xscript.model_dump_yaml())
        

if __name__ == "__main__":
    cli_whisper_transcribe()