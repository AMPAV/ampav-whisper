import whisper
import torch
from pathlib import Path
from ampav.core.logging import LOG_FORMAT, ListLoggingHandler
from ampav.core.schema import ToolOutput, Transcript, WordSegment, ParagraphSegment, AVMetadata, load_ampav_file
from ampav.core.gpu import ForceComputeDevice
from time import time
import logging
import argparse
from ampav.core.media import load_and_resample_audio_file, ChunkedAudio
from ampav.core.utils import dump_data


def detect_language(audiofile: Path, modelname: str, device: str=None) -> dict:
    """Detect the language of the audio file

    Args:
        audiofile (Path): Audio file to process
        modelname (str): Whisper model to use
        device (str, optional): Compute device name. Defaults to best available.

    Returns:
        dict: a mapping of language -> probability values
    """
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
    """Transcribe an entire file without chunking

    Args:
        audiofile (Path): File to transcribe
        modelname (str): whisper model to use
        language (str | None, optional): language used in the file. Defaults to detect.
        device (str, optional): Compute device to use. Defaults to best available.

    Returns:
        ToolOutput: a transcription tool output
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
        result = model.transcribe(adata, language=language, word_timestamps=True)
        output.end_time = time()
    # build the transcript structure
    xscript = Transcript(text=result['text'].strip(),
                         media_duration=av.duration)
    for s in result['segments']:        
        xscript.paragraphs.append(ParagraphSegment(start_time=s['start'],
                                                   end_time = s['end'],
                                                   text=s['text'].strip()))
        for w in s['words']:
            xscript.words.append(WordSegment.from_str(w['word'].strip(),
                                                      start_time=w['start'],
                                                      end_time=w['end'],
                                                      tool_specific={'probability': float(w['probability'])}))
    output.output = xscript
    logging.info(f"Finished transcript, {len(xscript.paragraphs)} paragraphs, {len(xscript.words)} words.")

    return output


def transcribe_chunked_file(audiofile: Path, modelname: str, 
                    language: str | None=None, device: str=None,
                    chunk_size: float=60, chunk_overlap: float=0) -> ToolOutput:
    """Transcribe a file, chunk at a time

    Args:
        audiofile (Path): File to transcribe
        modelname (str): Whisper model to use
        language (str | None, optional): Language used in the audio. Defaults to detect.
        device (str, optional): Compute device to use. Defaults to best available.
        chunk_size (float, optional): Audio chunk size, in seconds. Defaults to 60.
        chunk_overlap (float, optional): Audio chunk overlap, in seconds. Defaults to 0.

    Returns:
        ToolOutput: A transcription tool output
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
            with ChunkedAudio(audiofile, 0, 16000, 1) as chunked_audio:
                _, audio = next(chunked_audio.get_chunks(chunk_size, chunk_overlap))
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
                _, probs = model.detect_language(mel)
                language = max(probs, key=probs.get)
                choices = {k: round(v * 100, 2) for k, v in probs.items() if v * 100 > 1}
                logging.info(f"Detected language {language}.  Reasonable choices: {choices}")
                output.parameters['language'] = language  # update this.
                del audio, mel

        # on some audio whisper will just drop the end of some audio (especially
        # the gettysburg address example) so let's chunk the data in chunk_size second
        # bits and then piece them back together      
        output.start_time = time()
        words = []
        with ChunkedAudio(audiofile, 0, 16000, 1) as chunked_audio:
            for start_timestamp, chunk in chunked_audio.get_chunks(chunk_size, chunk_overlap):
                for segment in whisper.transcribe(model, chunk,
                                                  word_timestamps=True, 
                                                  language=language)['segments']:
                    for word in segment['words']:
                        words.append(WordSegment.from_str(word['word'].strip(), 
                                                          start_time=float(word['start'] + start_timestamp),
                                                          end_time=float(word['end'] + start_timestamp),
                                                          tool_private={'probability': float(word['probability'])}))                         
                        #logging.debug(f"{words[-1]}")    

    # build the transcript structure
    xscript = Transcript(words=words,                         
                         media_duration=av.duration) 
    xscript.remove_overlapping_words()    
    output.end_time = time()
    output.output = xscript
    logging.info(f"Finished transcript, {len(xscript.paragraphs)} paragraphs, {len(xscript.words)} words.")

    return output


def cli_whisper_transcribe():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path, help="File to transcribe using whisper")
    parser.add_argument("output", type=Path, help="Output file")
    parser.add_argument("--model", type=str, default="medium", help="Model to use")
    parser.add_argument("--language", type=str, default=None, help="Audio Language")
    parser.add_argument("--device", type=str, default=None, help="Device to use (default: best)")
    parser.add_argument("--debug", action="store_true", help="Enable debugging")
    parser.add_argument("--chunk_size", type=int, default=30, help="Size of chunks to process")
    parser.add_argument("--chunk_overlap", type=int, default=5, help="Number of seconds of audio overlap")    
    parser.add_argument("--chunked", action="store_true", help="Chunk the file manually")
    parser.add_argument("--format", choices=['yaml', 'json', 'pickle'], default='yaml', help="Output format, default yaml")
    args = parser.parse_args()
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)

    # capture the logging
    logs = []
    loghandler = ListLoggingHandler(logs)
    logging.getLogger().addHandler(loghandler)

    logging.info("Starting processing")
    start = time()    
    if args.chunked:
        result = transcribe_chunked_file(args.file, modelname=args.model, language=args.language,
                                device=args.device, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    else:
        result = transcribe_full_file(args.file, modelname=args.model, language=args.language,
                                device=args.device)

    # update the tool_output structure with the runtime things            
    result.start_time = start
    result.end_time = time()            
    logging.info(f"Saving data to {args.output} in {args.format} format")
    result.messages = logs
    dump_data(result, args.format, args.output)   


def cli_whisper_adjust_transcript():
    # This is mostly for debugging the remove_overlapping_words and rebuilding
    # the text/paragraph fields
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow_pickle", action="store_true", help="Allow pickle files to be loaded")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("file", type=Path, help="File to transcribe using whisper")
    parser.add_argument("output", type=Path, help="Output file")
    parser.add_argument("--format", choices=['yaml', 'json', 'pickle'], default='yaml', help="Output format, default yaml")
    parser.add_argument("--paragraph_gap", type=float, default=1.5, help="Time gap between paragraphs")
    parser.add_argument("--max_paragraph", type=float, default=10, help="Maximum time length for a paragraph")

    args = parser.parse_args()    

    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)

    logging.info(f"Loading data file {args.file}")
    data = load_ampav_file(args.file, args.allow_pickle)
    data.output.remove_overlapping_words(paragraph_gap=args.paragraph_gap,
                                         max_paragraph=args.max_paragraph)

    loghandler = ListLoggingHandler(data.messages)
    logging.getLogger().addHandler(loghandler)

    logging.info(f"Writing {args.output} in {args.format}")
    dump_data(data, args.format, args.output)

if __name__ == "__main__":
    cli_whisper_transcribe()
    #cli_whisper_adjust_transcript()