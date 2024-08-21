import argparse
import os
from pathlib import Path
import shutil
import torch
import torchaudio
from faster_whisper import WhisperModel
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Assume these utility functions are defined in separate files
from utils.formatter import format_audio_list, list_audios
from utils.gpt_train import train_gpt

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def preprocess_dataset(audio_path, audio_folder_path, language, whisper_model, out_path):
    clear_gpu_cache()
    
    dataset_path = os.path.join(out_path, "dataset")
    os.makedirs(dataset_path, exist_ok=True)
    
    if audio_folder_path:
        audio_files = list(list_audios(audio_folder_path))
    else:
        audio_files = audio_path
    
    if not audio_files:
        print("No audio files found! Please provide files or specify a folder path.")
        return None, None
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "float32"
        asr_model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
        train_meta = os.path.join(dataset_path, "metadata_train.csv")
        eval_meta = os.path.join(dataset_path, "metadata_eval.csv")
        audio_total_size = format_audio_list(audio_files, asr_model=asr_model, target_language=language, out_path=dataset_path)
    except Exception as e:
        print(f"Error during data processing: {str(e)}")
        return None, None
    
    if audio_total_size < 120:
        print("The sum of the duration of the audios should be at least 2 minutes!")
        return None, None
    
    print("Dataset Processed!")
    return train_meta, eval_meta

def train_model(custom_model, version, language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
    clear_gpu_cache()
    
    run_dir = Path(output_path) / "run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    
    lang_file_path = Path(output_path) / "dataset" / "lang.txt"
    if lang_file_path.exists():
        with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
            current_language = existing_lang_file.read().strip()
            if current_language != language:
                print(f"Warning: Dataset language ({current_language}) does not match specified language ({language}). Using dataset language.")
                language = current_language
    
    try:
        max_audio_length = int(max_audio_length * 22050)
        speaker_xtts_path, config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(
            custom_model, version, language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, 
            output_path=output_path, max_audio_length=max_audio_length
        )
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return None, None, None, None, None
    
    ready_dir = Path(output_path) / "ready"
    ready_dir.mkdir(exist_ok=True)
    ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
    shutil.copy(ft_xtts_checkpoint, ready_dir / "unoptimize_model.pth")
    ft_xtts_checkpoint = os.path.join(ready_dir, "unoptimize_model.pth")
    
    speaker_reference_path = Path(speaker_wav)
    speaker_reference_new_path = ready_dir / "reference.wav"
    shutil.copy(speaker_reference_path, speaker_reference_new_path)
    
    print("Model training done!")
    return config_path, vocab_file, ft_xtts_checkpoint, speaker_xtts_path, str(speaker_reference_new_path)

def optimize_model(out_path, clear_train_data):
    out_path = Path(out_path)
    ready_dir = out_path / "ready"
    run_dir = out_path / "run"
    dataset_dir = out_path / "dataset"
    
    if clear_train_data in {"run", "all"} and run_dir.exists():
        shutil.rmtree(run_dir)
    
    if clear_train_data in {"dataset", "all"} and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    
    model_path = ready_dir / "unoptimize_model.pth"
    if not model_path.is_file():
        print("Unoptimized model not found in ready folder")
        return None
    
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    del checkpoint["optimizer"]
    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]
    
    os.remove(model_path)
    optimized_model = ready_dir / "model.pth"
    torch.save(checkpoint, optimized_model)
    ft_xtts_checkpoint = str(optimized_model)
    
    clear_gpu_cache()
    print(f"Model optimized and saved at {ft_xtts_checkpoint}")
    return ft_xtts_checkpoint

def load_model(xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker):
    clear_gpu_cache()
    if not all([xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker]):
        print("Missing model files. Please provide all required paths.")
        return None
    
    config = XttsConfig()
    config.load_json(xtts_config)
    model = Xtts.init_from_config(config)
    print("Loading XTTS model...")
    model.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, speaker_file_path=xtts_speaker, use_deepspeed=False)
    if torch.cuda.is_available():
        model.cuda()
    
    print("Model Loaded!")
    return model

def run_tts(model, lang, tts_text, speaker_audio_file, temperature, length_penalty, repetition_penalty, top_k, top_p, sentence_split, use_config):
    if model is None or not speaker_audio_file:
        print("Model not loaded or speaker audio file not provided.")
        return None, None
    
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_audio_file, gpt_cond_len=model.config.gpt_cond_len, max_ref_length=model.config.max_ref_len, sound_norm_refs=model.config.sound_norm_refs)
    
    if use_config:
        out = model.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=model.config.temperature,
            length_penalty=model.config.length_penalty,
            repetition_penalty=model.config.repetition_penalty,
            top_k=model.config.top_k,
            top_p=model.config.top_p,
            enable_text_splitting=True
        )
    else:
        out = model.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=float(repetition_penalty),
            top_k=top_k,
            top_p=top_p,
            enable_text_splitting=sentence_split
        )
    
    out_path = os.path.join(os.path.dirname(speaker_audio_file), "output.wav")
    out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
    torchaudio.save(out_path, out["wav"], 24000)
    
    print(f"Speech generated and saved to {out_path}")
    return out_path, speaker_audio_file

def speak(args):
    model = load_model(args.checkpoint, args.config, args.vocab, args.speaker)
    if model:
        output_path, reference_audio = run_tts(model, args.language, args.text, args.speaker_audio, 
                                               args.temperature, args.length_penalty, args.repetition_penalty, 
                                               args.top_k, args.top_p, args.sentence_split, args.use_config)
        if output_path:
            print(f"TTS output saved to: {output_path}")
            print(f"Reference audio used: {reference_audio}")
    else:
        print("Failed to load the model.")

def main():
    parser = argparse.ArgumentParser(description="XTTS fine-tuning and inference CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Preprocess dataset
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess dataset")
    preprocess_parser.add_argument("--audio_path", nargs="+", help="Path to audio files")
    preprocess_parser.add_argument("--audio_folder_path", help="Path to folder containing audio files")
    preprocess_parser.add_argument("--language", required=True, help="Dataset language")
    preprocess_parser.add_argument("--whisper_model", default="large-v3", choices=["large-v3", "large-v2", "large", "medium", "small"], help="Whisper model to use")
    preprocess_parser.add_argument("--out_path", required=True, help="Output path")

    # Train model
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--custom_model", default="", help="Path to custom model.pth file")
    train_parser.add_argument("--version", default="v2.0.2", choices=["v2.0.3", "v2.0.2", "v2.0.1", "v2.0.0", "main"], help="XTTS base version")
    train_parser.add_argument("--language", required=True, help="Training language")
    train_parser.add_argument("--train_csv", required=True, help="Path to train CSV")
    train_parser.add_argument("--eval_csv", required=True, help="Path to eval CSV")
    train_parser.add_argument("--num_epochs", type=int, default=6, help="Number of epochs")
    train_parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    train_parser.add_argument("--grad_acumm", type=int, default=1, help="Gradient accumulation steps")
    train_parser.add_argument("--out_path", required=True, help="Output path")
    train_parser.add_argument("--max_audio_length", type=float, default=11, help="Max permitted audio size in seconds")

    # Optimize model
    optimize_parser = subparsers.add_parser("optimize", help="Optimize model")
    optimize_parser.add_argument("--out_path", required=True, help="Output path")
    optimize_parser.add_argument("--clear_train_data", choices=["none", "run", "dataset", "all"], default="none", help="Clear training data after optimizing")

    # Combined load and tts into speak command
    speak_parser = subparsers.add_parser("speak", help="Load model and run text-to-speech")
    speak_parser.add_argument("--checkpoint", required=True, help="Path to XTTS checkpoint")
    speak_parser.add_argument("--config", required=True, help="Path to XTTS config")
    speak_parser.add_argument("--vocab", required=True, help="Path to XTTS vocab")
    speak_parser.add_argument("--speaker", required=True, help="Path to XTTS speaker file")
    speak_parser.add_argument("--language", required=True, help="TTS language")
    speak_parser.add_argument("--text", required=True, help="Input text")
    speak_parser.add_argument("--speaker_audio", required=True, help="Path to speaker reference audio")
    speak_parser.add_argument("--temperature", type=float, default=0.75, help="Temperature")
    speak_parser.add_argument("--length_penalty", type=float, default=1, help="Length penalty")
    speak_parser.add_argument("--repetition_penalty", type=float, default=5, help="Repetition penalty")
    speak_parser.add_argument("--top_k", type=int, default=50, help="Top K")
    speak_parser.add_argument("--top_p", type=float, default=0.85, help="Top P")
    speak_parser.add_argument("--sentence_split", action="store_true", help="Enable text splitting")
    speak_parser.add_argument("--use_config", action="store_true", help="Use inference settings from config")

    args = parser.parse_args()

    if args.command == "preprocess":
        train_csv, eval_csv = preprocess_dataset(args.audio_path, args.audio_folder_path, args.language, args.whisper_model, args.out_path)
        if train_csv and eval_csv:
            print(f"Train CSV: {train_csv}")
            print(f"Eval CSV: {eval_csv}")
    elif args.command == "train":
        results = train_model(args.custom_model, args.version, args.language, args.train_csv, args.eval_csv, 
                              args.num_epochs, args.batch_size, args.grad_acumm, args.out_path, args.max_audio_length)
        if results:
            config_path, vocab_file, ft_xtts_checkpoint, speaker_xtts_path, speaker_reference = results
            print(f"Config path: {config_path}")
            print(f"Vocab file: {vocab_file}")
            print(f"Fine-tuned checkpoint: {ft_xtts_checkpoint}")
            print(f"Speaker XTTS path: {speaker_xtts_path}")
            print(f"Speaker reference: {speaker_reference}")
    elif args.command == "optimize":
        optimized_checkpoint = optimize_model(args.out_path, args.clear_train_data)
        if optimized_checkpoint:
            print(f"Optimized checkpoint: {optimized_checkpoint}")
    elif args.command == "speak":
        speak(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
