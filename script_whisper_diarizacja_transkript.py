from pyannote.audio import Pipeline
import whisper
from pydub import AudioSegment
import os

# === KONFIGURACJA ===
INPUT_FOLDER = r"C:\Users\jakpel\OneDrive - Kungsbacka kommun\Dokument\Ljudinspelningar"  # Folder wejściowy z plikami audio
WHISPER_MODEL = "large"  # Model Whispera (np. 'base', 'medium', 'large')
LANGUAGE = "pl"  # Język transkrypcji (np. 'pl', 'en', 'sv')

# === FUNKCJE ===
def process_file(audio_file, output_folder):
    print(f"Przetwarzanie pliku: {audio_file}")
    
    # Nazwa pliku wyjściowego
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_file = os.path.join(output_folder, f"{base_name}_transcription.txt")
    
    # Diarizacja
    print("  Przeprowadzanie diarizacji...")
    diarization_result = pipeline(audio_file)
    
    # Transkrypcja segmentów
    print("  Rozpoczynanie transkrypcji...")
    audio = AudioSegment.from_file(audio_file)
    transcriptions = []
    
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        # Wycinanie segmentu audio
        segment = audio[turn.start * 1000 : turn.end * 1000]  # W milisekundach
        temp_segment_path = "temp_segment.wav"
        segment.export(temp_segment_path, format="wav")  # Eksport segmentu
        
        # Transkrypcja segmentu
        result = model.transcribe(temp_segment_path, language=LANGUAGE)
        transcriptions.append(f"{speaker}: {result['text']}")  # Dodanie mówcy i tekstu
        
        # Czyszczenie pliku tymczasowego
        os.remove(temp_segment_path)
    
    # Zapis wyników
    print(f"  Zapisywanie wyników do pliku: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(transcriptions))
    print(f"  Transkrypcja pliku {audio_file} zakończona.")

# === GŁÓWNY PROCES ===
print("Ładowanie modelu diarizacji...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

print("Ładowanie modelu Whisper...")
model = whisper.load_model(WHISPER_MODEL)

print(f"Rozpoczynanie przetwarzania plików w folderze: {INPUT_FOLDER}")
for file in os.listdir(INPUT_FOLDER):
    file_path = os.path.join(INPUT_FOLDER, file)
    if os.path.isfile(file_path) and file.lower().endswith((".wav", ".mp3", ".m4a")):
        process_file(file_path, INPUT_FOLDER)

print("Wszystkie pliki zostały przetworzone!")
