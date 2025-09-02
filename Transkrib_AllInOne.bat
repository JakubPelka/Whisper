@echo off
echo Running script 1: Convert files to Wav...
python script_recordings_2_wav.py
echo Finished script 1.

echo Running script 2: move m4a, and mp3 to arkiv...
python script_move_m4a_mp3_to_arkiv.py
echo Finished script 2.

echo Running script 3: Diarisation + transkrib...
python script_whisper_diarizacja_transkript.py
echo Finished script 3.

echo All tasks completed.
pause