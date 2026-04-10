import os
import kagglehub
import pandas as pd
import numpy as np


def download_ravdess():
    """
    Downloads the RAVDESS Emotional Speech Audio dataset to the local project folder.
    """
    target_dir = os.path.join(os.getcwd(), "data_acquisition", "ravdess")
    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f"RAVDESS dataset already exists locally: {target_dir}")
        return target_dir

    try:
        print(f"Downloading RAVDESS to {target_dir}...")
        # Note: we use the return value as kagglehub might create a subfolder
        path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
        
        # Move files from cache to local if they aren't already there? 
        # Actually, let's just use the path returned if we can't force it easily, 
        # BUT the user wants it in THIS folder system.
        
        # Better approach: kagglehub.dataset_download doesn't have a direct 'dest' for the files
        # but we can copy them.
        import shutil
        if path != target_dir:
            print(f"Moving dataset to local folder...")
            for item in os.listdir(path):
                s = os.path.join(path, item)
                d = os.path.join(target_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
        
        return target_dir
    except Exception as e:
        print(f"Error downloading RAVDESS: {e}")
        return None


def download_crema_d():
    """
    Downloads the CREMA-D dataset to the local project folder.
    """
    target_dir = os.path.join(os.getcwd(), "data_acquisition", "crema_d")
    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f"CREMA-D dataset already exists locally: {target_dir}")
        return target_dir

    try:
        print(f"Downloading CREMA-D to {target_dir}...")
        path = kagglehub.dataset_download("orvile/crema-d-emotional-multimodal-dataset")
        
        import shutil
        if path != target_dir:
            for item in os.listdir(path):
                s = os.path.join(path, item)
                d = os.path.join(target_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
        
        return target_dir
    except Exception as e:
        print(f"Error downloading CREMA-D: {e}")
        return None


# Emotion label mappings
RAVESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

CREMA_EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def get_ravdess_file_paths(dataset_path):
    """
    Crawls RAVDESS directory and returns file paths with emotion labels.

    Returns:
        dict: {file_path: {'emotion': emotion_label, 'intensity': intensity, 'domain': 'ravdess'}}
    """
    if dataset_path is None:
        return {}

    files_dict = {}

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if not file.endswith(".wav"):
                continue

            # RAVDESS filename format: 02-01-01-01-01-01-01-01.wav
            # Modalities- Vocal_channel- Emotion- Intensity - Statement- Actor- Repetition- Actor
            parts = file.split("-")
            if len(parts) < 3:
                continue

            emotion_code = parts[2]
            emotion = RAVESS_EMOTION_MAP.get(emotion_code, "unknown")

            full_path = os.path.join(root, file)
            files_dict[full_path] = {"emotion": emotion, "domain": "ravdess"}

    print(f"Found {len(files_dict)} RAVDESS audio files.")
    return files_dict


def get_crema_d_file_paths(dataset_path):
    """
    Crawls CREMA-D directory and returns file paths with emotion labels.

    Returns:
        dict: {file_path: {'emotion': emotion_label, 'domain': 'crema_d'}}
    """
    if dataset_path is None:
        return {}

    files_dict = {}

    # CREMA-D has AudioFiles/ directory
    audio_dir = os.path.join(dataset_path, "AudioFiles")
    if not os.path.exists(audio_dir):
        audio_dir = dataset_path

    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if not file.endswith(".wav"):
                continue

            # CREMA-D format: 1001_IEO_ANG_HI.wav
            # Actor_Emotion_Level_Sentence.wav
            parts = file.split("_")
            if len(parts) < 2:
                continue

            emotion_code = parts[1][:3]  # First 3 chars of second part
            emotion = CREMA_EMOTION_MAP.get(emotion_code, "unknown")

            full_path = os.path.join(root, file)
            files_dict[full_path] = {"emotion": emotion, "domain": "crema_d"}

    print(f"Found {len(files_dict)} CREMA-D audio files.")
    return files_dict


def get_emotion_scream_files():
    """
    Combines RAVDESS and CREMA-D into a unified dataset for emotion training.

    Returns:
        dict: {file_path: {'emotion': emotion, 'domain': 'emotion_dataset', 'is_crime_scream': bool}}
    """
    emotion_files = {}

    # Download and load RAVDESS
    ravdess_path = download_ravdess()
    if ravdess_path:
        ravdess_files = get_ravdess_file_paths(ravdess_path)
        # Map to crime-relevant emotions
        for path, info in ravdess_files.items():
            emotion = info["emotion"]
            # Map to crime-related categories
            if emotion in ["fearful", "angry", "sad", "disgust"]:
                is_crime = True
            elif emotion in ["happy", "surprised", "neutral", "calm"]:
                is_crime = False
            else:
                is_crime = False

            # Only include fear/anger for high-priority crime detection
            if emotion in ["fearful", "angry"]:
                emotion = "distress"
            elif emotion == "disgust":
                emotion = "pain"

            emotion_files[path] = {
                "emotion": emotion,
                "domain": "emotion_dataset",
                "is_crime_scream": is_crime,
            }

    # Download and load CREMA-D
    crema_path = download_crema_d()
    if crema_path:
        crema_files = get_crema_d_file_paths(crema_path)
        for path, info in crema_files.items():
            emotion = info["emotion"]
            if emotion in ["fearful", "angry", "sad"]:
                is_crime = True
            else:
                is_crime = False

            if emotion in ["fearful", "angry"]:
                emotion = "distress"

            emotion_files[path] = {
                "emotion": emotion,
                "domain": "emotion_dataset",
                "is_crime_scream": is_crime,
            }

    print(f"Combined emotion dataset: {len(emotion_files)} files")
    return emotion_files


def download_dataset():
    """
    Downloads the Human Screaming Detection Dataset to the local project folder.
    """
    target_dir = os.path.join(os.getcwd(), "data_acquisition", "scream_detection")
    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f"Scream dataset already exists locally: {target_dir}")
        return target_dir

    try:
        print(f"Downloading Scream Detection Dataset to {target_dir}...")
        path = kagglehub.dataset_download("whats2000/human-screaming-detection-dataset")
        
        import shutil
        if path != target_dir:
            for item in os.listdir(path):
                s = os.path.join(path, item)
                d = os.path.join(target_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)

        return target_dir
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


def get_file_paths(dataset_path):
    """
    Crawls the dataset directory and returns lists of (path, label).
    Label 1 = Scream, 0 = Non-Scream.
    """
    if dataset_path is None:
        print("Dataset path is None. Skipping file crawling.")
        return [], []

    scream_files = []
    non_scream_files = []

    # Structure might be: /dataset/Scream/*.wav or similar.
    # Or /dataset/Positive/..., /dataset/Negative/...

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            ext = file.lower()
            if ext.endswith((".wav", ".mp3", ".ogg", ".flac", ".m4a")):
                full_path = os.path.join(root, file)
                root_lower = root.lower()
                # Heuristic for labeling based on folder names
                # We check for the negative case first to avoid 'screaming' matching 'notscreaming'
                if (
                    "notscreaming" in root_lower
                    or "noise" in root_lower
                    or "negative" in root_lower
                ):
                    non_scream_files.append(full_path)
                elif (
                    "screaming" in root_lower
                    or "positive" in root_lower
                    or "scream" in root_lower
                ):
                    scream_files.append(full_path)

    print(
        f"Found {len(scream_files)} scream files and {len(non_scream_files)} non-scream files."
    )
    return scream_files, non_scream_files


if __name__ == "__main__":
    download_dataset()
