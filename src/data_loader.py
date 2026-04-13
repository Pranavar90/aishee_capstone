"""
src/data_loader.py — Dataset download and path enumeration.

Datasets used
─────────────
  • Human Screaming Detection   (binary scream/ambient)   kaggle: whats2000/human-screaming-detection-dataset
  • RAVDESS                     (emotional speech)         kaggle: uwrfkaggler/ravdess-emotional-speech-audio
  • CREMA-D                     (emotional speech)         kaggle: ejlok1/cremad
  • TESS                        (emotional speech)         kaggle: ejlok1/toronto-emotional-speech-set-tess

Canonical 5-class emotion scheme
─────────────────────────────────
  0  Neutral   ← neutral, calm
  1  Happy     ← happy, surprised, joy
  2  Sad       ← sad
  3  Angry     ← angry, disgust
  4  Fearful   ← fearful, fear, distress, pain
"""

import os
import shutil
import warnings
warnings.filterwarnings("ignore")

import kagglehub

# ── Canonical label maps ──────────────────────────────────────────────────────

CANONICAL_MAP = {
    "neutral":  0, "calm":      0,
    "happy":    1, "surprised": 1, "joy": 1,
    "sad":      2,
    "angry":    3, "disgust":   3,
    "fearful":  4, "fear":      4, "distress": 4, "pain": 4,
}

EMOTION_LABELS = ["Neutral", "Happy", "Sad", "Angry", "Fearful"]

# Raw label maps per dataset
_RAVDESS_MAP = {
    "01": "neutral",  "02": "calm",
    "03": "happy",    "04": "sad",
    "05": "angry",    "06": "fearful",
    "07": "disgust",  "08": "surprised",
}

_CREMA_MAP = {
    "ANG": "angry",  "DIS": "disgust",
    "FEA": "fearful","HAP": "happy",
    "NEU": "neutral","SAD": "sad",
}


# ── Generic downloader ────────────────────────────────────────────────────────

def _kaggle_download(kaggle_id):
    """Download a Kaggle dataset and return the local cache path."""
    return kagglehub.dataset_download(kaggle_id)


def _wav_exists(directory):
    """True if *directory* contains at least one .wav file (recursively)."""
    for _, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".wav"):
                return True
    return False


# ── Dataset downloaders ───────────────────────────────────────────────────────

def download_dataset():
    """
    Download the Human Screaming Detection dataset.
    Returns the local root directory that contains Screaming/ and NotScreaming/.
    """
    dest = os.path.join(os.getcwd(), "data_acquisition", "scream_detection")
    if os.path.isdir(dest) and _wav_exists(dest):
        return dest
    os.makedirs(dest, exist_ok=True)
    src = _kaggle_download("whats2000/human-screaming-detection-dataset")
    for dp, dns, fns in os.walk(src):
        rel = os.path.relpath(dp, src)
        out = os.path.join(dest, rel)
        os.makedirs(out, exist_ok=True)
        for fn in fns:
            if fn.lower().endswith(".wav"):
                shutil.copy2(os.path.join(dp, fn), os.path.join(out, fn))
    print(f"  Scream detection dataset → {dest}")
    return dest


def download_ravdess():
    """Download RAVDESS; returns kagglehub cache path."""
    return _kaggle_download("uwrfkaggler/ravdess-emotional-speech-audio")


def download_crema_d():
    """Download CREMA-D; returns kagglehub cache path."""
    # Try the ejlok1 version first, fall back to orvile mirror
    try:
        return _kaggle_download("ejlok1/cremad")
    except Exception:
        return _kaggle_download("orvile/crema-d-emotional-multimodal-dataset")


def download_tess():
    """Download TESS (Toronto Emotional Speech Set); returns kagglehub cache path."""
    # Note: the correct slug is '-tess', NOT '-data'
    try:
        return _kaggle_download("ejlok1/toronto-emotional-speech-set-tess")
    except Exception:
        # Community mirror fallback
        return _kaggle_download("shivampanwar/toronto-emotional-speech-set")


# ── Path enumerators ──────────────────────────────────────────────────────────

def get_file_paths(dataset_path):
    """
    Walk the scream detection dataset and return
    (scream_files, non_scream_files) — lists of absolute .wav paths.
    """
    if not dataset_path:
        return [], []
    scream, ambient = [], []
    for root, _, files in os.walk(dataset_path):
        folder = os.path.basename(root).lower()
        is_scream  = ("screaming" in folder or "scream" in folder) and "not" not in folder
        is_ambient = ("notscreaming" in folder or "notscream" in folder
                      or "ambient" in folder or "noise" in folder)
        for fn in files:
            if not fn.lower().endswith(".wav"):
                continue
            fp = os.path.join(root, fn)
            if is_scream:
                scream.append(fp)
            elif is_ambient:
                ambient.append(fp)
    return scream, ambient


def get_ravdess_file_paths(dataset_path):
    """
    Walk RAVDESS, return list of (filepath, raw_emotion_str).
    Filename format: 02-01-EM-IN-ST-RP-AC.wav  (EM = 2-digit emotion code at index 2).
    """
    result = []
    for root, _, files in os.walk(dataset_path):
        for fn in files:
            if not fn.lower().endswith(".wav"):
                continue
            parts = fn.replace(".wav", "").split("-")
            if len(parts) < 3:
                continue
            emotion = _RAVDESS_MAP.get(parts[2])
            if emotion:
                result.append((os.path.join(root, fn), emotion))
    return result


def get_crema_d_file_paths(dataset_path):
    """
    Walk CREMA-D, return list of (filepath, raw_emotion_str).
    Filename format: 1001_DFA_ANG_XX.wav  (emotion = parts[2][:3]).
    """
    result = []
    audio_dir = os.path.join(dataset_path, "AudioFiles")
    if not os.path.isdir(audio_dir):
        audio_dir = dataset_path
    for root, _, files in os.walk(audio_dir):
        for fn in files:
            if not fn.lower().endswith(".wav"):
                continue
            parts = fn.split("_")
            if len(parts) < 3:
                continue
            code = parts[2][:3].upper()
            emotion = _CREMA_MAP.get(code)
            if emotion:
                result.append((os.path.join(root, fn), emotion))
    return result


# TESS folder → canonical raw emotion
# TESS folders look like: OAF_angry, YAF_happy, OAF_neutral, YAF_sad,
#                          OAF_fear, YAF_disgust, OAF_Pleasant_surprised
_TESS_FOLDER_MAP = {
    "angry":    "angry",
    "disgust":  "disgust",    # CANONICAL_MAP → 3 (Angry)
    "fear":     "fearful",
    "happy":    "happy",
    "neutral":  "neutral",
    "sad":      "sad",
    "ps":       "happy",      # Pleasant_surprise
    "pleasant": "happy",
    "surprise": "happy",
    "surprised": "happy",
}


def get_tess_file_paths(dataset_path):
    """
    Walk TESS, return list of (filepath, raw_emotion_str).

    TESS folder structure (two speaker prefixes OAF / YAF):
        <root>/OAF_angry/   <root>/YAF_happy/  etc.
    Emotion is the part after the first underscore, lower-cased.
    """
    result = []
    for root, _, files in os.walk(dataset_path):
        folder = os.path.basename(root).lower()   # e.g. "oaf_angry"
        # strip speaker prefix (OAF_ or YAF_) to get emotion token(s)
        parts = folder.split("_", 1)              # ["oaf", "angry"]
        suffix = parts[1] if len(parts) > 1 else folder  # "angry"

        # Check each known keyword in the suffix
        emotion = None
        for keyword, mapped in _TESS_FOLDER_MAP.items():
            if keyword in suffix:
                emotion = mapped
                break

        if emotion is None:
            continue

        for fn in files:
            if fn.lower().endswith(".wav"):
                result.append((os.path.join(root, fn), emotion))
    return result


# ── Combined emotion dataset ──────────────────────────────────────────────────

def get_emotion_scream_files():
    """
    Aggregate RAVDESS + CREMA-D + TESS into a unified dict.

    Returns:
        dict  {filepath: {"emotion": raw_str, "is_crime_scream": bool}}

    Note: raw emotion strings are directly usable with CANONICAL_MAP.
    Disgust/angry map to class 3; fearful/fear map to class 4.
    No remapping to "distress" or "pain" is done here — that flattening
    caused all samples to land on class 0 in previous versions.
    """
    result = {}

    # ── RAVDESS ───────────────────────────────────────────────────────────────
    try:
        path = download_ravdess()
        before = len(result)
        for fp, emotion in get_ravdess_file_paths(path):
            result[fp] = {
                "emotion": emotion,
                "is_crime_scream": emotion in ("fearful", "angry"),
            }
        print(f"  RAVDESS  : {len(result) - before:5d} samples")
    except Exception as e:
        print(f"  RAVDESS download failed: {e}")

    # ── CREMA-D ───────────────────────────────────────────────────────────────
    try:
        path = download_crema_d()
        before = len(result)
        for fp, emotion in get_crema_d_file_paths(path):
            result[fp] = {
                "emotion": emotion,
                "is_crime_scream": emotion in ("fearful", "angry"),
            }
        print(f"  CREMA-D  : {len(result) - before:5d} samples")
    except Exception as e:
        print(f"  CREMA-D download failed: {e}")

    # ── TESS ──────────────────────────────────────────────────────────────────
    try:
        path = download_tess()
        before = len(result)
        for fp, emotion in get_tess_file_paths(path):
            result[fp] = {
                "emotion": emotion,
                "is_crime_scream": emotion in ("fearful", "angry"),
            }
        print(f"  TESS     : {len(result) - before:5d} samples")
    except Exception as e:
        print(f"  TESS download failed: {e}")

    print(f"  Total emotion samples: {len(result)}")
    return result
