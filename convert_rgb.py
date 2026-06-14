import os
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


def wav_to_rgb_spectrogram(
    wav_path,
    output_path,
    sr=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=128,
    fmin=20,
    fmax=8000,
    cmap_name="plasma"
):
    """
    Convert one WAV file into a 224x224 RGB spectrogram.
    Uses ONLY the Near Microphone (Channel 0).
    """

    # =====================================================
    # Load stereo audio
    # Channel 0 = Near microphone
    # Channel 1 = Far microphone
    # =====================================================

    audio, file_sr = sf.read(wav_path)

    # Handle stereo audio
    if len(audio.shape) == 2:
        y = audio[:, 0]  # Near microphone
    else:
        y = audio

    # Resample if needed
    if file_sr != sr:
        y = librosa.resample(
            y,
            orig_sr=file_sr,
            target_sr=sr
        )

    # =====================================================
    # Log-Mel Spectrogram
    # =====================================================

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0
    )

    log_S = librosa.power_to_db(
        S,
        ref=np.max
    )

    # =====================================================
    # Normalize to [0,255]
    # =====================================================

    norm = 255 * (
        (log_S - log_S.min())
        /
        (log_S.max() - log_S.min() + 1e-6)
    )

    norm = norm.astype(np.uint8)

    # =====================================================
    # Apply RGB colormap
    # =====================================================

    cmap = plt.get_cmap(cmap_name)

    rgba = cmap(norm / 255.0)

    rgb = (
        rgba[:, :, :3] * 255
    ).astype(np.uint8)

    # =====================================================
    # Save image
    # =====================================================

    img = Image.fromarray(rgb)

    img = img.resize(
        (224, 224),
        Image.BILINEAR
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    img.save(output_path)


if __name__ == "__main__":

    base_dir = r"C:\Users\Nighil Natarajan\ckmam_proj\dcase_2026"

    categories = [
        "train",
        "supplemental",
        "test"
    ]

    for machine in sorted(os.listdir(base_dir)):

        machine_dir = os.path.join(
            base_dir,
            machine
        )

        if not os.path.isdir(machine_dir):
            continue

        print(f"\nProcessing machine: {machine}")

        for cat in categories:

            wav_dir = os.path.join(
                machine_dir,
                cat
            )

            # ==========================================
            # NEW OUTPUT FOLDER
            # ==========================================

            rgb_dir = os.path.join(
                machine_dir,
                f"{cat}RGB_NEAR"
            )

            if not os.path.isdir(wav_dir):
                print(f"  [skip] no folder: {wav_dir}")
                continue

            print(
                f"  • Converting '{cat}' → '{cat}RGB_NEAR'"
            )

            os.makedirs(
                rgb_dir,
                exist_ok=True
            )

            wav_files = [
                f for f in os.listdir(wav_dir)
                if f.lower().endswith(".wav")
            ]

            for fname in tqdm(
                wav_files,
                desc=f"{machine}/{cat}",
                ncols=80
            ):

                in_path = os.path.join(
                    wav_dir,
                    fname
                )

                out_name = (
                    os.path.splitext(fname)[0]
                    + ".png"
                )

                out_path = os.path.join(
                    rgb_dir,
                    out_name
                )

                wav_to_rgb_spectrogram(
                    in_path,
                    out_path
                )

    print(
        "\n✅ Near-Microphone RGB Spectrogram Conversion Complete!"
    )