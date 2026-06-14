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

    audio, file_sr = sf.read(wav_path)

    # Near microphone only (Channel 0)
    if len(audio.shape) == 2:
        y = audio[:, 0]
    else:
        y = audio

    if file_sr != sr:
        y = librosa.resample(
            y,
            orig_sr=file_sr,
            target_sr=sr
        )

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

    norm = 255 * (
        (log_S - log_S.min())
        /
        (log_S.max() - log_S.min() + 1e-6)
    )

    norm = norm.astype(np.uint8)

    cmap = plt.get_cmap(cmap_name)

    rgba = cmap(norm / 255.0)

    rgb = (
        rgba[:, :, :3] * 255
    ).astype(np.uint8)

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

    base_dir = r"C:\Users\Nighil Natarajan\ckmam_proj\dcase_2026_eval_data"

    for machine in sorted(os.listdir(base_dir)):

        machine_dir = os.path.join(
            base_dir,
            machine
        )

        if not os.path.isdir(machine_dir):
            continue

        print(f"\nProcessing machine: {machine}")

        wav_dir = os.path.join(
            machine_dir,
            "test"
        )

        rgb_dir = os.path.join(
            machine_dir,
            "testRGB_NEAR"
        )

        if not os.path.isdir(wav_dir):
            print(f"  [skip] no folder: {wav_dir}")
            continue

        print(
            f"  • Converting 'test' → 'testRGB_NEAR'"
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
            desc=f"{machine}/test",
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
        "\n✅ Evaluation Dataset RGB Conversion Complete!"
    )