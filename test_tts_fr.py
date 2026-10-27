"""Test autonome : synthèse française avec Fun-CosyVoice3-0.5B (CPU)."""
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'third_party', 'Matcha-TTS'))

MODEL_DIR = os.path.join(ROOT, 'pretrained_models', 'Fun-CosyVoice3-0.5B')
# Référence du dépôt (voix chinoise) : suffisant pour un test technique uniquement.
# Pour du français de qualité, passer un WAV français : python test_tts_fr.py ma_voix.wav
PROMPT_WAV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'asset', 'zero_shot_prompt.wav')
OUTPUT = os.path.join(ROOT, 'output_fr.wav')
# CosyVoice3 exige ce préfixe système suivi de <|endofprompt|>
TEXT = ("You are a helpful assistant.<|endofprompt|>"
        "Bonjour, ceci est un premier test de synthèse vocale en français.")


def main():
    import torch
    import soundfile as sf
    from cosyvoice.cli.cosyvoice import AutoModel

    for p in (MODEL_DIR, PROMPT_WAV):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    t0 = time.time()
    print(f'Chargement du modèle : {MODEL_DIR}')
    model = AutoModel(model_dir=MODEL_DIR)
    print(f'Modèle chargé en {time.time() - t0:.1f}s (sample_rate={model.sample_rate})')

    t0 = time.time()
    chunks = [out['tts_speech'] for out in
              model.inference_cross_lingual(TEXT, PROMPT_WAV, stream=False)]
    if not chunks:
        raise RuntimeError('Le modèle n\'a produit aucun audio.')
    speech = torch.cat(chunks, dim=1)
    sf.write(OUTPUT, speech.squeeze(0).numpy(), model.sample_rate)
    print(f'Audio généré en {time.time() - t0:.1f}s ({speech.shape[1] / model.sample_rate:.1f}s d\'audio)')
    print(f'Fichier WAV : {OUTPUT}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('ERREUR :', file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
