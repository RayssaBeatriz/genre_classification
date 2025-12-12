from tensorflow.keras.models import load_model
import numpy as np
import librosa

CAMINHO_MODELO = "modelo_classificador_genero.h5"

def processar_wav(wav_file, n_mfcc=13, max_len=130):

    """
    Carrega um arquivo WAV e extrai os MFCCs,
    garantindo que o número de frames seja igual a max_len.
    """
    y, sr = librosa.load(wav_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T  # (tempo, coeficientes)

    # Padding se menor que max_len
    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0,pad_width),(0,0)), mode='constant')
    # Corte se maior que max_len
    else:
        mfcc = mfcc[:max_len, :]

    mfcc = mfcc[np.newaxis, ...]  # adiciona dimensão de batch
    return mfcc

def prever_genero(modelo, mfcc_sample):
    """
    Recebe o modelo e MFCC de um áudio e retorna a classe prevista
    """
    predicao = modelo.predict(mfcc_sample)
    indice_label = np.argmax(predicao)

    mapeamento_genero = {
        0: "blues", 1: "classical", 2: "country", 3: "disco",
        4: "hiphop", 5: "jazz", 6: "metal", 7: "pop",
        8: "reggae", 9: "rock"
    }

    return indice_label, mapeamento_genero[indice_label]

if __name__ == "__main__":
    # Carrega modelo treinado
    modelo = load_model(CAMINHO_MODELO)

    # Solicita caminho do WAV
    wav_file = input("Digite o caminho do arquivo WAV: ")

    # Processa WAV com padding/corte
    n_mfcc = 13      # Número de coeficientes MFCC
    max_len = 130    # Número de frames
    mfcc_sample = processar_wav(wav_file, n_mfcc=n_mfcc, max_len=max_len)

    # Predição do gênero
    indice, genero = prever_genero(modelo, mfcc_sample)
    print(f"Gênero previsto: {genero} (classe {indice})")
