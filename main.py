import json, os, math, librosa

# Caminhos e constantes
CAMINHO_DATASET = "genres_original"
CAMINHO_JSON = "data_10.json"
TAXA_AMOSTRAGEM = 22050
DURACAO_FAIXA = 30 
AMOSTRAS_POR_FAIXA = TAXA_AMOSTRAGEM * DURACAO_FAIXA


def salvar_mfcc(caminho_dataset, caminho_json, num_mfcc=13, n_fft=2048, hop_length=512, num_segmentos=5):

    # dicionario para armazenar mapeamento (mapping), rótulos (labels) e MFCCs
    dados = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    amostras_por_segmento = int(AMOSTRAS_POR_FAIXA / num_segmentos)
    num_vetores_mfcc_por_segmento = math.ceil(amostras_por_segmento / hop_length)

    # itera por todas as subpastas de gênero
    for i, (caminho_diretorio, nomes_diretorios, nomes_arquivos) in enumerate(os.walk(caminho_dataset)):

        # garante que estamos processando uma subpasta de gênero
        if caminho_diretorio is not caminho_dataset:

            # salva o rótulo do gênero (nome da subpasta) no mapeamento (mapping)
            rotulo_semantico = caminho_diretorio.split("/")[-1]
            dados["mapping"].append(rotulo_semantico)
            print("\nProcessando: {}".format(rotulo_semantico))

            # processa todos os arquivos de áudio no subdiretório do gênero
            for nome_arquivo in nomes_arquivos:

                # carrega o arquivo de áudio
                caminho_arquivo = os.path.join(caminho_diretorio, nome_arquivo)
                sinal, taxa_amostragem = librosa.load(caminho_arquivo, sr=TAXA_AMOSTRAGEM)

                # processa todos os segmentos do arquivo de áudio
                for d in range(num_segmentos):

                    # calcula a amostra inicial e final para o segmento atual
                    inicio = amostras_por_segmento * d
                    fim = inicio + amostras_por_segmento

                    # extrai os coeficientes mfcc
                    # CORREÇÃO: Argumento 'y' nomeado explicitamente
                    mfcc = librosa.feature.mfcc(y=sinal[inicio:fim], sr=taxa_amostragem, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # armazena apenas a característica mfcc com o número esperado de vetores
                    if len(mfcc) == num_vetores_mfcc_por_segmento:
                        dados["mfcc"].append(mfcc.tolist())
                        dados["labels"].append(i-1)
                        print("{}, segmento:{}".format(caminho_arquivo, d+1))

    # salva os MFCCs no arquivo json
    with open(caminho_json, "w") as fp:
        json.dump(dados, fp, indent=4)
        
        
if __name__ == "__main__":
    salvar_mfcc(CAMINHO_DATASET, CAMINHO_JSON, num_segmentos=10)