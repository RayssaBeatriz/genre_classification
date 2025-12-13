# Agente classificador de gêneros musicais

**Disciplina:** Introdução à Inteligência Artificial  
**Semestre:** 2025.2  
**Professor:** Andre Luis Fonseca Faustino  
**Turma:** T04

## Integrantes do Grupo
* Pedro Henrique Álvares Róbias (20240022879)
* Rayssa Beatriz Ribeiro Cavalcante (20240019470)


## Descrição do Projeto
Nosso projeto consiste no desenvolvimento de uma rede neural capaz de identificar e classificar diferentes gêneros musicais. A ideia surgiu a partir da observação de como aplicativos de música conseguem recomendar faixas personalizadas com base nas preferências de cada usuário, facilitando a descoberta de novas músicas. Com isso, buscamos explorar e compreender uma parte dessa tecnologia, aplicando conceitos de aprendizado de máquina e inteligência artificial em um contexto musical.
Para a implementação do projeto, utilizamos tecnologias como TensorFlow, Keras e Librosa, que permitiram o processamento, análise e classificação dos dados musicais. Através dessas ferramentas, foi possível extrair características relevantes das músicas e treinar a rede neural para reconhecer padrões específicos de cada gênero.

## Guia de Instalação e Execução

### 1. Instalação das Dependências
Certifique-se de ter o **Python 3.11.9** instalado. Clone o repositório e instale as bibliotecas listadas:

```bash
# Clone o repositório
git clone https://github.com/RayssaBeatriz/genre_classification

# Entre na pasta do projeto
cd genre_classification

# Ative o ambiente virtual
.\venv\Scripts\Activate

# Instale as dependências
pip install tensorflow numpy scikit-learn matplotlib librosa
````

### 2. Como Executar

Execute o comando abaixo para extrair os MFCCs das músicas presentes no Dataset:

```bash
python extracao_mfcc.py
```

Após isso, execute o seguinte comando para treinar o modelo:

```bash
python modelo_treinamento.py
```

Por fim, execute o seguinte comando para testar o agente:

```bash
python main.py
```
## Estrutura dos Arquivos

  * `...`: Código-fonte da aplicação contendo os arquivos para extração dos MFCCs e do modelo
  * `genres_original/`: Dataset utilizado

## Resultados e Demonstração
<img width="846" height="703" alt="image" src="https://github.com/user-attachments/assets/4e44324b-ece5-42c3-9adf-637cf435115e" />


## Referências

  * [Link para o Dataset original](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
  * [Artigo, Documentação ou Tutorial utilizado como base](https://www.youtube.com/watch?v=fMqL5vckiU0&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf)
