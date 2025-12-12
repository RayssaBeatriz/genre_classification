import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

# Caminho do arquivo JSON com os MFCCs e labels
DATA_PATH = "/content/drive/MyDrive/classificação de genero/data_10.json"
MODEL_PATH = "modelo_classificador_genero.h5"  # Onde o modelo será salvo

def load_data(data_path):
    """Carrega os dados do JSON"""
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Dados carregados com sucesso!")
    print(f"Formato dos MFCCs: {X.shape}, Labels: {y.shape}")
    return X, y

def plot_history(history):
    """Plota acurácia e erro do treino/validação"""
    fig, axs = plt.subplots(2, figsize=(10,8))

    # Acurácia
    axs[0].plot(history.history["accuracy"], label="Acurácia treino")
    axs[0].plot(history.history["val_accuracy"], label="Acurácia teste")
    axs[0].set_ylabel("Acurácia")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Acurácia durante o treino")

    # Erro
    axs[1].plot(history.history["loss"], label="Erro treino")
    axs[1].plot(history.history["val_loss"], label="Erro teste")
    axs[1].set_ylabel("Erro")
    axs[1].set_xlabel("Época")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Erro durante o treino")

    plt.show()

if __name__ == "__main__":
    # Carregar dados
    X, y = load_data(DATA_PATH)

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    print(f"Dados de treino: {X_train.shape[0]}, Dados de teste: {X_test.shape[0]}")

    # Construção do modelo
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(10, activation='softmax')
    ])

    # Compilação do modelo
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # Treinamento
    print("\nIniciando o treinamento...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=40,
        epochs=125
    )

    # Plotar métricas
    plot_history(history)

    # Salvar modelo
    model.save(MODEL_PATH)
    print(f"\nModelo salvo em: {MODEL_PATH}")
