# 🧠 Modelo Simples de Rede Neural Convolucional (CNN)

Este é um modelo introdutório de **Rede Neural Convolucional (CNN)** usando **Keras** e **TensorFlow**. Ele foi projetado para classificar imagens em duas categorias (binário), utilizando um conjunto de dados de treinamento e teste.

## 📌 Estrutura do Modelo

- **Camada de Convolução**: Extração de características com 32 filtros e ativação ReLU.
- **Camada de Pooling**: Redução da dimensionalidade usando MaxPooling.
- **Camada de Flattening**: Transformação das matrizes em um vetor unidimensional.
- **Camadas Densas**: Totalmente conectadas com ativação ReLU e uma saída com ativação sigmoide.

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

classifier.fit(training_set, steps_per_epoch=8000, epochs=25, validation_data=test_set, validation_steps=2000)
classifier.save("cnn.h5")
```

## 🚀 Observações

- **Modelo básico**: Este é um modelo introdutório adequado para iniciantes.
- **Pré-processamento de imagens**: As imagens são redimensionadas para **64x64** e normalizadas.
- **Uso de Data Augmentation**: Para melhorar o desempenho em imagens variadas.
- **Treinamento supervisionado**: Classificação binária com **função de perda `binary_crossentropy`**.

Para um modelo mais robusto, explorar **arquiteturas mais profundas**, **mais camadas convolucionais** e **outros otimizadores**. 🚀
