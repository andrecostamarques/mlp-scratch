# Documentação da Rede Neural (README)

## Descrição Geral

Este projeto implementa uma rede neural **feita à mão**, sem o uso de bibliotecas de IA como TensorFlow ou PyTorch, para realizar a classificação de imagens. O modelo foi criado para ser **genérico**, ou seja, pode ser aplicado a **qualquer banco de dados** de imagens de tamanho **28x28** com um processo de decisão baseado em classificação. No exemplo atual, o modelo utiliza o dataset **Fashion MNIST**, mas pode ser facilmente adaptado para o **MNIST** ou outros datasets similares.

## Estrutura do Código

O código foi desenvolvido em Python e utiliza apenas bibliotecas básicas como `numpy` e `matplotlib` para manipulação de arrays e visualização de dados. A classe principal, `NeuralNetwork`, implementa os componentes fundamentais de uma rede neural, incluindo propagação para frente (forward propagation), retropropagação (backpropagation), e atualização de pesos e vieses.

## Novas Funcionalidades:
- Facilitação de import e implementação.

Ao ter um arquivo .pkl contendo o modelo [(Você encontra ele aqui!)](https://drive.google.com/file/d/1Vw0WhBdbYJqnzA9YmhG3qfz0J3sPHXe9/view?usp=sharing), para utilização pode se utilizar a versão Lib do projeto!

Para a importação utilize do arquivo NMLPlib.py, após a importação, só é necessário fazer o load do modelo e fazer a inferencia com a função **preverImageCustomOnlyText**!

``` python
from NMLPlib import *

#import do modelo
modelo = loadModelo('NeuralNetwork.pkl')

# Cria uma matriz 28x28 com números inteiros aleatórios entre 0 e 255
img = Image.open("desenho.png").convert("L").resize((28, 28))
img = np.array(img, dtype=np.uint8)

#faz a inferencia e retorna o previsto e a certeza
numero, certeza = modelo.preverImageCustomOnlyText(img)
print(f"Previsto: {numero}, Certeza: {certeza:.2f}%")
```

## Principais Funções

### 1. **Classe NeuralNetwork**

Esta classe é responsável por construir e treinar a rede neural. Ela contém diversos métodos que manipulam os pesos, vieses e valores das camadas durante o processo de treino e predição.

#### **Métodos principais:**

- **`__init__(images_train, labels_train, cam_escondida, cam_final, index_image)`**
    - Inicializa os valores da rede neural. Define as camadas de input, as camadas escondidas (ocultas) e a camada final, além de inicializar a imagem de treino.

- **`initiateNN()`**
    - Inicializa os pesos, vieses e valores das camadas. Gera pesos aleatórios para as conexões entre as camadas e cria matrizes de valores para armazenar os outputs das camadas.

- **`fPropagation()`**
    - Implementa a propagação para frente (forward propagation), passando os valores de entrada através das camadas ocultas até a camada final, aplicando a função de ativação sigmoide.

- **`bPropagation()`**
    - Implementa a retropropagação (backpropagation), calculando os gradientes de erro a partir da camada final até as camadas de entrada. Retorna os gradientes dos pesos e vieses.

- **`updateWeights(learning_rate, gradiente_pesos)`**
    - Atualiza os pesos da rede neural utilizando o gradiente médio obtido no batch de treino e o fator de aprendizado (learning rate).

- **`updateBias(learning_rate, gradiente_bias)`**
    - Atualiza os vieses da rede neural utilizando o gradiente médio obtido no batch de treino e o fator de aprendizado.

- **`testeModelo(test_images, test_labels)`**
    - Testa o modelo treinado com um conjunto de dados de teste e retorna a precisão obtida.

- **`preverImagem(test_images, test_labels, index)`**
    - Realiza a previsão para uma imagem específica do conjunto de teste e compara o valor previsto com o valor esperado. Exibe a imagem e o resultado da previsão.

- **`copy()`**
    - Cria uma cópia profunda da rede neural, preservando seus estados atuais de pesos, vieses e valores.

- **`verCamadaHidden()`**
    - Exibe as camadas escondidas da rede neural, visualizando os pesos em formato de imagem.

### 2. **Funções Auxiliares**

- **`sigmoid(x)`**
    - Aplica a função de ativação sigmoide para normalizar os valores da rede.

- **`sigmoid_derivative(x)`**
    - Calcula a derivada da função sigmoide, usada no cálculo dos gradientes durante a retropropagação.

- **`media_gradiente(gradientePeso, gradienteBias, batch_size)`**
    - Calcula a média dos gradientes de pesos e vieses para um determinado batch de treino.

- **`treinarRede(redeNeural, learning_rate, batch_size, epocas, test_images, test_labels)`**
    - Função responsável pelo treinamento da rede neural. Realiza o processo de embaralhamento de dados, forward propagation, retropropagação e atualização de pesos e vieses ao longo de várias épocas. Armazena a precisão obtida a cada época e seleciona o melhor modelo durante o treino.

## Como o Modelo Funciona

1. **Inicialização e Estruturação**: O modelo começa inicializando uma rede neural com uma camada de entrada, camadas escondidas configuráveis e uma camada final de saída. Os pesos são gerados aleatoriamente, e os vieses são inicializados como zeros.
   
2. **Forward Propagation**: A imagem é passada pela rede, onde os valores de entrada são multiplicados pelos pesos, somados aos vieses e então normalizados pela função sigmoide.

3. **Backpropagation**: O erro entre o valor previsto e o valor real é propagado de volta pela rede, ajustando os pesos e vieses para minimizar esse erro.

4. **Atualização dos Pesos e Vieses**: Os gradientes calculados pela retropropagação são aplicados aos pesos e vieses da rede usando um fator de aprendizado.

5. **Teste e Validação**: Após o treinamento, a rede é testada com um conjunto de dados de teste, e sua precisão é registrada. O melhor modelo é salvo.

## Flexibilidade do Modelo

Este código foi projetado para ser genérico, permitindo que seja utilizado em **qualquer banco de dados** de imagens com dimensões **28x28**. Basta ajustar as imagens de entrada e os rótulos (labels) para que a rede neural seja capaz de fazer previsões.

## Visualização dos Resultados

A rede neural também inclui ferramentas para visualizar o desempenho:

- **Precisão por Épocas**: O gráfico "Precisão x Épocas" mostra como a precisão melhora ao longo do tempo.
- **Visualização dos Pesos**: O método `verCamadaHidden` exibe os pesos da camada de entrada, representando-os como imagens, o que permite observar como os pesos se ajustam durante o treinamento.

## Requisitos

- Python 3.x
- Numpy
- Matplotlib
- TensorFlow/Keras para importação de datasets (opcional, para Fashion MNIST)

## Como Usar

1. Instale as dependências:
    ```bash
    conda env create -f environment.yml
    ```

2. Execute o script de treino da rede neural. O exemplo atual usa o MNIST, mas pode ser adaptado para qualquer dataset de imagens 28x28 com ajuste mínimo no código.

3. Observe a precisão ao longo das épocas e visualize o melhor modelo treinado.

## Conclusão

Este projeto mostra uma implementação manual de uma rede neural, sem o uso de bibliotecas de IA. Ele é flexível, podendo ser adaptado a diferentes conjuntos de dados de imagens 28x28. A abordagem permite uma maior compreensão dos conceitos fundamentais de redes neurais, como forward propagation, backpropagation e a otimização dos pesos e vieses.



