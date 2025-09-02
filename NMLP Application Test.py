from NMLPlib import *

#import do modelo
modelo = loadModelo('NeuralNetwork.pkl')

# Cria uma matriz 28x28 com números inteiros aleatórios entre 0 e 255
img = Image.open("desenho.png").convert("L").resize((28, 28))
img = np.array(img, dtype=np.uint8)

#faz a inferencia e retorna o previsto e a certeza
numero, certeza = modelo.preverImageCustomOnlyText(img)
print(f"Previsto: {numero}, Certeza: {certeza:.2f}%")
