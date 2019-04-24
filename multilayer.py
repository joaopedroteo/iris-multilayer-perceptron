import random
import math

def getValor(txt):
    if(txt == "Iris-setosa"): return 0
    if(txt == "Iris-versicolor"): return 1
    if(txt == "Iris-virginica"): return 2



if __name__ == "__main__":

    with open('iris.data', 'r') as f:
        dataset = f.read().split()
        for i in range(len(dataset)):
            dataset[i] = dataset[i].split(',')
        print(dataset)

    #troca nomes por valores inteiros
    for linha in dataset:
        linha[4] = getValor(linha[4])
        linha[:4] = [float(linha[j]) for j in range(len(linha))]
        print(linha)

    random.shuffle(dataset)
    treinamento = dataset[:int(len(dataset) * 0.8)]
    testes = dataset[int(len(dataset) * 0.8):]

    valores_treinamento = []
    resposta_treinamento = []
    for data in treinamento:
        valores_treinamento.append(data[:4])
        resposta_treinamento.append(data[4])

    valores_teste = []
    resposta_teste = []
    for data in testes:
        valores_teste = data[:4]
        resposta_teste = data[4]
    
