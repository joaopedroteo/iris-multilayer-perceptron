import random
import math

def getValor(txt):
    if(txt == "Iris-setosa"): return 0
    if(txt == "Iris-versicolor"): return 1
    if(txt == "Iris-virginica"): return 2

# multiplicação de matriz de valores e de pesos somados com bias
def matrixMul_bias(valores, pesos, bias):
    # C = [[0 for i in range(len(peso[0]))] for i in range(len(valores))]    
    C =[]
    for i in range(len(valores)):
        C.append([])
        for j in range(len(pesos[0])):
            C[i].append(0)

    for i in range(len(valores)):
        for j in range(len(pesos[0])):
            for k in range(len(pesos)):
                C[i][j] += valores[i][k] * pesos[k][j]
            C[i][j] += bias[j]
    return C

# multiplicação dos vetores de valores e de pesos somados com bias
def vecMat_bias(valores, pesos, bias):
    C = []
    for i in range(len(pesos[0])):
        C.append(0)

    for j in range(len(pesos[0])):
        for k in range(len(pesos)):
            C[j] += valores[k] * pesos[k][j]
        C[j] += bias[j]
    return C

def mat_vec(pesos, delta): # Matrix (pesos) x vector (delta) multipilicatoin (for backprop)
    # C = [0 for i in range(len(pesos))]
    C = []
    for i in range(len(pesos)):
        C.append(0)

    for i in range(len(pesos)):
        for j in range(len(delta)):
            C[i] += pesos[i][j] * delta[j]
    return C


def sigmoid(A, deriv=False):
    if deriv: # derivation of sigmoid (for backprop)
        for i in range(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in range(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A

def sigmoid_2(A, deriv=False):
    if deriv: # derivation of sigmoid (for backprop)
        A = A * (1 - A)
    else:
        A = 1 / (1 + math.exp(-A))
    return A

if __name__ == "__main__":

    treinamento = [
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]]
    testes = treinamento

    valores_treinamento = []
    resposta_treinamento = []
    for data in treinamento:
        valores_treinamento.append(data[:2])
        resposta_treinamento.append(data[2])

    print(valores_treinamento)
    print(resposta_treinamento)
    valores_teste = []
    resposta_teste = []
    for data in testes:
        valores_teste.append(data[:2])
        resposta_teste.append(data[2])

    alfa = 0.01
    epocas = 700

    neuronio = [2, 4, 2] # 4 entradas, 4 neuronios intermediários, 3 valores de saída

    pesos1 = []
    pesos2 = []
    bias1 = []
    bias2 = []
    for i in range(neuronio[0]):
        pesos1.append([])
        for j in range(neuronio[1]):
            pesos1[i].append(2 * random.random() - 1)

    for i in range(neuronio[1]):
        pesos2.append([])
        for j in range(neuronio[2]):
            pesos2[i].append(2 * random.random() - 1)
    
    for i in range(neuronio[1]):
        bias1.append(0)
    
    for i in range(neuronio[2]):
        bias2.append(0)

    for e in range(epocas):
        cost_total = 0
        for indice, valor in enumerate(valores_treinamento): # Update for each data; SGD
            
            # Forward propagation
            h_1 = vecMat_bias(valor, pesos1, bias1)
            X_1 = sigmoid(h_1)
            h_2 = vecMat_bias(X_1, pesos2, bias2)
            X_2 = sigmoid(h_2)
            
            # Convert to One-hot target
            target = [0, 0]
            target[int(resposta_treinamento[indice])] = 1
            # print("target ", target)
            # print("chute ", X_2)

            # Cost function, Square Root Eror
            erro = 0
            # for i in range(3):
            #     erro +=  0.5 * (target[i] - X_2[i]) ** 2 
            # cost_total += erro

            # Backward propagation
            # Update pesos2 and bias2 (layer 2)
            delta_2 = []
            for j in range(neuronio[2]):
                # delta_2.append((target[j]-X_2[j]) * X_2[j] * (1-X_2[j]))
                delta_2.append(-1* (target[j]-X_2[j]) * sigmoid_2(sigmoid_2(h_2[j]), True))

            for i in range(neuronio[1]):
                for j in range(neuronio[2]):
                    pesos2[i][j] -= alfa * (delta_2[j] * X_1[i])
                    bias2[j] -= alfa * delta_2[j]
            
            # Update pesos and bias (layer 1)
            delta_1 = mat_vec(pesos2, delta_2)
            for j in range(neuronio[1]):
                delta_1[j] = delta_1[j] * (X_1[j] * (1-X_1[j]))
            
            for i in range(neuronio[0]):
                for j in range(neuronio[1]):
                    pesos1[i][j] -=  alfa * (delta_1[j] * valor[i])
                    bias1[j] -= alfa * delta_1[j]
        
        cost_total /= len(valores_treinamento)
        #if(e % 100 == 0):
            #print(cost_total)


    res = matrixMul_bias(valores_teste, pesos1, bias1)
    res_2 = matrixMul_bias(res, pesos2, bias2)
    # print('RESSSSS', res_2)

    # Get prediction
    preds = []
    for r in res_2:
        preds.append(max(enumerate(r), key=lambda x:x[1])[0])

    # Print(prediction)
    # print(res_2)
    print(preds)

    # Calculate accuration
    acc = 0.0
    for i in range(len(preds)):
        if preds[i] == int(resposta_teste[i]):
            acc += 1
    print(acc / len(preds) * 100, "%")