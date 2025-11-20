import torch
from torch import nn

def corr2d(X, K):  #@save
    """2 boyutlu çapraz korelasyonu hesapla."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

# Giriş ve çekirdek
X = torch.ones((6, 8))
X[:, 2:6] = 0
print("Giriş X:")
print(X)

K = torch.tensor([[1.0, -1.0]])
print("\nGerçek çekirdek K:")
print(K)

# Elle hesaplanan çıktı
Y = corr2d(X, K)
print("\nElle hesaplanan sonuç Y:")
print(Y)

# PyTorch Conv2d tanımı
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
print("\nBaşlangıçta rastgele ağırlık:")
print(conv2d.weight.data)

# Şekil dönüştürme
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # Öğrenme oranı




# Eğitim döngüsü
for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'dönem {i + 1}, loss {l.sum():.6f}')

# Sonuçlar
print("\nEğitim sonrası öğrenilen çekirdek:")
print(conv2d.weight.data)

print("\nModelin tahmini (Y_hat):")
print(Y_hat)
