# -*- coding: utf-8 -*-
# cnn1.py
# Basit 2D korelasyon + Conv2d'nin aynı çekirdeği öğrenmesi

import torch
from torch import nn


def corr2d(X, K):
    """2 boyutlu çapraz korelasyonu hesaplar (elle)."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


def main():
    # Giriş matrisi: ortada 0'lar olan 1'ler matrisi
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    print("Giriş X:")
    print(X)

    # Aradığımız gerçek çekirdek
    K = torch.tensor([[1.0, -1.0]])
    print("\nGerçek çekirdek K:")
    print(K)

    # Elle hesaplanan çıktı
    Y = corr2d(X, K)
    print("\nElle hesaplanan sonuç Y:")
    print(Y)

    # PyTorch Conv2d
    conv2d = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=(1, 2),
        bias=False
    )

    print("\nBaşlangıçta rastgele ağırlık:")
    print(conv2d.weight.data)

    # Şekilleri Conv2d'nin beklediği forma sok
    X4d = X.reshape((1, 1, 6, 8))
    Y4d = Y.reshape((1, 1, 6, 7))

    lr = 3e-2  # öğrenme oranı

    # Eğitim döngüsü
    for epoch in range(10):
        Y_hat = conv2d(X4d)
        loss = (Y_hat - Y4d) ** 2

        conv2d.zero_grad()
        loss.sum().backward()

        # Gradient descent güncellemesi
        with torch.no_grad():
            conv2d.weight -= lr * conv2d.weight.grad

        if (epoch + 1) % 2 == 0:
            print(f"dönem {epoch + 1}, loss {loss.sum():.6f}")

    print("\nEğitim sonrası öğrenilen çekirdek:")
    print(conv2d.weight.data)

    print("\nModelin tahmini (Y_hat):")
    print(Y_hat)


if __name__ == "__main__":
    main()



