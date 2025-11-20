import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------------------------
# 1) Dataset klasörlerini tanımla (script klasörüne göre)
# --------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "acikhali2")

train_dir = os.path.join(DATASET_ROOT, "train")
test_dir = os.path.join(DATASET_ROOT, "test")

print("[INFO] Train dir :", train_dir)
print("[INFO] Test dir  :", test_dir)


# --------------------------------------------------------------------
# 2) Sınıf isimleri
# --------------------------------------------------------------------
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# --------------------------------------------------------------------
# 3) Dataset sınıfı
# --------------------------------------------------------------------
class FashionMNISTDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.images = []
        self.labels = []

        labels_path = os.path.join(image_folder, "labels.txt")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"labels.txt bulunamadı: {labels_path}")

        with open(labels_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            name, lbl = line.strip().split()
            img_path = os.path.join(image_folder, name)

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {img_path}")

            self.images.append(img_path)
            self.labels.append(int(lbl))

        print(f"✅ {image_folder} klasöründen {len(self.images)} görüntü yüklendi!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("L")
        arr = np.array(image, dtype=np.float32) / 255.0
        arr = torch.tensor(arr).unsqueeze(0)
        return arr, self.labels[idx]


# --------------------------------------------------------------------
# 4) Dataloaders
# --------------------------------------------------------------------
train_dataset = FashionMNISTDataset(train_dir)
test_dataset = FashionMNISTDataset(test_dir)

batch_size = 128
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# --------------------------------------------------------------------
# 5) Softmax Model
# --------------------------------------------------------------------
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


def softmax(X):
    X_exp = torch.exp(X)
    return X_exp / X_exp.sum(1, keepdim=True)


def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    y_hat = y_hat.clamp(min=1e-9)
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    y_pred = y_hat.argmax(axis=1)
    return float((y_pred == y).sum())


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# --------------------------------------------------------------------
# 6) Eğitim fonksiyonları
# --------------------------------------------------------------------
def train_epoch(net, train_iter, loss, updater):
    metric = Accumulator(3)

    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)

        updater.zero_grad()
        l.mean().backward()
        updater.step()

        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    return metric[0] / metric[2], metric[1] / metric[2]


def train_model(net, train_iter, test_iter, loss, num_epochs, updater):
    train_losses, train_accs, test_accs = [], [], []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

    # Sonuçlar.txt dosyasını oluştur
    with open("sonuçlar.txt", "w", encoding="utf-8") as f:
        last_test = test_accs[-1] * 100
        f.write("sonuçlar:\n")
        f.write(f"test sonuçları %{last_test:.0f} doğruluk oranı\n")

        for i in range(num_epochs):
            f.write(
                f"Epoch {i+1}: Loss={train_losses[i]:.4f}, "
                f"Train Acc={train_accs[i]:.4f}, "
                f"Test Acc={test_accs[i]:.4f}\n"
            )

    # Grafik
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(train_losses)
    axs[0].set_title("Eğitim Kaybı")

    axs[1].plot(train_accs, label="Train Acc")
    axs[1].plot(test_accs, label="Test Acc")
    axs[1].legend()
    axs[1].set_title("Doğruluk")
    plt.show()


# --------------------------------------------------------------------
# 7) Eğitimi Başlat
# --------------------------------------------------------------------
lr = 0.1
updater = torch.optim.SGD([W, b], lr=lr)
num_epochs = 10

train_model(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


# --------------------------------------------------------------------
# 8) Tahmin Göster
# --------------------------------------------------------------------
def predict(net, test_iter, n=6):
    for X, y in test_iter:
        break

    X, y = X[:n], y[:n]
    preds = net(X).argmax(axis=1)

    fig, axes = plt.subplots(1, n, figsize=(10, 3))
    for i in range(n):
        img = X[i].reshape(28, 28).numpy()
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"{class_names[y[i]]}\n→ {class_names[preds[i]]}")
        axes[i].axis("off")

    plt.show()


predict(net, test_iter, n=6)

