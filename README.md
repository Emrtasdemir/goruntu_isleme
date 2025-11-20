# FashionMNIST – Softmax Regression & CNN Demo

Bu proje FashionMNIST veri seti üzerinde hem **Softmax Regresyon** hem de küçük bir **CNN demo (PyTorch + Keras)** çalıştırmak için hazırlanmış tam bir örnektir.  
Proje GitHub’dan klonlandığında veya ZIP olarak indirildiğinde **direkt çalışır**.

---

## 📁 Proje Yapısı

```
FashionMNIST/
│  softmax_regresyon_diskten.py
│  export_fashionmnist_to_png.py
│  cnn1.py
│  Main.py
│  sonuçlar.txt
│
└─ acikhali2/
      ├─ train/
      └─ test/
```

---

## 🚀 Nasıl Çalıştırılır?

### 1) Sanal ortam

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Gerekli paketler

```
pip install torch torchvision pillow matplotlib keras
```

---

## 🖼 Dataset Oluşturma

```
python export_fashionmnist_to_png.py
```

Bu komut train/test klasörlerini PNG + labels.txt ile oluşturur.

---

## 🧠 Softmax Modeli Eğitme

```
python softmax_regresyon_diskten.py
```

- Loss & Accuracy grafiklerini gösterir  
- Tahmin örnekleri çıkar  
- sonuçlar.txt oluşturur  

---

## 📊 Örnek Sonuç

```
Epoch 10: Loss=0.4266, Train Acc=0.8533, Test Acc=0.8357
```

Toplam doğruluk: **%83**

---

## 🧪 CNN Demo

```
python cnn1.py
```

---

## ⭐ Amacı

Bu proje, FashionMNIST üzerinde:

- Softmax regresyon  
- CNN filtre öğrenimi  
- PNG dataset kullanımı  
- Eğitim sonuçlarının otomatik kaydı  

konularını uygulamalı öğretir.

---
