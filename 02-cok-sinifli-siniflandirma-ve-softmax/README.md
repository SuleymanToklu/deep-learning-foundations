# Çok Sınıflı Sınıflandırma ve Softmax Aktivasyonu

Bu modül; ikiden fazla sınıfa sahip hedef değişkenlerin sınıflandırılmasında kullanılan Softmax çıktı katmanı ve Categorical Cross-Entropy kayıp fonksiyonunun hematolojik laboratuvar kan değerleri (anemi teşhisi) üzerindeki uygulamasını gösterir.

## Notebook
- [keras-softmax-with-anem-a-type-classification.ipynb](keras-softmax-with-anem-a-type-classification.ipynb)

## Temel Kavramlar
- Softmax formülü: $\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$
- One-hot etiket kodlama (`to_categorical`)
- Çok sınıflı kayıp fonksiyonu (Categorical Cross-Entropy)
- Erken durdurma (EarlyStopping) ve doğruluk metrikleri
