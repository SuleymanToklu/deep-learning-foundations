# Transfer Learning ve Fine-Tuning ile Görüntü Sınıflandırma

Önceden ImageNet üzerinde eğitilmiş MobileNetV2 taban mimarisi, dondurulmuş katmanlar, Global Average Pooling, veri artırma ve iki aşamalı ince ayar (fine-tuning) pipeline'ı.

## İlgili Notebook
- [transfer-learning-mobilenet.ipynb](transfer-learning-mobilenet.ipynb)

## Temel Fonksiyonlar
- `keras.applications.MobileNetV2`
- `layers.GlobalAveragePooling2D`
- `layer.trainable = False` & Fine-Tuning
