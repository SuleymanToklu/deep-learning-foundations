# Evrişimli Sinir Ağları (CNN) ile Görüntü Sınıflandırma

Bu modül; 2 boyutlu uzamsal ızgaralara (görsellere) sahip verilerde öznitelik haritalarının çıkarılmasını sağlayan Konvolüsyon (Conv2D) ve Alt Örnekleme (MaxPooling2D) katmanlarının MNIST el yazısı rakam veri seti üzerindeki mimarisini açıklar.

## Notebook
- [digit-recognizer-project-with-cnn.ipynb](digit-recognizer-project-with-cnn.ipynb)

## Temel Kavramlar
- 2B Konvolüsyon çekirdeği ($3 \times 3$) ve filtre sayısı
- Havuzlama (`MaxPooling2D`) ile boyutsal küçültme
- Düzleştirme (`Flatten`) ve Karışıklığı Önleme (`Dropout`)
- MNIST doğruluğu ve model test başarımı
