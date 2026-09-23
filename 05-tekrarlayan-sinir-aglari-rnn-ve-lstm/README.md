# Tekrarlayan Sinir Ağları (RNN & LSTM) ile Dizi Modelleme

LSTM hücresi kapı denklemleri (Unutma, Girdi, Çıktı kapıları), kayan pencere (sliding window) tensör veri hazırlığı ve Keras çift katmanlı LSTM ile zaman serisi tahmini.

## İlgili Notebook
- [lstm-sequence-prediction.ipynb](lstm-sequence-prediction.ipynb)

## Temel Fonksiyonlar
- `layers.LSTM(units, return_sequences)`
- Kayan pencere `[samples, timesteps, features]`
- `Inverse scaling` ve MSE/MAE analizi
