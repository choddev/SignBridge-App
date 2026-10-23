# 🤟 SignBridge — Web & Android Application (YOLOv11)

Türk İşaret Dili ↔ Türkçe çift yönlü gerçek zamanlı çeviri arayüzü (**Web** & **Android**), **YOLOv11 (`yolo11n.pt`)** modeli ile birlikte.

## 📁 Proje Yapısı

- [`models/yolo11n.pt`](models/yolo11n.pt) & [`models/data.yaml`](models/data.yaml): YOLOv11 el/kişi ve işaret dili modeli
- [`backend/app.py`](backend/app.py): Flask + Ultralytics YOLOv11 çıkarım sunucusu (`/api/predict`, `/api/info`, `/api/health`)
- [`web/`](web/): SignBridge çift yönlü Web uygulaması (**Kamera** & **Avatar** sekmeleri, Türkçe NLP cümle motoru, TTS/STT, 122 kelimelik sözlük)
  - [`web/src/`](web/src/): Kaynak dosyalar (`index.html`, `styles.css`, `app.js`, `manifest.json`)
  - [`web/dist/`](web/dist/): Derlenmiş Web uygulaması (`python web/build.py`)
- [`android/`](android/): SignBridge Android mobil uygulaması (.NET 9 Android / APK)
  - [`android/MainActivity.cs`](android/MainActivity.cs): Android WebView + `SignBridgeNative` YOLO köprüsü
  - [`android/Assets/mobile_ui.html`](android/Assets/mobile_ui.html): Mobil **Kamera** ve **Avatar** arayüzü

## 🚀 Çalıştırma ve Derleme

### 1. Web Uygulamasını Derleme ve Başlatma
```bash
python web/build.py
python backend/app.py
```
Tarayıcıda `http://localhost:5000` adresini açın.

### 2. Android Uygulamasını Derleme (APK)
```bash
dotnet build android/SignBridgeAndroid.csproj -c Release
```
