# 🧾 Receipt Extractor

***

## 👤 Projektinformationen

| **Autor** | thedawid999 |
| :--- | :--- |
| **Studiengang** | Angewandte Künstliche Intelligenz |
| **Projekt/Modul** | Projekt – Vom Modell zum Produktivsystem |

***

## 🌟 Projektziel

Ziel dieses Projekts ist die Entwicklung eines Systems zur automatischen Extraktion relevanter Informationen aus Belegen und Rechnungen.

Im Mittelpunkt steht die Entwicklung einer vollständigen Machine-Learning-Pipeline bestehend aus:

- Bildvorverarbeitung
- OCR-basierter Texterkennung
- Informationsextraktion mittels LayoutLMv3
- Batch-Verarbeitung
- Automatisierung über Scheduler
- Bereitstellung über eine REST-API

Das Projekt zeigt, wie moderne KI-Modelle zur Dokumentenanalyse in eine produktionsreife Pipeline integriert werden können, um Buchhaltungsprozesse zu automatisieren.

***

## 📊 Datensatz

**Quelle:** SROIE datasetv2 (Scanned Receipts OCR and Information Extraction)

**Trainingsdaten:** 620 Belege

**Testdaten:** 150 Belege

**Link:** https://www.kaggle.com/datasets/urbikn/sroie-datasetv2/data

Der Datensatz enthält gescannte Kassenbelege inklusive OCR-Bounding-Boxes sowie den gelabelten Entitäten:

- COMPANY
- ADDRESS
- DATE
- TOTAL

***

## 🏗️ Systemarchitektur

Das System wurde modular aufgebaut und besteht aus mehreren unabhängigen Komponenten.

```
Client
   │
   ▼
FastAPI
   │
   ▼
Upload
   │
   ▼
Bildvorverarbeitung
   │
   ▼
EasyOCR
   │
   ▼
LayoutLMv3
   │
   ▼
Postprocessing (BIO + Regex)
   │
   ▼
JSON-Ausgabe
```

Die Anwendung kann sowohl lokal als CLI als auch als REST-Service betrieben werden.

***

## 📂 Projektstruktur

```text
receipt-extractor/
├── layoutlm/
│   └── dataset/
│       └── train/
│       └── test/
│   └── layoutlmv3-final-v1/          # manuell gespeicherter fine-getunter LayoutLMv3 Model und Processor
│   └── layoutlmv3-finetuned-sroie/   # automatisch generiert nach dem Training (enthält checkpoints und runs)
│   └── dataset.py                    # Vorbereitung des SROIE Datensatzes für LayoutLMv3 Training
│   └── training.py                   # Training und Evaluierung
│   └── labels.json                   # enthält Labels und die dazugehörigen IDs
│
├── src/
│   └── api.py                        # RESTful API
│   └── layoutlm.py                   # zuständig für die Vorhersage des fine-getuntes LayoutLMv3 Model
│   └── main.py                       # für lokale Anwendung
│   └── ocr.py                        # extrahiert den Text mit EasyOCR
│   └── preprocessing.py              # Vorverarbeitung des Bildes
│   └── scheduler.py                  # automatische Batch-Verarbeitung
│   └── utils                         # Hilfsmethoden
│   └── config.json                   # Einstellungen für das Programm
│
├── requirements.txt
└── README.md
```

***

## 🛠️ Datenvorverarbeitung

Die Vorverarbeitung umfasst mehrere Schritte.

### 📥 Datenaufnahme

- Überprüfung des Eingabepfades
- Upload einzelner Bilder oder kompletter Ordner

### 🖼️ Bildvorverarbeitung

- Graustufen erzeugt
- Bilder automatisch rotiert

### 🔤 OCR mit EasyOCR

- Erkennung des Texten
- Berechnung der Bounding Boxes
- Berechnung der Confidence

Alle Textfelder mit einer Confidence kleiner als **0,4** werden verworfen.

***

## 🧩 Erstellung des Trainingsdatensatzes

Da der originale SROIE-Datensatz nicht direkt zum Fine-Tuning von LayoutLMv3 verwendet werden kann, wurde zunächst ein eigener Trainingsdatensatz erzeugt.

Hierfür wurden mehrere Verarbeitungsschritte implementiert:

- Einlesen der Bounding-Box-Dateien
- Einlesen der Entity-Dateien
- Zusammenführen beider Dateien
- Zuordnung der Labels
- Aufteilung von Bounding Boxes in einzelne Wörter
- Hinzufügen von BIO-Tags
- Speicherung als finales Trainingsdataset

***

## 🤖 Modelltraining

Zur Informationsextraktion wurde **LayoutLMv3** verwendet.

Das Modell wurde auf dem selbst erzeugten Datensatz fine-getunt.

Während des Trainings wurden zusätzlich folgende Metriken berechnet:

- Accuracy (0.962)
- F1-Score (0.769)
- Loss (0.107)
- Precision (0.744)
- Recall (0.816)

***

## 🌐 REST API

Die Anwendung wurde mit **FastAPI** bereitgestellt.

Unterstützt werden:

- Upload einzelner Belege
- Upload kompletter Ordner
- Verarbeitung mehrerer Dateien
- JSON-Ausgabe

Dadurch lässt sich die Anwendung problemlos in andere Systeme integrieren.

### Beispiel

#### Request

```http
POST /predict
```

```config.json
{
    "input_dir": "./uploads",
    "output_dir": "./outputs",
    "output_file": "results",
    "schedule": {
        "hour": 9,
        "minute": 45
    }
}
```

#### Response

```json
{
    "company": "TESCO STORES",
    "date": "04/02/2017",
    "address": "NO.12 JALAN ABC",
    "total": "128.90"
}
```

***

## ⚙️ Installation

Repository klonen

```bash
git clone https://github.com/thedawid999/Receipt-Extractor.git
```

Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

main.py starten

```bash
python -m src.main
```

RESTful API starten

```bash
uvicorn src.api:app --reload
```

***

## 🔬 Verwendete Technologien

- Python 3.12.4 (Visual Studio Code)
- FastAPI
- LayoutLMv3
- Kaggle
- EasyOCR
- OpenCV
- Pandas
- NumPy
- APScheduler
- Regular Expressions (Regex)

***

## 📚 Quellen

- https://www.kaggle.com/datasets/urbikn/sroie-datasetv2/data
- https://www.kaggle.com/code/urbikn/layoutlm-using-the-sroie-dataset
- https://colab.research.google.com/drive/1laRdh8CMWtaMClX9BA0F6nWn8aFauYQl
- https://arxiv.org/pdf/2204.08387
