# Encrypted Network Traffic Protocol Fingerprinting Using Machine Learning

## 1. Project Overview

This project implements a machine learning model for encrypted network traffic protocol fingerprinting.  
The main goal is to classify network flows into different application or protocol categories using only flow-level statistical features.

The model does **not** inspect packet payload content.  
Instead, it relies on features such as packet size, packet count, flow duration, bytes per second, packets per second, and packet inter-arrival time.

This approach is useful for analyzing encrypted traffic while preserving payload privacy.

---

## 2. Task Description

### Task Name

**Network Traffic Protocol Fingerprinting**

### Challenge

Implement a machine learning model that classifies encrypted network flows into application or protocol categories such as:

- HTTP / HTTPS or Browsing
- DNS
- QUIC
- VoIP
- BitTorrent / P2P
- Streaming
- Chat

The classification must be done using only packet timing and size statistics, without accessing payload content.

---

## 3. Core AI Technique

The main machine learning model used in this project is:

- **Random Forest Classifier**

Random Forest was selected because it works well with tabular data, handles non-linear patterns, and can provide feature importance analysis.

Possible extension:

- **XGBoost Classifier**

---

## 4. Dataset

The recommended dataset for this project is:

**ISCX VPN-nonVPN Dataset / ISCXVPN2016**

Alternative dataset:

**CICIDS2017**

The dataset should contain flow-level statistical features extracted from network traffic.  
If the original files are in PCAP format, features can be extracted using tools such as CICFlowMeter.

---

## 5. Success Metrics

The model should aim to achieve:

| Metric | Target |
|---|---:|
| Accuracy | Greater than 92% |
| Macro F1-score | Greater than 0.88 |

Other reported metrics include:

- Precision
- Recall
- Classification report
- Confusion matrix
- Cross-validation mean ± standard deviation

---

## 6. Project Structure

```text
network-traffic-fingerprinting/
│
├── data/
│   └── traffic_features.csv
│
├── results/
│   ├── confusion_matrix_test_full_model.png
│   ├── confusion_matrix_test_without_timing_features.png
│   └── random_forest_model.pkl
│
├── main.py
├── requirements.txt
└── README.md
