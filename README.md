
# Network Intrusion Detection System (NIDS)

### Machine Learning-Based Network Traffic Classification Using Flask

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=for-the-badge)](https://network-intrusion-detection-system-fukn.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)

## Live Demo

**Try the application:** [Network Intrusion Detection System](https://network-intrusion-detection-system-fukn.onrender.com/)

The Network Intrusion Detection System (NIDS) is a machine learning-based web application that analyzes network traffic records and classifies them as **Normal** or **Attack**. It also provides estimated attack-family classifications and investigation recommendations to support further security analysis.

## Problem Statement

Network traffic may contain malicious activities that threaten the confidentiality, integrity, and availability of computer systems. Manually examining large volumes of network traffic is time-consuming.

This project demonstrates how traditional machine learning algorithms can help classify network traffic and support preliminary intrusion detection through a web interface.

## Features

- **Network Traffic Classification:** Classifies traffic records as Normal or Attack.
- **Batch CSV Prediction:** Analyzes multiple network traffic records through CSV upload.
- **Attack-Family Estimation:** Estimates the likely attack category using a separate classification model.
- **Prediction Explanations:** Provides decision-related information to help interpret predictions.
- **Investigation Recommendations:** Displays suggested follow-up actions based on prediction results.
- **CSV Export:** Allows batch prediction results to be downloaded for further analysis.
- **Input Validation:** Rejects invalid CSV files that do not meet the expected feature requirements.
- **Web-Based Interface:** Provides an interface for interacting with the detection models.

> Note: Attack-family predictions and recommendations are model-generated estimates, not confirmation that an attack occurred.

## Machine Learning Approach

The application uses traditional machine learning rather than a deep learning-based detection model.

The binary classification model distinguishes between Normal and Attack traffic. A separate model estimates attack families, including:

- DOS — Denial of Service
- PROBE — Probing and scanning activities
- R2L — Remote to Local
- U2R — User to Root

The project uses the NSL-KDD dataset for model development and evaluation.

Model performance should be assessed using independent test data, precision, recall, F1-score, and false-positive and false-negative counts. Rare attack categories may be harder to detect reliably.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Backend | Flask |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas |
| Model Serialization | Joblib |
| Frontend | HTML, CSS, JavaScript |
| Dataset | NSL-KDD |
| Deployment | Render |
| Version Control | Git and GitHub |

## System Architecture

1. **Input:** The user submits an individual network traffic record or uploads a CSV file.
2. **Validation:** The application validates the submitted data and checks the expected features.
3. **Preprocessing:** The input is prepared in the format expected by the trained model.
4. **Binary Classification:** The saved model predicts whether the traffic is Normal or Attack.
5. **Attack-Family Estimation:** A separate model estimates the likely attack family.
6. **Result Presentation:** Predictions, explanations, and recommendations are displayed.
7. **Export:** Batch analysis results can be downloaded as a CSV file.

## Project Structure

```text
Network-Intrusion-Detection-System/
├── dataset/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── models/
│   ├── decision_tree.joblib
│   └── attack_family.joblib
├── src/
│   ├── train.py
│   ├── train_attack_family.py
│   ├── evaluate.py
│   └── evaluate_attack_family.py
├── templates/
├── static/
├── app.py
├── requirements.txt
└── README.md
```

*The structure above represents the main project components; filenames and folders may vary slightly depending on the current repository version.*

## Run Locally

### Prerequisites

- Python 3.12 or a compatible Python version
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Palla-Umadevi/Network-Intrusion-Detection-System.git
cd Network-Intrusion-Detection-System
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the application

If the Flask entry point is `app.py`:

```bash
python app.py
```

Open the local address printed in the terminal, typically:

```text
http://127.0.0.1:5000/
```

The application requires the trained model files to be available at the paths expected by the Flask code.

## 📊 Evaluation

The project includes evaluation scripts for examining model performance on the NSL-KDD test dataset.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- False positives
- False negatives

These metrics help assess detection performance and identify cases where malicious traffic may be missed or normal traffic may be incorrectly flagged.

## Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- Performance on NSL-KDD data does not guarantee equivalent performance on modern production networks.
- Rare attack families may have lower detection recall.
- Model-generated predictions should be validated before taking security actions.
- This project is an educational intrusion detection demonstration, not a replacement for a production-grade network security monitoring system.

## Future Enhancements

- Integrate live network traffic capture.
- Add real-time traffic monitoring and alerting.
- Improve detection of rare attack categories.
- Add model performance visualizations.
- Introduce authentication and role-based access control.
- Evaluate the models against additional datasets.



- GitHub: [Palla-Umadevi](https://github.com/Palla-Umadevi)
- Project Repository: [Network Intrusion Detection System](https://github.com/Palla-Umadevi/Network-Intrusion-Detection-System)
- Live Demo: [Try the Application](https://network-intrusion-detection-system-fukn.onrender.com/)

---

⭐ If you find this project interesting, consider starring the repository!
