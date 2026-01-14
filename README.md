# Advanced Resume Skill Extraction System

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NLP](https://img.shields.io/badge/NLP-spaCy-green.svg)](https://spacy.io/)

A professional-grade NLP pipeline designed to extract, normalize, and visualize technical skills from resume PDF documents. This system leverages `spaCy` for linguistic analysis and `pdfplumber` for high-fidelity text extraction.

## 🚀 Features

- **Hybrid Extraction Engine**: Combines `pdfplumber` and `PyMuPDF` for robust text recovery from complex PDF layouts.
- **NLP-Powered Heuristics**: Uses spaCy noun chunks and Named Entity Recognition (NER) to identify professional skills.
- **Skill Normalization**: Built-in mapping system to standardize nomenclature (e.g., "ML" -> "Machine Learning").
- **Benchmarking Suite**: Evaluates extraction accuracy (Precision, Recall, F1) against annotated gold-standard datasets.
- **Dimensionality Reduction**: Visualizes the skill vector space using Principal Component Analysis (PCA).

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Guyatu1627/advanced-resume-skill-extraction-nlp.git
   cd advanced-resume-skill-extraction-nlp
   ```

2. **Setup virtual environment**:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## 📂 Project Structure

- `resume_skill_extraction.py`: Main execution pipeline.
- `annotated_resumes.json`: JSONL dataset for cross-validation and benchmarking.
- `requirements.txt`: Project dependencies.
- `skill_space_pca.png`: Generated visualization of skills.

## 📊 Usage

Simply place your target resume as `test_resume.pdf` in the root directory and run:

```bash
python resume_skill_extraction.py
```

## 🧪 Evaluation

The system automatically performs a sample evaluation against the provided `annotated_resumes.json` dataset, providing a performance benchmark for the extraction logic.

---
*Developed with ❤️ for Advanced Agentic Coding.*
