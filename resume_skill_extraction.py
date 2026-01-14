"""
Advanced Resume Skill Extraction System
=======================================

This module provides a robust pipeline for extracting and normalizing professional skills
from resume PDF files using spaCy and heuristics. It includes evaluation against
annotated datasets and visualization of the skill space.

Author: Antigravity AI
Version: 1.0.0
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np
import pdfplumber
import spacy
from sklearn.decomposition import PCA
from spacy.language import Language

# Professional logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SkillExtractor")

# Load spaCy model (small & fast for resumes)
try:
    nlp: Language = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    sys.exit(1)

# ── Extraction Logic ─────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts plain text from a PDF file using pdfplumber with a fallback to PyMuPDF.

    Args:
        pdf_path (str): The absolute or relative path to the PDF file.

    Returns:
        str: The extracted text content.
    """
    text: str = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logger.warning(f"pdfplumber failed for {pdf_path}: {e} → falling back to PyMuPDF")
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        except Exception as fallback_err:
            logger.error(f"Critical failure: Unable to extract text from {pdf_path}. Error: {fallback_err}")
            return ""
    
    clean_text = text.strip()
    logger.info(f"Successfully extracted {len(clean_text)} characters from {pdf_path}")
    return clean_text


def extract_skills_spacy(text: str) -> List[str]:
    """
    Identifies potential skills using spaCy's linguistic features and rule-based heuristics.

    Args:
        text (str): The resume text to analyze.

    Returns:
        List[str]: A list of raw extracted skill entities.
    """
    doc = nlp(text.lower())
    skills_found: Set[str] = set()
    
    # Heuristic 1: Noun chunks filtered by keywords
    skill_keywords = {'skill', 'python', 'java', 'sql', 'aws', 'excel', 'machine learning', 
                     'data', 'engineering', 'cloud', 'devops', 'automation'}
    
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        # Keep short phrases containing relevant keywords
        if len(chunk_text.split()) <= 4 and any(word in chunk_text for word in skill_keywords):
            skills_found.add(chunk_text.title())
    
    # Heuristic 2: Entity Recognition for Organizations and Technologies
    # (Note: ORG often captures specific software or frameworks in resumes)
    labels_to_extract = {'ORG', 'PRODUCT', 'NORP', 'LANGUAGE'}
    for ent in doc.ents:
        if ent.label_ in labels_to_extract:
            skills_found.add(ent.text.strip().title())
    
    logger.info(f"Identified {len(skills_found)} unique potential skill entities")
    return sorted(list(skills_found))


def normalize_skills(skills: List[str]) -> List[str]:
    """
    Normalizes and deduplicates skills into standard professional nomenclature.

    Args:
        skills (List[str]): Raw skill strings extracted from the text.

    Returns:
        List[str]: Sorted list of normalized skills.
    """
    normalization_map = {
        'python programming': 'Python',
        'machine learning': 'Machine Learning',
        'ml': 'Machine Learning',
        'data analysis': 'Data Analysis',
        'data science': 'Data Science',
        'sql database': 'SQL',
        'aws cloud': 'AWS',
        'microsoft azure': 'Azure',
        'nlp': 'Natural Language Processing'
    }
    
    normalized: List[str] = []
    seen: Set[str] = set()
    
    for skill in skills:
        s_norm = normalization_map.get(skill.lower(), skill)
        if s_norm.lower() not in seen:
            seen.add(s_norm.lower())
            normalized.append(s_norm)
    
    return sorted(normalized)


# ── Evaluation & Visualization ───────────────────────────────────────────────

def evaluate_extraction(pred_skills: List[str], gold_skills: List[str]) -> Tuple[float, int, int]:
    """
    Calculates Entity-level Precision, Recall, and F1-score against ground truth.

    Args:
        pred_skills (List[str]): Predicted skills extracted by the system.
        gold_skills (List[str]): Ground truth (annotated) skills.

    Returns:
        Tuple[float, int, int]: F1 score, True Positives, and Total Gold count.
    """
    pred_set = {s.lower() for s in pred_skills}
    gold_set = {s.lower() for s in gold_skills}
    
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    logger.info(f"Performance Metrics - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    return f1, tp, len(gold_set)


def visualize_skill_clusters(X: np.ndarray, labels: Optional[np.ndarray] = None) -> None:
    """
    Generates a 2D PCA plot of the skill/resume vector space.

    Args:
        X (np.ndarray): High-dimensional vector data (e.g., embeddings).
        labels (Optional[np.ndarray]): Optional cluster labels for coloring.
    """
    if X.shape[1] < 2:
        logger.warning("Feature dimension too low for PCA visualization")
        return
    
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 7), dpi=100)
    if labels is not None:
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', s=60, alpha=0.8)
        plt.colorbar(scatter, label='Semantic Cluster')
    else:
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], color='#2c3e50', alpha=0.6, s=40)
    
    plt.title('Resume Skill Vector Space (Truncated SVD/PCA)', fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=10)
    plt.ylabel('Principal Component 2', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_path = 'skill_space_pca.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Visualization saved to: {output_path}")


# ── Main Execution Flow ──────────────────────────────────────────────────────

def run_pipeline() -> None:
    """Orchestrates the resume extraction and evaluation pipeline."""
    logger.info("Starting Advanced Resume Skill Extraction System")
    
    # 1. Load annotated training data (JSONL format)
    annotated_path = "annotated_resumes.json"
    annotated_data: List[Dict[str, Any]] = []
    
    if Path(annotated_path).exists():
        try:
            with open(annotated_path, 'r', encoding='utf-8') as f:
                annotated_data = [json.loads(line) for line in f if line.strip()]
            logger.info(f"Loaded {len(annotated_data)} annotated resumes for evaluation")
        except Exception as e:
            logger.error(f"Failed to parse {annotated_path}: {e}")
    else:
        logger.warning(f"Metadata file '{annotated_path}' not found - skipping cross-validation")
    
    # 2. Process Target Resume
    test_pdf = "test_resume.pdf"
    if not Path(test_pdf).exists():
        logger.error(f"Target PDF '{test_pdf}' not found. Please provide a sample resume.")
        sys.exit(1)
    
    raw_text = extract_text_from_pdf(test_pdf)
    if not raw_text:
        logger.error("Empty extraction result. Aborting.")
        return

    extracted_raw = extract_skills_spacy(raw_text)
    normalized = normalize_skills(extracted_raw)
    
    results = {
        "source": test_pdf,
        "skill_count": len(normalized),
        "top_skills": normalized
    }
    
    print("\n--- Extraction Results ---")
    print(json.dumps(results, indent=2))
    print("--------------------------\n")
    
    # 3. Evaluation against gold standard (demo)
    if annotated_data:
        # Extract skills from the first gold sample for a comparison demo
        first_sample_annotations = annotated_data[0].get('annotation', [])
        gold_skills: List[str] = []
        for ann in first_sample_annotations:
            if 'Skills' in ann.get('label', []):
                for p in ann.get('points', []):
                    gold_skills.append(p.get('text', ''))
        
        if gold_skills:
            logger.info("Conducting benchmark evaluation...")
            evaluate_extraction(normalized, gold_skills)
    
    # 4. Visualization (Simulated embeddings for layout demo)
    # Note: In a production setting, these would be generated by a Transformer model
    simulated_vectors = np.random.rand(50, 64)
    visualize_skill_clusters(simulated_vectors)
    
    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    run_pipeline()