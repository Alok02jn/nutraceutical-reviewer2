# Nutraceutical Formulation Reviewer

AI-powered nutraceutical formulation analysis platform built using:

- Streamlit
- Sentence Transformers
- FAISS Vector Search
- Retrieval-Augmented Generation (RAG)
- NIH Dietary Supplement Label Database

The system analyzes supplement formulations, evaluates dosage safety, detects suspicious marketing claims, and retrieves scientifically relevant ingredient information using semantic vector search.

---

# Features

## AI-Powered Ingredient Analysis
- Parses supplement formulations automatically
- Extracts ingredient names, dosage values, and units
- Supports mg, mcg, g, IU, and CFU formats

---

## Safety & Dosage Evaluation
The system compares ingredient dosages against:
- NIH reference values
- FDA upper intake limits
- Common nutraceutical safety thresholds

Outputs:
- Safe
- Near Upper Limit
- High-Risk Dosage

---

## Retrieval-Augmented Generation (RAG)
The application uses:
- SentenceTransformer embeddings (`all-MiniLM-L6-v2`)
- FAISS vector database
- Semantic retrieval from NIH supplement data

This allows the system to:
- Retrieve scientifically related ingredient information
- Generate contextual AI review summaries
- Perform meaning-based ingredient search

---

## Semantic Ingredient Search
Searches ingredients using meaning instead of keywords.

Example queries:
- sleep support
- immune boost
- stress recovery
- muscle performance
- anxiety support

---

## Marketing Claim Detection
Detects potentially misleading supplement claims such as:
- “Miracle cure”
- “Guaranteed results”
- “No side effects”
- “Clinically proven”
- “FDA approved”
- Disease reversal claims

---

## Safety Score
The system automatically calculates:
- formulation risk score
- number of safety flags
- dosage severity

---

# Project Structure

```text
project_folder/
│
├── app.py
├── nih_supplement_formulations.csv
├── requirements.txt
└── README.md