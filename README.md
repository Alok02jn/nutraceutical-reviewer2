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
- Supports mg, mcg, g, and IU formats

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
- Retrieve scientifically relevant ingredient information
- Generate contextual AI review summaries
- Perform meaning-based ingredient search

---

## Semantic Ingredient Search

Searches ingredients using semantic meaning instead of exact keywords.

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
```

---

# Installation

Clone the repository:

```bash
git clone <your-github-repo-link>
cd <project-folder>
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

# Required Python Packages

Main libraries used:

- streamlit
- pandas
- numpy
- sentence-transformers
- faiss-cpu

---

# Dataset Setup

Place the dataset file:

```text
nih_supplement_formulations.csv
```

inside the same folder as:

```text
app.py
```

Your project structure should look like:

```text
project_folder/
│
├── app.py
├── nih_supplement_formulations.csv
├── requirements.txt
└── README.md
```

---

# Running the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

The app will automatically open at:

```text
http://localhost:8501
```

---

# Example Ingredient Input

```text
Melatonin 5 mg
Magnesium 300 mg
Ashwagandha 600 mg
L-Theanine 200 mg
```

---

# Example Semantic Search Queries

```text
sleep support
stress recovery
muscle performance
immune boost
anxiety support
```

---

# Deployment

The project can be deployed using:

- Streamlit Cloud
- Hugging Face Spaces
- Render
- Railway

---

# Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Embeddings | Sentence Transformers |
| Vector Database | FAISS |
| NLP Model | all-MiniLM-L6-v2 |
| Dataset | NIH Dietary Supplement Label Database |
| Language | Python |

---

# Future Improvements
- Ingredient interaction analysis
- Clinical evidence scoring
- PDF supplement label upload
- Personalized recommendation engine
- API deployment

---

# Disclaimer

This application is intended for:
- educational purposes
