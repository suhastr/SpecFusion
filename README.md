# SpecFusion

SpecFusion is a machine learning and NLP project that detects semantic overlap between legacy and new requirements, and fuses them into concise, non-redundant statements.  
This helps reduce duplication and improve traceability in requirement management.

---

## 🚀 Features
- **Hybrid Similarity Pipeline**  
  - TF-IDF vectorization  
  - Sentence-BERT embeddings (all-MiniLM-L6-v2)  
  - RoBERTa cross-encoder (stsb-roberta-base)  
  - Fuzzy string similarity (RapidFuzz)  
  - Keyword Jaccard overlap  

- **Machine Learning Models**  
  - Ridge Regression  
  - ElasticNet  
  - Random Forest Regressor  
  - Trained and evaluated on **STS-Benchmark** for semantic similarity scoring  

- **Sentence Fusion**  
  - Rule-based + WordNet-informed integration of similar requirements  
  - Produces coherent fused requirement sentences  

- **Outputs**  
  - Integrated requirements exported as CSV  
  - Similarity scores and feature breakdown for transparency  

---

## ⚙️ Tech Stack
- Python 3.9+  
- scikit-learn  
- spaCy  
- SentenceTransformers  
- Hugging Face Transformers  
- RapidFuzz  
- NLTK / WordNet  

---

## 📂 Project Structure
├── en_m_3.py # Main pipeline script
├── legacy_requirements.txt # Sample input file
├── new_requirements.txt # Sample input file
├── integrated_output.csv # Example output file
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## 🧑‍💻 How It Works
1. **Input** two text files: `legacy_requirements.txt` and `new_requirements.txt`.  
2. **Preprocessing** with spaCy: tokenization, lemmatization, stopword handling.  
3. **Feature extraction**: TF-IDF, embeddings, cross-encoder, fuzzy ratios, Jaccard.  
4. **Similarity scoring** with trained regressors (Ridge, ElasticNet, RF).  
5. **Fusion**: if similarity exceeds threshold → merge into a coherent requirement.  
6. **Export** integrated requirements into a CSV file.  

---

## 📊 Example

**Legacy Requirement**  
> "Enable customers to track package delivery status in real time"  

**New Requirement**  
> "Provide live shipment updates and notifications through the mobile app"  

**Integrated Output**  
> "Provide real-time package tracking and shipment notifications via the mobile app."


---

## 🔬 Model Training
- **Dataset:** [STS-Benchmark](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark)  
- **Evaluation Metrics:** Pearson correlation, Spearman rank, MSE, MAE  
- **Feature Ensemble:** Weighted averaging and regression models  

---

## 📦 Installation & Usage
```bash
# Clone repo
git clone https://github.com/yourusername/reqalign.git
cd reqalign

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

## 🚀 How to Run

You can use the script in different modes depending on your workflow:

### 1. Hyperparameter Tuning
Run cross-validation on STS-Benchmark with model persistence:
```bash
python en_m_3.py --mode tune --train-limit 1000 --cv-folds 5 --save-model best_model.joblib --output results.json

python en_m_3.py --mode evaluate --load-model best_model.joblib --test-limit 500 --metrics-csv metrics.csv --pred-csv predictions.csv

python en_m_3.py --mode quick --sentence1 "Enable package tracking online" --sentence2 "Provide real-time shipment updates"


python en_m_3.py --file-a legacy_requirements.txt --file-b new_requirements.txt --integrated-csv integrated_output.csv --integrate-thr 4.0



### Run Flask based webapp
python en_m_3.py --serve --load-model tuned_model.joblib

### Run integration
python en_m_3.py legacy_requirements.txt new_requirements.txt




