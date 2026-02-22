cat << 'EOF' > README.md
# 🧠 AI vs Human Text Detection using XLM-RoBERTa

This project builds a deep learning classifier to distinguish between **human-written** and **AI-generated** text using a transformer-based model.

The model achieves **~99% validation accuracy**.

---

## 🚀 Project Overview

With the rapid growth of AI-generated content, detecting machine-written text has become increasingly important in:

- Academic integrity
- Journalism & media
- Content moderation
- Marketing authenticity
- AI governance

This project leverages XLM-RoBERTa, a multilingual transformer model, to classify text as:

- 🧑 Human-written  
- 🤖 AI-generated  

---

## 📂 Dataset

Dataset used:

**AI vs Human Content Detection Dataset 2026**

- Total samples: 686
- Languages: English, Hindi, Urdu, Arabic, Spanish, French, Code-mixed
- Domains: Social Media, Marketing, Technical Blog, Email, News, Education

### Original Classes:
- human
- ai
- post_edited_ai

### Final Setup (Binary Classification)

post_edited_ai → ai

Final classes:
- Human (0)
- AI (1)

---

## 🛠 Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Matplotlib & Seaborn
- NumPy & Pandas

Model used:

xlm-roberta-base

---

## 📊 Exploratory Data Analysis

Performed analysis on:

- Target distribution
- Edit levels
- Languages
- Domains
- Source models
- Word count distribution

Key observations:
- Moderate class imbalance (AI dominant)
- Word counts range from 9 to 1121 words

---

## 🧹 Data Cleaning

- Removed corrupted samples starting with "Error"
- Checked for missing values
- Encoded labels into numeric format

---

## ✂️ Train / Validation Split

- 80% Training
- 20% Validation
- Stratified sampling to preserve class distribution

---

## 🔤 Tokenization

Tokenizer:

AutoTokenizer.from_pretrained("xlm-roberta-base")

- Max length: 128
- Padding & truncation enabled

---

## 🤖 Model Architecture

Pretrained Transformer:
XLM-RoBERTa Base

Classification Setup:
- Output labels: 2
- Loss: CrossEntropyLoss
- Optimizer: AdamW (lr=2e-5)
- Gradient clipping: max_norm=1.0

Total Trainable Parameters:
~278 Million

---

## 🏋️ Training Strategy

- Batch size: 16
- Epochs: 30
- Early stopping patience: 6
- Best model saved based on validation loss

Early stopping triggered at epoch 9.

---

## 📈 Results

Validation Accuracy: ~99%

Confusion Matrix:

Actual \ Predicted | Human | AI
-------------------|-------|----
Human              | 36    | 0
AI                 | 2     | 98

Classification Report:

- Human F1-score: 0.97
- AI F1-score: 0.99
- Overall Accuracy: 0.99

The model demonstrates strong performance in distinguishing AI-generated text from human-written content.

---

## ⚠️ Limitations

- Small dataset (686 samples)
- Possible slight overfitting
- No external test set evaluation
- Real-world generalization not verified

---

## 🔮 Future Improvements

- Cross-validation
- Larger dataset
- External benchmark testing
- Learning rate scheduling
- Try larger models (e.g., xlm-roberta-large)

---

## ▶️ Installation

Install dependencies:

pip install torch transformers pandas numpy scikit-learn matplotlib seaborn tqdm kagglehub

---

## 📌 Key Learnings

- Transformer models perform exceptionally well on NLP classification tasks.
- Early stopping helps prevent overfitting.
- Multilingual transformers handle diverse datasets effectively.
- Clean preprocessing significantly improves performance.

---

## 🧪 Training Environment

The model was trained on Kaggle using GPU acceleration.

Kaggle Notebook Link:
https://www.kaggle.com/code/nimasaghi/ai-human-text-detection-xlm-roberta-99-accuracy

---

## 🙌 Support

If you found this project useful:

⭐ Star this repository  
📢 Share it  
💬 Provide feedback  

---

# 🚀 Final Thoughts

As AI-generated content becomes more widespread, detection systems will be critical for maintaining trust and authenticity in digital communication.

This project demonstrates how transformer-based models can effectively tackle AI text detection.

EOF
