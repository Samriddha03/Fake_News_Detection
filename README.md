🚀 Fake News Detection using BERT + Explainable AI








📌 Project Overview

Fake news has become a major challenge in today’s digital world. This project uses BERT (Bidirectional Encoder Representations from Transformers) to classify news as Real or Fake with high confidence.

Transformer-based models like BERT are widely used because they capture deep contextual meaning in text, making them highly effective for NLP tasks like fake news detection .

🧠 Features

✔ BERT-based deep learning model
✔ High accuracy fake news classification
✔ Confidence score output
✔ Clean CLI interface (main.py)
✔ Training pipeline included (train.py)
✔ Google Colab notebook for experimentation
✔ Explainable AI support (LIME / SHAP ready)

🏗️ Project Structure
fake-news-detector/
│
├── main.py                  # Run prediction
├── train.py                 # Train model
├── requirements.txt        # Dependencies
├── README.md               # Documentation
├── training.ipynb          # Colab notebook
│
└── fake_news_bert_model/   # Trained model (download required)
📥 Download Trained Model

⚠️ The trained model is too large for GitHub.

👉 Download from Google Drive:

[Download BERT Model]: (https://drive.google.com/drive/folders/1eaTniXsC60Tq9YbZByuo-aLu9DuKquAf)

📌 After Download:
Extract the folder
Place it like this:
fake-news-detector/
│
├── fake_news_bert_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
⚙️ Installation
git clone https://github.com/Samriddha03/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
▶️ How to Run
python main.py
🧪 Example

Input:

The government announced a new economic policy aimed at reducing inflation.

Output:

{
  "label": "Fake",
  "confidence": 0.9996
}
🏋️ Model Training

To train the model:

python train.py

Or use the notebook:

training.ipynb
📊 Methodology
Text preprocessing using tokenizer
Fine-tuning pretrained BERT model
Binary classification (Real vs Fake)
Evaluation using accuracy & confidence

Many modern fake news systems rely on transformer architectures due to their superior contextual understanding compared to traditional ML models .

🔍 Explainable AI (XAI)

This project supports interpretability using:

LIME (Local Interpretable Model-Agnostic Explanations)
SHAP (SHapley Additive exPlanations)

👉 Helps understand why a prediction was made.

🚀 Future Improvements
Deploy as web app (Streamlit / Flask)
Real-time news verification API
Multilingual fake news detection
Integration with fact-checking APIs
📜 License

This project is licensed under the MIT License.

👨‍💻 Author

Samriddha Chakraborty

⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!