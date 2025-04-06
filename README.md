---
title: Twitter Sentiment Analysis
emoji: 🐠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.17.1
app_file: app.py
pinned: false
license: mit
short_description: Sentiment Analysis Project trained on Twitter Data
---

## 🚀 **Key Features**
- **Text Sentiment Classification:** Predicts whether a given text expresses a positive or negative sentiment.
- **Real-Time Predictions:** Provides instant results via a simple web interface.
- **Visual Performance Metrics:** Displays accuracy and loss curves for model evaluation.
- **User-Friendly Interface:** Built with **Gradio** for easy accessibility.

---

## 🛠️ **Technology Stack**
- **Deep Learning:** TensorFlow Keras
- **Text Tokenization:** Keras Tokenizer
- **Web Interface:** Gradio
- **Data Manipulation:** Pandas
- **Visualization:** Matplotlib

---

## 📁 **File Structure**
```plaintext
sentiment-analysis-app
│
├── app.ipynb             # Main application notebook
├── requirements.txt      # Required packages
└── README.md             # Project documentation
```

---

## 💾 **Usage**
1. **Train the Model:** Load the dataset, preprocess text, and train the LSTM model.
2. **Visualize Performance:** View accuracy and loss plots for training and validation sets.
3. **Predict Sentiment:** Enter a text review in the Gradio interface and receive the predicted sentiment.

---

## 📦 **Dependencies**
```plaintext
pandas
matplotlib
tensorflow
gradio
```

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## 🧠 **How It Works**
1. **Data Loading:** The dataset is loaded using **Pandas** and filtered to exclude neutral sentiments.
2. **Text Tokenization:** Text is tokenized using Keras Tokenizer with a vocabulary size of **5000**.
3. **Sequence Padding:** Tokenized sequences are padded to ensure uniform input length of **200**.
4. **LSTM Model:** A sequential model with an embedding layer, LSTM layer, and dense output layer is trained.
5. **Prediction:** Given an input text, the model predicts a sentiment label (**positive** or **negative**).

---

## 🌐 **Access Live Demo**
🔗 [Sentiment Analysis](https://huggingface.co/spaces/ajnx014/twitter-sentiment-analysis)

---

## 📝 **License**
This project is licensed under the **MIT License**.

---

## 🤝 **Contributing**
Contributions are welcome! Feel free to fork the repository, create a new branch, and submit a pull request.

---

## 📧 **Contact**
For inquiries or support, please reach out to **[arjunjagdale14@gmail.com](mailto:arjunjagdale14@gmail.com)**.

---

> **Author:** Arjun Jagdale  
> **GitHub:** [ArjunJagdale](https://github.com/ArjunJagdale)  
> **Project:** Sentiment Analysis with LSTM
