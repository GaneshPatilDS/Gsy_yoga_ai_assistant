# 🧘‍♂️ Gsy_Yoga_AI_Assistant

An AI-powered assistant for **Govardhan School of Yoga** using **RAG (Retrieval Augmented Generation)** to answer yoga-related queries via a web app.

---

## 📑 How It Works  

1️⃣ **Vectorizing Documents (`vectorize_doc.py`)**  
- Reads PDFs from `Data/`
- Splits text into small chunks (to handle large files and improve retrieval)
- Converts chunks into vector embeddings using **TogetherEmbeddings**
- Stores them in **Chroma vector DB (`vector_db_dir_2/`)**

---

2️⃣ **Running the Assistant (`main.py`)**  
- Launch web app using:
  ```bash
  streamlit run main.py
