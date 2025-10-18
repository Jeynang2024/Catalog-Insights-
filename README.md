# Catalog-Insights-
# Product Price Prediction Using Text and Image Features

## Project Overview

This project focuses on predicting the **price of products** based on their textual descriptions and numeric features. The dataset contains product information including descriptions, bullet points, brand names, and image URLs. By combining **Natural Language Processing (NLP)** techniques with traditional machine learning, we aim to create a model that accurately predicts product prices.

---

## Dataset

The dataset consists of **three main fields**:  

- **`catalog_content`**: Contains textual information about each product, including:
  - `Item_name`: Name of the product, often containing the brand.  
  - `Bullet_Points`: Key features or highlights of the product.  
  - `Value`: Quantity or size of the product.  
- **`price`**: Selling price of the product.  
- **`url`**: Link to the product image.

**Feature extraction:**
- Brand Name → extracted from `Item_name`.  
- Description → extracted from `Bullet_Points`.  
- Size / Quantity → extracted from `Value`.  

For rows with missing size information, we leveraged **OCR on product images** to extract the size directly.

---

## Exploratory Data Analysis

- Distribution of prices was analyzed and categorized into **Low, Medium, and High** price segments.  
- Frequent words in `catalog_content` were analyzed for each price category, showing patterns like:
  - Low price: `artificial`, `basic`, etc.  
  - Medium & High price: `organic`, `premium`, `rich`, etc.  

 

---

## Feature Engineering

- **Text Features:**
  - TF-IDF embeddings were generated both for combined text and for separate columns (`Description`, `Product Name`, `Brand Name`).  
  - BERT embeddings (`distilbert-base-uncased`) were used to capture semantic meaning.  

- **Numeric Features:**
  - `Size`, `premium`, `artificial`, `certified` were included as numeric features.  

- **Combined Features:**  
  - TF-IDF or BERT embeddings were concatenated with numeric features for modeling.

---

## Modeling

We experimented with multiple approaches:  

1. **Random Forest Regressor**
   - Trained on TF-IDF embeddings of combined text.  
   - Trained again using TF-IDF embeddings of separate columns for better efficiency.
*![Example Image 1](image1.png)*  

2. **LightGBM + BERT**
   - Used BERT embeddings of combined text + numeric features.  
   - Dimensionality reduction using PCA was applied to speed up training.
*![Example Image 1](image2.png)*   

3. **LightGBM + BERT**
   - BERT embeddings reduced with PCA and combined with numeric features.  
   - LightGBM was used for faster training and higher efficiency.
*![Example Image 1](image3.png)*  

**Evaluation Metric:**  
- RMSE (Root Mean Squared Error) was used to evaluate model performance.

---

## Tools & Libraries

- **Python:** `pandas`, `numpy`, `scipy`, `sklearn`, `torch`, `transformers`, `tqdm`  
- **NLP:** TF-IDF, BERT (`distilbert-base-uncased`)  
- **Machine Learning:** Random Forest, LightGBM  
- **Image Processing:** OCR for extracting missing sizes  
- **Visualization:** Matplotlib, Seaborn  

---


