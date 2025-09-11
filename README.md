# 📈 Stock Return Prediction Dashboard  

Predict stock returns with deep learning — wrapped in a clean, interactive **Streamlit dashboard**.  

This project allows you to upload stock data (CSV/Excel), preprocess it, run predictions using a trained deep learning model, and visualize the results.  

---

## 🚀 Features  
- **Upload Data:** Supports CSV and Excel with `Date`, `Close`, `Volume`, etc.  
- **Preprocessing:** Handles missing values, normalization (carefully!), and feature engineering.  
- **Prediction:** Predicts **log returns**, converts them to actual returns.  
- **Visualization:** Line charts, top gainers, and top losers highlighted in the dashboard.  
- **Export:** Download predictions as CSV.  
- **Deployment:** Fully containerized with Docker.  

---

## 🗂️ Project Structure  
📦 stock-return-prediction
┣ 📂 src/ # Core model + training scripts
┣ 📂 pipeline/ # Preprocessing and prediction pipeline
┣ 📂 utils/ # Helper functions
┣ 📂 dashboard/ # Streamlit app
┣ 📂 models/ # Saved models
┣ 📄 requirements.txt
┣ 📄 Dockerfile
┣ 📄 README.md
┗ 📄 app.py


## SCREENSHOTS
<img width="944" height="391" alt="Screenshot 2025-09-11 200354" src="https://github.com/user-attachments/assets/7aa79c8e-1cd2-4d6f-8eab-68a7828d299d" />

<img width="882" height="437" alt="Screenshot 2025-09-11 200424" src="https://github.com/user-attachments/assets/613d0f2a-036a-4860-906b-e528df38b3cc" />

