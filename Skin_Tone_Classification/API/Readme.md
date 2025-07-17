# 🎨✨ Skin Tone Prediction API — Powered by FastAPI & Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Fastest%20Python%20API-green?logo=fastapi)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-blue?logo=opencv)
![License](https://img.shields.io/github/license/Alok-2002/skin-tone-api?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/Alok-2002/skin-tone-api)
![Contributions](https://img.shields.io/badge/PRs-Welcome-ff69b4?logo=github)
![Issues](https://img.shields.io/github/issues/Alok-2002/skin-tone-api)
![Stars](https://img.shields.io/github/stars/Alok-2002/skin-tone-api?style=social)
![Last Commit](https://img.shields.io/github/last-commit/Alok-2002/skin-tone-api)
![Uptime](https://img.shields.io/badge/Uptime-100%25-brightgreen)
![Model](https://img.shields.io/badge/ML_Model-KNN%20Trained-orange)
![Code Style](https://img.shields.io/badge/Code%20Style-PEP8-informational)

> 📷 Upload your face or skin image → 🎯 Detect average skin tone → 🎨 Get the predicted skin tone and its HEX code.  
> Powered by **Machine Learning** and **OpenCV**, wrapped inside a clean **FastAPI** interface!

---

## 🚀 Key Features

- 🧠 AI-powered **Skin Tone Prediction**
- 🔬 Detects **average RGB** value from face/body region
- 🎨 Returns closest **HEX Code** & predefined **skin shade label**
- 📊 Uses **Euclidean distance** for accurate shade matching
- ⚡ Fast, asynchronous, developer-friendly API with **FastAPI**
- 🔁 Cleans up uploaded files automatically
- 📂 Clean, modular project structure

---

## 🗂️ Folder Structure

```bash
skin-tone-api/
│
├── main.py                      # 🔥 FastAPI app logic
├── requirements.txt             # 📦 Dependency list
│
├── Models/                      # 🧠 Trained ML model files
│   ├── skin_tone_model.pkl
│   └── label_encoder.pkl
│
├── Dataset/                     # 🎨 Skin shade references
│   └── skin_shades_india.csv
│
├── static/                      # 🖼️ (Optional) Sample image storage
│   └── example_image.jpg
│
└── README.md                    # 📘 You are here!
````

---

## ⚙️ Installation Guide

```bash
# 🚀 Clone the repository
git clone https://github.com/Alok-2002/skin-tone-api.git
cd skin-tone-api

# 🧪 Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 📦 Install required packages
pip install -r requirements.txt
```

---

## 🚦 Run the API Server

```bash
uvicorn main:app --reload
```

📍 Visit **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** for interactive Swagger UI

---

## 📡 API Endpoint – `/predict/`

### 🔸 `POST` request with an image:

* **Form field**: `file` (image file: JPG, PNG)
* **Returns JSON** with:

  * Predicted Skin Tone Label
  * HEX Color Code
  * Average RGB Value

📤 **Sample Response**:

```json
{
  "Predicted Skin Tone": "Light Skin",
  "Predicted HEX Code": "#F2C9B1",
  "Average RGB Color": [242, 200, 180]
}
```

---

## 🧠 Machine Learning Details

| Component         | Description                                 |
| ----------------- | ------------------------------------------- |
| 🎯 Model          | Trained on labeled average RGB skin samples |
| 📚 Algorithm      | K-Nearest Neighbors (KNN)                   |
| 🧩 Label Encoding | Encodes skin tone categories                |
| 🎯 Color Match    | Euclidean distance with HEX RGBs            |

---

## 🎨 Skin Shade Reference Table

| 🔢 ID   | 🧾 Description    | 🎨 HEX Code |
| ------- | ----------------- | ----------- |
| Shade 1 | Very Light Skin   | `#F5E0D8`   |
| Shade 2 | Light Skin        | `#F2C9B1`   |
| Shade 3 | Fair Skin         | `#EAB7A1`   |
| Shade 4 | Light Medium Skin | `#E3A58D`   |

---

## 🧰 Built With

* 🧠 `Scikit-learn` – ML modeling
* ⚡ `FastAPI` – Web framework
* 📸 `OpenCV` – Image processing
* 🐍 `Python 3.8+` - Python Language
* 📊 `Pandas / NumPy` – Data operations
* 💾 `Joblib` – Model serialization

---

## 📬 Sample cURL Request

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -F 'file=@your_image.jpg'
```

---

## 🛠️ Future Enhancements

* 📱 Mobile-ready skin tone prediction app (Flutter/React Native)
* 🌐 Streamlit Web App interface
* 🧪 Diverse & inclusive dataset training
* 🎯 Real-time video-based detection
* 🗃️ Color palette suggestions based on tone

---

## 🛡 License

📜 This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Maintainer

**Atul Sharma**
🔗 [LinkedIn](https://www.linkedin.com/in/atul-sharma2002)
💻 [GitHub](https://github.com/Alok-2002)

---

## 🌟 Show Your Support

If you found this helpful, give it a ⭐ on GitHub and share it with friends or developers who might benefit!

> *"Empowering personalized tech through color & code."* 🎨🧠
