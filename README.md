
# 🍽️ Nutri Track - AI-Powered Food Recognition & Nutritional Analysis

Nutri Track is a deep learning-based food classification and nutrition analysis system that empowers individuals to make informed dietary decisions. By analyzing food images using a CNN trained on the Food-101 dataset, Nutri Track provides accurate macronutrient estimations and personalized dietary recommendations through an interactive web interface.

---

## 🚀 Features

- 📷 **Image-Based Food Recognition**  
  Upload food images and get instant predictions with a Convolutional Neural Network (CNN) model.

- 📊 **Nutritional Analysis**  
  Automatically extract calorie and macronutrient data (carbs, proteins, fats, sugars) from recognized food items.

- 🧠 **Personalized Recommendations**  
  Google Generative AI is used to generate smart diet suggestions and answer health-related queries.

- 📅 **Dashboard & Tracking**  
  User-friendly dashboard with daily, weekly, and monthly dietary tracking features.

- 👨‍⚕️ **Support for Health Professionals**  
  Designed for dietitians and healthcare providers to monitor and guide clients effectively.

---

## 📂 Dataset

- **Food-101 Dataset**  
  - 101 food categories  
  - 101,000 images  
  - Publicly available: [Food-101 Dataset](https://paperswithcode.com/dataset/food-101)

---

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript, Bootstrap  
- **Backend**: Flask, SQLAlchemy, MySQL  
- **Deep Learning**: TensorFlow, Keras (CNN)  
- **AI Integration**: Google Generative AI (Gemini API)  
- **Database**: MySQL

---

## ⚙️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/nutri-track.git
   cd nutri-track
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**

   Create a `config.json` file:
   ```json
   {
     "GENAI_API_KEY": "your_google_genai_api_key"
   }
   ```

4. **Run the App**
   ```bash
   python app.py
   ```

5. **Access the App**  
   Open your browser and go to [http://localhost:5000](http://localhost:5000)

---


## 📈 Performance Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix used for performance evaluation on test images

## 🧩 Future Enhancements
- Real-time portion size estimation (using AR/depth cameras)
- Micronutrient analysis (vitamins, minerals)
- Integration with fitness trackers and health apps
- Multilingual support and voice-based input



## 📄 License
This project is licensed under the MIT License.

