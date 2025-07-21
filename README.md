
# SignSense: Real-Time Sign Language Translator 🤟

SignSense is a Python application that uses computer vision and machine learning to translate American Sign Language (ASL) gestures into text in real-time. Built with a MediaPipe backend and a Streamlit frontend, it provides an intuitive and seamless communication bridge.

<p align="center">
<img src="https://i.ibb.co/TDDHP4gH/Screenshot.png" alt="Screenshot" border="0">
<img src="https://i.ibb.co/gbFWmxkB/demo.png" alt="demo" border="0">
</p>

## 🌟 Key Features

* **Real-Time Translation**: Translates gestures from a live webcam feed with low latency (<0.3s).
* **High Accuracy**: The Random Forest model achieves over 92% accuracy on the test dataset.
* **Comprehensive Gesture Support**: Recognizes 40 distinct gestures, including the ASL alphabet (A-Z), numbers (0-9), and common phrases like "I Love You", "OK", "HELP", and "EAT".
* **Interactive UI**: A simple and clean user interface built with Streamlit that allows users to easily start and stop the camera for translation.

---

## 🛠️ Tech Stack

* **Core Logic**: Python
* **Computer Vision**: OpenCV
* **Hand Tracking**: MediaPipe
* **Machine Learning**: Scikit-learn
* **Web App Framework**: Streamlit
* **Data Handling**: NumPy, Pickle
* **UI Enhancements**: Streamlit Lottie

---

## ⚙️ How It Works

The project follows a complete machine learning pipeline from data collection to deployment:

1.  **Data Collection**: The `scripts/collect_imgs.py` script captures hundreds of images for each of the 40 sign language gestures using a webcam.
2.  **Dataset Creation**: The `scripts/create_dataset.py` script processes the collected images. It uses MediaPipe to detect hand landmarks, normalizes the coordinates to ensure scale and position invariance, and saves the processed data into a single `data.pickle` file.
3.  **Model Training**: The `scripts/train_classifier.py` script loads the dataset, trains a **Random Forest Classifier** model, and saves the final trained model as `model/model.p`.
4.  **Real-Time Inference**: The main application, `app.py`, loads the trained model. It captures video from the webcam, performs the same landmark extraction and normalization in real-time, and feeds the data to the model to predict the corresponding gesture, displaying the result on screen.

---

## 📂 Project Structure

The project is organized into distinct folders for clarity and scalability.

```

SignSense/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                     \# Main Streamlit application
│
├── assets/                    \# All UI files
│   ├── Animation/
│   │   └── Ani.json
│   └── image/
│       └── sign.png
│
├── model/                     \# Trained model file
│   └── model.p
│
└── scripts/                   \# Model pipeline scripts
├── collect\_imgs.py
├── create\_dataset.py
└── train\_classifier.py

````

---

## 🚀 Setup and Usage

To run this project locally, follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/SignSense.git](https://github.com/your-username/SignSense.git)
cd SignSense
````

**2. Create and activate a virtual environment:**

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install the required dependencies:**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit Application:**

```bash
streamlit run app.py
```

Your web browser will open with the SignSense application running. Click "Open Camera\!" to begin.

-----

## 🔄 How to Retrain the Model

If you wish to collect your own data and retrain the model, follow these steps in order:

1.  **Collect Data**: Run the data collection script. It will create a `data/` directory and save the images there.
    ```bash
    python scripts/collect_imgs.py
    ```
2.  **Create Dataset**: Run the dataset creation script. This will process the images in `data/` and create `data.pickle`.
    ```bash
    python scripts/create_dataset.py
    ```
3.  **Train Model**: Run the training script. This will use `data.pickle` to train a new classifier and save it as `model/model.p`.
    ```bash
    python scripts/train_classifier.py
    ```

Once complete, the main application will automatically use your newly trained model.

