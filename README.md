# Face Recognition System using PCA and ANN

This folder contains a complete, end-to-end implementation of a Face Recognition System using Principal Component Analysis (PCA) for dimensionality reduction (Eigenfaces) and Artificial Neural Networks (ANN) for classification.

The project is structured for academic submission and includes:
- A clean, optimized Python implementation (`face_recognition.py`)
- An academic project report (`report.md`)
- A presentation outline and viva questions (`presentation_and_viva.md`)

## Features
- **Eigenfaces Generation (PCA):** Reduces high-dimensional image data while keeping maximum variance.
- **Neural Network Classification:** Uses an `MLPClassifier` to map features to distinct identities.
- **Accuracy Evaluation:** Evaluates and plots the accuracy for different numbers of principal components ($k$).
- **Imposter Detection:** Uses Euclidean distance thresholding to reject unknown faces.

## Prerequisites

Ensure you have Python installed. Then, install the required dependencies:

```bash
pip install numpy opencv-python scikit-learn matplotlib scipy
```

## Project Structure

```text
FaceRecognitionProject/
│
├── dataset/                  # Directory for your face image dataset
│   ├── person1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── person2/
│       ├── img1.jpg
│       └── img2.jpg
│
├── results/                  # Directory where the accuracy graph will be saved
│
├── face_recognition.py       # Main Python script (Object-Oriented)
├── Face_Recognition_Experiment.ipynb # Jupyter Notebook version
├── generate_dummy_dataset.py # Script to capture custom dataset via webcam
├── requirements.txt          # Python project dependencies
├── run_project.bat           # Executable script to run the project easily (Windows)
├── report.md                 # Academic project report
├── presentation_and_viva.md  # Presentation outline & viva script
└── README.md                 # Project documentation and instructions
```

## Next Steps / How to Run

### Method 1: The Easy Way (Windows)
1. Double-click the `run_project.bat` file.
2. It will automatically install the necessary libraries and execute the main face recognition script.
3. If it tells you the dataset is empty, proceed to step 2 below to capture your face using your webcam.

### Method 2: Manual Execution

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate a Dataset (Optional but highly recommended for testing):**
   If you don't have images ready, use the built-in webcam script.
   ```bash
   python generate_dummy_dataset.py
   ```
   - Enter your name when prompted.
   - Look at your webcam; it will automatically capture 20 face images and save them in `dataset/your_name/`.
   - Run the script again with a friend or different name to build a larger database!

3. **Run the Main Script:**
   ```bash
   python face_recognition.py
   ```
   *(Alternatively, if you prefer an interactive environment, open `Face_Recognition_Experiment.ipynb` in Jupyter Notebook.)*

4. **View Results:**
   - The script will extract Eigenfaces and train the ANN model.
   - It will evaluate accuracies across different $k$ values (number of Eigenfaces) and output the metrics.
   - Furthermore, it will generate an `accuracy_graph.png` inside the `results/` folder visually showing the optimal $k$ value.

4. **Testing an Individual Image (Optional):**
   Place an image named `test.jpg` in the root `FaceRecognitionProject` folder alongside the script. When you run `face_recognition.py`, it will automatically attempt to recognize who it is, or reject the face as an Imposter if not found in the trained database.
