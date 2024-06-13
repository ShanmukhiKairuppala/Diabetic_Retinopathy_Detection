# DeepVision: Automated Diabetic Retinopathy Evaluation


## Overview
This project aims to detect and grade the level of Diabetic Retinopathy (DR) in retinal images using advanced machine learning techniques. DR is a complication of diabetes that affects the eyes and can lead to blindness if not detected and treated early. It is one of the leading causes of vision impairment among working-age adults globally.

### DR Levels
- **0**: No DR (No Diabetic Retinopathy)
- **1**: Mild DR (Mild Non-Proliferative Retinopathy)
- **2**: Moderate DR (Moderate Non-Proliferative Retinopathy)
- **3**: Severe DR (Severe Non-Proliferative Retinopathy)
- **4**: Proliferative DR (Proliferative Retinopathy)

## Importance of Detecting Diabetic Retinopathy
- **Prevalence**: Diabetic Retinopathy affects nearly one-third of the diabetic population, with millions at risk of vision loss.
- **Early Detection**: Early detection through regular screening can prevent 90% of diabetes-related vision loss.
- **Impact**: DR can lead to severe complications, including blindness, if left untreated. It significantly affects the quality of life and imposes a substantial economic burden due to treatment costs and loss of productivity.

### Affected Population

- **Diabetics**: All individuals with diabetes are at risk, making up a significant portion of the global population.
- **Elderly**: Older adults with diabetes are at a higher risk due to the prolonged duration of diabetes.
- **Underserved Communities**: Populations with limited access to healthcare services are less likely to undergo regular eye screenings, increasing their risk.

## Project Implementation

The project uses a pre-trained InceptionV3 model to classify the DR levels into five categories: No DR, Mild DR, Moderate DR, Severe DR, and Proliferative DR. The project encompasses the following steps:

### Dataset

We used the APTOS dataset, which contains labeled images indicating different levels of Diabetic Retinopathy (DR).

### Data Preprocessing
- Images are resized to 256x256 pixels.
- Normalized to a [0, 1] range.
- Enhanced using Contrast Limited Adaptive Histogram Equalization (CLAHE).

### Data Augmentation
- Rotations (90, 120, 180, and 270 degrees).
- Horizontal flips to increase the dataset size and improve model generalization.

### Model Training
- **Initial Training**: Used InceptionV3 pre-trained on ImageNet, added custom layers, and trained with the base layers frozen for 5 epochs using RMSprop optimizer.
- **Fine-Tuning**: Unfroze all layers and continued training for 4 more epochs using the SGD optimizer with a learning rate of 0.0001 and momentum of 0.9.

### Model Saving
The trained model was saved to Google Drive for persistent storage using `model.save('my_model')`.

### Deployment
- **Flask Web Application**
  - **Home Page**: Users can upload images.
  - **Backend**: The uploaded image is sent to a backend server hosted on Colab via a POST request.
  - **Model Loading**: The Colab server loads the model from Google Drive, predicts the DR level, and sends the result back to the Flask app for display.
- **Ngrok**
  - **Exposing Local Server**: Used Ngrok to expose the local Flask app to the internet, enabling user access.

## How to Run the Project

### Clone the Repository:
```bash
git clone https://github.com/ShanmukhiKairuppala/Diabetic_Retinopathy_Detection.git
cd Diabetic_Retinopathy_Detection
```

### Model Training and Saving:
- Train the model using `HyperParameterTuning_%2B_CLAHE_on_DeepVision.ipynb` and save it to Google Drive.

### Backend Setup:

- Open `Mini_Project_DR_Flask_App.ipynb` in Google Colab.
- Run the notebook to start the backend server and obtain the Ngrok link.
 
### Flask App Setup:
- Copy the Ngrok link and paste it into the COLAB_SERVER_URL variable in main.py.
- Run the Flask app using the command:
```bash
python main.py
```
- Access the app via http://localhost:5000.

### User Interaction:
- Upload an image via the Flask app.
- The Flask app sends the image to the Colab server for prediction.
- The Colab server processes the image, predicts the DR level, and returns the result.
- The Flask app displays the prediction result to the user.

### Home Page

<img src="https://github.com/ShanmukhiKairuppala/Diabetic_Retinopathy_Detection/blob/main/homepage.jpg" alt="Home Page" width="800"/>

### Display Page

<img src="https://github.com/ShanmukhiKairuppala/Diabetic_Retinopathy_Detection/blob/main/displayPage.jpg" alt="Display Page" width="800"/>

## Conclusion
This project provides an accessible tool for early detection of Diabetic Retinopathy, potentially aiding in timely medical intervention and reducing the risk of vision loss in diabetic patients.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
