# ECG_Classification

Final Project of **Intro. Machine Learning** course (EE-25737) @ Sharif University of Technology.

Instructor: Dr. Hoda Mohammadzade; Spring 2021

---

### Preprocessing (using MATLAB):

- Baseline drift removal using highpass filter.
- Powerline noise removal using bandstop filter.
- High-frequency noise removal using lowpass filter.

Results:

Original Signal:

<img src="https://user-images.githubusercontent.com/94138466/155301209-35b664ac-7b9b-4e3c-9917-4ba913640f51.png" width=70% height=70%>

Preprocessed Signal:

<img src="https://user-images.githubusercontent.com/94138466/155301146-98ee4b8b-beb2-4564-9bf2-f4cbd0f68d60.png" width=70% height=70%>

### Feature Extraction (using Python):

- Time domain features (mean, variance, histogram, mean rate, ...)
- Frequency domain features (maximum frequency, ...)
- Morphological features (QRS Interval, PQRST, ...) using [biosppy](https://biosppy.readthedocs.io/en/stable/)

![image](https://user-images.githubusercontent.com/94138466/155302352-1c21531b-20e2-4785-9d09-7bbee297b5c9.png)
![image](https://user-images.githubusercontent.com/94138466/155302378-9e7a53e8-9a76-4bca-b33c-cda433a793bd.png)

### Model Evaluation (Hyperparameters were tuned using gridsearch)

| Accuracy | Model 
| --------------- | --------------- 
| 58.20 | KNN
| 61.00 | Logistic Regression
| 66.62 | SVM
| 30.51 | Random Forest
| 76.65 | LDA (Linear Discriminant Analysis)

