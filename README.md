# Advanced MNIST Analyzer

## What is this?

This is a project that allows for the analysis of consecutive MNIST images. It is a project that is able to decipher images containing 3 digits stacking on top of each other with a certain amount of noise. The project is able to detect the 3 digits and classify them accordingly.

## How does it work?

The project features four different classification models that are able to classify the digits in the image. The models are:
- A simple CNN model
- A SVM classifier
- A Random Forest classifier
- A logistic regression classifier

## How to run the project?

1. Clone the repository using the following command:
```bash
git clone https://github.com/Nguyen-HanhNong/Advanced-MNIST-Analyzer.git
```

2. Install python and pip if you haven't already. You can download python from the following link: https://www.python.org/downloads/

3. Install the necessary dependencies using the following command:
```bash
pip install -r requirements.txt
```
or manually install the following dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

4. Run the project using the following command:
```bash
python main.py
```
