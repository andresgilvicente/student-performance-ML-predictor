<div align="center">
  <img src="assets/banner_readme.png" alt="Project Banner" width="100%" style="border-radius: 10px;">

  <h1>🎓 Student Performance ML Predictor</h1>
  <p><strong>Machine Learning Final Project: Educational Data Analysis & Forecasting</strong></p>

  <p>
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    </a>
    <a href="https://scikit-learn.org/">
      <img src="https://img.shields.io/badge/Scikit_Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit Learn">
    </a>
    <a href="https://pandas.pydata.org/">
      <img src="https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
    </a>
    <a href="https://jupyter.org/">
      <img src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
    </a>
  </p>
</div>

<hr />

## 📖 Project Overview

This project aims to apply **supervised and unsupervised machine learning techniques** to analyze real educational data and predict the academic performance of secondary school students in Madrid.  
Furthermore, it analyzes which variables are most influential in students' school performance, with the goal of proposing measures to help improve it in cases where necessary.

## 📂 Repository Structure

The project follows a modular structure to ensure reproducibility and clean code management:

```text
├── 📂 assets/                   # Images and other resources
├── 📂 docs/                     # Official Final Report
│
├── 📂 src/                      # Source Code & Jupyter Notebooks
│   ├── 📄 exploracion.ipynb     # Initial Exploratory Data Analysis (EDA)
│   ├── 📄 functions.py          # Functions for visualization, calculations, etc.
│   ├── 📄 modelo1.ipynb         # Experimentation and selection of Model 1
│   ├── 📄 modelo2.ipynb         # Experimentation and selection of Model 2
│   ├── 📄 predicciones.ipynb    # Implementation of models and prediction
│   └── 📄 preprocesado.ipynb    # Data cleaning and processing
│
├── 📄 .gitignore                # .gitignore file
├── 📄 README.md                 # Main Project Documentation
└── 📄 requirements.txt          # Project dependencies
```

## 🎯 Project Objectives

1. **Analyze** the most influential factors in academic performance.
2. **Build predictive models** both with and without variables `T1` and `T2` (previous grades).
3. **Compare models** based on metrics such as , MAE and MSE.
4. **Propose measures** to improve academic performance based on the findings.

## 🚀 Installation and Deployment

To avoid conflicts with other libraries or Python projects, it is recommended to work within a **virtual environment**.

You can create and activate a virtual environment using the following commands:

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate

```

### Windows

```bash
python -m venv venv
venv\Scripts\activate

```

Next, install the libraries specified in **requirements.txt** using the command:

```bash
pip install -r requirements.txt

```

To deploy the project in a way that allows for a step-by-step understanding of the decisions and processes carried out, it is recommended to execute the attached files one by one in the order marked in the **Project Structure** section.

## 💾 Results Storage

The final predictions are stored in a folder named **`submission`**, in a file named **`predicciones_finales.csv`**.
However, the folder and file names can be easily modified as needed, just like the folder where the initial project data is stored.

## 👤 Author

* **Andrés Gil Vicente**
  * *Degree in Mathematical Engineering and Artificial Intelligence (iMAT)*
  * *Machine Learning Course*
  * *Universidad Pontificia Comillas - ICAI*

*Completion Date: May 4, 2025*




