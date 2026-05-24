An interactive machine learning web app that predicts whether you would have survived the Titanic disaster based on your passenger details.

🔗 **Live App:** [sv2008-titanic-predictor.streamlit.app](https://sv2008-titanic-predictor.streamlit.app)



## 📌 Features 

- Enter your age, gender, passenger class, fare, and port of embarkation
- Get an instant survival prediction with probability score
- View feature importance chart showing what factors influenced survival most
- Explore real historical Titanic survival stats by gender and class



## 🛠️ Tech Stack

- **Python** — core language
- **Pandas** — data manipulation
- **Seaborn / Matplotlib** — data visualization
- **Scikit-learn** — Random Forest classification model
- **Streamlit** — web app framework and deployment



## 📊 Model Performance

- Algorithm: Random Forest Classifier (100 estimators)
- Accuracy: **76.9%** on test set
- Train/Test Split: 80/20



## 🔍 Key Findings from EDA

- Only **38%** of passengers survived
- Women had a **74% survival rate** vs **19% for men**
- 1st class passengers survived at **63%** vs **24% for 3rd class**
- Gender and fare were the strongest predictors of survival
- ~177 age values were missing and were handled before training



## 📁 Project Structure

```
summer-ds/
├── app.py               # Streamlit web app
├── titanic_eda.ipynb    # Exploratory Data Analysis notebook
├── requirements.txt     # Python dependencies
└── README.md
```



## 🚀 Run Locally

```bash
git clone https://github.com/sv2008/summer-ds.git
cd summer-ds
pip install -r requirements.txt
streamlit run app.py
```

---

*Built during my first CS Summer break — 2026*
