import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Page Config ──
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ── Load & Prepare Data ──
@st.cache_data
def load_and_train():
    df = sns.load_dataset('titanic')
    df = df[['survived', 'pclass', 'sex', 'age', 'fare', 'embarked']].dropna()

    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df['embarked'] = le.fit_transform(df['embarked'])

    X = df[['pclass', 'sex', 'age', 'fare', 'embarked']]
    y = df['survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, accuracy, X.columns.tolist()

model, accuracy, feature_names = load_and_train()

# ── UI ──
st.title("🚢 Titanic Survival Predictor")
st.markdown("Would you have survived the Titanic? Enter your details and find out!")
st.markdown(f"*Model accuracy: **{accuracy*100:.1f}%***")
st.divider()

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3],
                          format_func=lambda x: f"{'First' if x==1 else 'Second' if x==2 else 'Third'} Class")
    sex = st.radio("Gender", ["Female", "Male"])
    age = st.slider("Age", min_value=1, max_value=80, value=25)

with col2:
    fare = st.slider("Ticket Fare (£)", min_value=0, max_value=500, value=30)
    embarked = st.selectbox("Port of Embarkation",
                            ["Cherbourg", "Queenstown", "Southampton"])

st.divider()

if st.button("🔮 Predict My Survival", use_container_width=True):

    sex_encoded = 1 if sex == "Male" else 0
    embarked_encoded = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}[embarked]

    input_data = pd.DataFrame([[pclass, sex_encoded, age, fare, embarked_encoded]],
                               columns=['pclass', 'sex', 'age', 'fare', 'embarked'])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.divider()

    if prediction == 1:
        st.success(" ✅ You would have SURVIVED!")
        st.metric("Survival Probability", f"{probability[1]*100:.1f}%")
    else:
        st.error("## ❌ You would NOT have survived.")
        st.metric("Survival Probability", f"{probability[1]*100:.1f}%")

    st.divider()
    st.markdown("### 💡 Why?")
    if sex == "Female":
        st.write("Women had a **74% survival rate** — 'women and children first' was real.")
    else:
        st.write("Men had only a **19% survival rate** — most men didn't make it.")
    if pclass == 1:
        st.write("Being in **1st class** gave you much better access to lifeboats.")
    elif pclass == 3:
        st.write("**3rd class passengers** had the lowest survival rate (~24%).")

# ── Feature Importance Chart ──
st.divider()
st.markdown("📊 What Factors Mattered Most?")
st.markdown("Based on the Random Forest model, here's how much each feature influenced survival:")

importances = model.feature_importances_
feat_df = pd.DataFrame({
    'Feature': ['Passenger Class', 'Gender', 'Age', 'Fare', 'Embarkation Port'],
    'Importance': importances
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(feat_df['Feature'], feat_df['Importance'], color='steelblue')
ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=10)
ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance (Random Forest)")
ax.set_xlim(0, feat_df['Importance'].max() + 0.08)
plt.tight_layout()
st.pyplot(fig)

st.caption("Higher score = stronger influence on survival prediction.")

# ── Historical Stats ──
st.divider()
st.markdown("📜 Real Titanic Survival Stats")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**By Gender**")
    gender_data = pd.DataFrame({
        'Group': ['Women', 'Men'],
        'Survival Rate': [74, 19]
    })
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.bar(gender_data['Group'], gender_data['Survival Rate'],
            color=['#f7a8c4', '#7eb8f7'])
    ax2.set_ylabel("Survival Rate (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("Survival by Gender")
    for i, v in enumerate(gender_data['Survival Rate']):
        ax2.text(i, v + 1, f"{v}%", ha='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2)

with col2:
    st.markdown("**By Class**")
    class_data = pd.DataFrame({
        'Class': ['1st', '2nd', '3rd'],
        'Survival Rate': [63, 47, 24]
    })
    fig3, ax3 = plt.subplots(figsize=(4, 3))
    ax3.bar(class_data['Class'], class_data['Survival Rate'],
            color=['#a8e6cf', '#f0c060', '#f7a8c4'])
    ax3.set_ylabel("Survival Rate (%)")
    ax3.set_ylim(0, 100)
    ax3.set_title("Survival by Class")
    for i, v in enumerate(class_data['Survival Rate']):
        ax3.text(i, v + 1, f"{v}%", ha='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig3)

st.caption("Source: Historical Titanic passenger data")