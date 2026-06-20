# JAMB Score Band Predictor

A machine learning web application that predicts a student's likely JAMB performance band based on their study habits, school background, and socioeconomic factors. Built to help Nigerian students plan and improve their JAMB preparation before sitting the exam.

---

## Problem Statement

Every year, thousands of Nigerian students sit the Joint Admissions and Matriculation Board (JAMB) examination underprepared. Many do not know which areas of their preparation are weak until after they receive a poor score. This application addresses that gap by giving students a predicted performance band before the exam, along with specific advice on what to improve.

---

## Objectives

- Predict a student's JAMB score band using machine learning
- Identify the key factors that influence JAMB performance
- Provide actionable feedback to students based on their predicted band
- Deploy the model as a accessible web application

---

## Score Categories

| Band | Score Range | Meaning |
|---|---|---|
| On Track | 220 and above | Student is well prepared |
| At Risk | 160 – 219 | Student needs serious intervention |
| Critical | Below 160 | Student needs major changes immediately |

---

## Dataset

- 900 student records collected via survey
- 22 input features covering study habits, school background, socioeconomic factors, and exam readiness
- Target variable: JAMB score band (3 classes)
- Data collected from Nigerian secondary school students and JAMB candidates

### Features Used

| Category | Features |
|---|---|
| Personal | Gender, Age when took JAMB, JAMB attempts count |
| School | School type, School location, Quality of instruction |
| Study Habits | Study hours per day, Study plan adherence, Consistent study schedule, Attended extra tutorials |
| Study Resources | Past questions, Textbooks, Online tutorials, Coaching centers, Private lessons, Study groups |
| Home and Access | Guardian education level, Household dependents, Access to computer/smartphone, Internet reliability, Access to textbooks, Electricity consistency |
| Exam Readiness | Familiar with CBT format |

---

## Methodology

### 1. Data Preprocessing
- Handled missing values using mode for categorical columns and median for numeric columns
- Mapped range-based text columns to numeric midpoints
- Applied ordinal encoding for ordered categorical variables
- Applied label encoding for binary and nominal variables
- Performed multi-label encoding for study resources column
- Checked and handled outliers using IQR method

### 2. Exploratory Data Analysis
- Distribution of JAMB scores
- Score category distribution
- Average score by school type and school location
- Impact of study hours on performance
- Score distribution by gender
- Effect of extra tutorial attendance
- Correlation heatmap of numeric features

### 3. Model Training
- Split data into 80% training and 20% testing with stratification
- Trained two models: Random Forest Classifier and XGBoost Classifier
- Applied class_weight='balanced' to handle class imbalance
- Compared models using accuracy, precision, recall, F1-score, and confusion matrix

### 4. Model Selection
Random Forest was selected as the final model based on superior performance.

| Model | Accuracy |
|---|---|
| Random Forest | 90.00% |
| XGBoost | 88.89% |

### 5. Deployment
- Built REST API using FastAPI
- Deployed on Render
- No frontend framework required. Pure HTML and CSS served directly from FastAPI.

---

## Model Performance

### Random Forest Classifier

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Critical | 0.89 | 0.83 | 0.86 |
| At Risk | 0.86 | 0.97 | 0.91 |
| On Track | 1.00 | 0.86 | 0.93 |
| Overall Accuracy | | | 90.00% |

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python |
| ML Libraries | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web Framework | FastAPI |
| Model Serialization | Joblib |
| Deployment | Render |
| Development | Google Colab, VS Code |

---

## Project Structure
---

## How to Run Locally

1. Clone the repository

```bash
git clone [https://github.com/Directorstat/DecodeLabs-Internship](https://github.com/Directorstat/DecodeLabs-Internship-Data-Science/tree/main/Task1).git
cd DecodeLabs-Internship
```

2. Create and activate virtual environment

```bash
python -m venv venv
source venv/Scripts/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the application

```bash
uvicorn App:app --reload
```

5. Open in browser
---

## Key Findings from EDA

- Study hours per day is the strongest predictor of JAMB performance with a correlation of 0.35
- Students studying 4 or more hours daily averaged 212, approaching the On Track threshold
- Semi-Urban students recorded higher median scores than Urban and Rural students
- Extra tutorial attendance alone does not guarantee higher scores. Consistency and quality of study matter more.
- 75% of students in the dataset scored below 220, confirming the At Risk and Critical bands dominate

---

## Author

**Waliyullahi**
Data Scientist | 3MTT Nigeria Fellow | NYSC Corps Member
President, Digital Literacy For All (DL4ALL) CDS Group, Uhunwode LGA, Edo State

---

## Acknowledgement

This project was developed as part of the Decode Labs Data Science Internship program. Special thanks to the students who participated in the data collection survey.

---

## Disclaimer

This tool is for planning and advisory purposes only. Predictions are based on patterns in survey data and do not guarantee actual JAMB scores.
