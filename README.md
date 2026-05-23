Meaning:

```text
0 = Not delayed
1 = Delayed
```

---

## Feature Engineering

New features were created to improve the model:

- `is_peak_hour`
- `heavy_package`
- `distance_category`
- `driver_experience_level`

These features help the model understand real-world delivery conditions better.

---

## Machine Learning Model

The project uses a Random Forest Classifier.

The model was trained using delivery features such as:

- Distance
- Package weight
- Traffic level
- Weather condition
- Vehicle type
- Delivery hour
- Driver experience
- Peak hour status
- Package weight category
- Distance category
- Driver experience level

---

## Model Performance

The model achieved approximately:

```text
87% accuracy
```

The confusion matrix showed that the model was able to correctly predict many delayed and non-delayed deliveries.

---

## Streamlit App

The project includes a Streamlit web app where users can enter delivery details and get a prediction.

The app predicts:

```text
Delayed
or
Not Delayed
```

It also shows prediction confidence.

---

## Project Structure

```text
smartlogistics-delivery-delay-prediction/
│
├── data/
│   └── delivery_data.csv
│
├── models/
│   ├── delivery_delay_model.pkl
│   └── model_columns.pkl
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── outputs/
│
├── screenshots/
│
├── src/
│   ├── generate_dataset.py
│   └── train_model.py
│
├── app.py
├── README.md
└── requirements.txt
```

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/ruvarashe12/smartlogistics-delivery-delay-prediction.git
```

### 2. Move into the project folder

```bash
cd smartlogistics-delivery-delay-prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

On Windows:

```bash
venv\Scripts\activate
```

On Mac/Linux:

```bash
source venv/bin/activate
```

### 5. Install requirements

```bash
pip install -r requirements.txt
```

### 6. Train the model

```bash
python src/train_model.py
```

### 7. Run the Streamlit app

```bash
streamlit run app.py
```

Or:

```bash
python -m streamlit run app.py
```

---

## Example Prediction

Example high-risk delivery:

```text
Distance: 35 km
Package weight: 25 kg
Traffic: High
Weather: Stormy
Vehicle: Bike
Delivery hour: 17
Driver experience: 1 year
```

The model is likely to predict:

```text
Delayed
```

Example low-risk delivery:

```text
Distance: 4 km
Package weight: 2 kg
Traffic: Low
Weather: Clear
Vehicle: Bike
Delivery hour: 11
Driver experience: 8 years
```

The model is likely to predict:

```text
Not Delayed
```

---

## Skills Demonstrated

This project demonstrates:

- Python programming
- Data analysis
- Data cleaning
- Exploratory data analysis
- Feature engineering
- Machine learning classification
- Model evaluation
- Streamlit app development
- GitHub project documentation

---

## Future Improvements

Possible improvements include:

- Use real logistics data
- Add delivery cost prediction
- Add route optimization
- Add map-based delivery tracking
- Deploy the app online
- Improve model performance with more data

---

## Author

Created by Ruvarashe as a beginner data science project focused on transport and logistics.