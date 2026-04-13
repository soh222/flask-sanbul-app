from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "sanbul2district-divby100.csv")
model_path = os.path.join(BASE_DIR, "sanbul_model.keras")

# 데이터 불러오기
fires = pd.read_csv(csv_path, sep=",")
fires["burned_area"] = np.log(fires["burned_area"] + 1)

# stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set = fires.loc[test_index]

fires_data = strat_train_set.drop("burned_area", axis=1)

# 전처리 파이프라인
fires_num = fires_data.drop(["month", "day"], axis=1)
num_attribs = list(fires_num.columns)
cat_attribs = ["month", "day"]

num_pipeline = Pipeline([
    ("std_scaler", StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
])

full_pipeline.fit(fires_data)

# 모델 불러오기
model = keras.models.load_model(model_path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "GET":
        return render_template("prediction.html")

    try:
        input_data = pd.DataFrame([{
            "longitude": int(request.form["longitude"]),
            "latitude": int(request.form["latitude"]),
            "month": request.form["month"],
            "day": request.form["day"],
            "avg_temp": float(request.form["avg_temp"]),
            "max_temp": float(request.form["max_temp"]),
            "max_wind_speed": float(request.form["max_wind_speed"]),
            "avg_wind": float(request.form["avg_wind"])
        }])

        input_prepared = full_pipeline.transform(input_data)
        pred_log = model.predict(input_prepared)
        pred_value = float(np.exp(pred_log[0][0]) - 1)

        return render_template("result.html", prediction=round(pred_value, 2))

    except Exception as e:
        return f"오류 발생: {e}"


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)