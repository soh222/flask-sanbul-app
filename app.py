from flask import Flask, render_template, request
import os
import traceback

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "sanbul2district-divby100.csv")
MODEL_PATH = os.path.join(BASE_DIR, "sanbul_model.keras")

model = None
full_pipeline = None


def load_resources():
    global model, full_pipeline

    if model is not None and full_pipeline is not None:
        return model, full_pipeline

    print("리소스 로딩 시작", flush=True)

    import numpy as np
    import pandas as pd
    from tensorflow import keras
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedShuffleSplit

    fires = pd.read_csv(CSV_PATH, sep=",")
    fires["burned_area"] = np.log(fires["burned_area"] + 1)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(fires, fires["month"]):
        strat_train_set = fires.loc[train_index]

    fires_data = strat_train_set.drop("burned_area", axis=1)
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
    model = keras.models.load_model(MODEL_PATH)

    print("리소스 로딩 완료", flush=True)
    return model, full_pipeline


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "GET":
        return render_template("prediction.html")

    try:
        import numpy as np
        import pandas as pd

        model, full_pipeline = load_resources()

        input_data = pd.DataFrame([{
            "longitude": int(request.form["longitude"]),
            "latitude": int(request.form["latitude"]),
            "month": request.form["month"].strip(),
            "day": request.form["day"].strip(),
            "avg_temp": float(request.form["avg_temp"]),
            "max_temp": float(request.form["max_temp"]),
            "max_wind_speed": float(request.form["max_wind_speed"]),
            "avg_wind": float(request.form["avg_wind"])
        }])

        print("입력 데이터:", input_data.to_dict(orient="records"), flush=True)

        input_prepared = full_pipeline.transform(input_data)

        if hasattr(input_prepared, "toarray"):
            input_prepared = input_prepared.toarray()

        input_prepared = np.asarray(input_prepared, dtype=np.float32)

        pred_log = model.predict(input_prepared, verbose=0)
        pred_value = float(np.exp(pred_log[0][0]) - 1)

        print("예측 완료:", pred_value, flush=True)

        return render_template("result.html", prediction=round(pred_value, 2))

    except Exception as e:
        print("prediction 예외 발생", flush=True)
        traceback.print_exc()
        return f"오류 발생: {e}", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)