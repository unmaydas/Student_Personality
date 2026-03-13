from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__, template_folder="src/templates")


@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'POST':

        data = CustomData(
            Time_spent_Alone=float(request.form.get('Time_spent_Alone')),
            Stage_fear=request.form.get('Stage_fear'),
            Social_event_attendance=float(request.form.get('Social_event_attendance')),
            Going_outside=float(request.form.get('Going_outside')),
            Drained_after_socializing=request.form.get('Drained_after_socializing'),
            Friends_circle_size=float(request.form.get('Friends_circle_size')),
            Post_frequency=float(request.form.get('Post_frequency'))
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template("home.html", results=results[0])

    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)