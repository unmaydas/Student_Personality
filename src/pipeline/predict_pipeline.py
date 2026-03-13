import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            result = ["Extrovert" if p == 0 else "Introvert" for p in preds]
            return result
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Time_spent_Alone: float,
        Stage_fear: str,
        Social_event_attendance: float,
        Going_outside: float,
        Drained_after_socializing: str,
        Friends_circle_size: int,
        Post_frequency: int):

        self.Time_spent_Alone = Time_spent_Alone
        self.Stage_fear = Stage_fear
        self.Social_event_attendance = Social_event_attendance
        self.Going_outside = Going_outside
        self.Drained_after_socializing = Drained_after_socializing
        self.Friends_circle_size = Friends_circle_size
        self.Post_frequency = Post_frequency

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Time_spent_Alone": [self.Time_spent_Alone],
                "Stage_fear": [self.Stage_fear],
                "Social_event_attendance": [self.Social_event_attendance],
                "Going_outside": [self.Going_outside],
                "Drained_after_socializing": [self.Drained_after_socializing],
                "Friends_circle_size": [self.Friends_circle_size],
                "Post_frequency": [self.Post_frequency],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)