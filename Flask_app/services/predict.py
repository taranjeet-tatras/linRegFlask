"""Functions that will be used for prediction """
import  numpy as np

def reshape_json(data_json):
    """we are taking the raw json that's given to us and reshaping for perdiction"""
    data_values=[data_json[key] for key in data_json.keys()]
    data_arr = np.asarray(data_values)
    print(data_values)
    train_x = data_arr.reshape(1,10)
    return train_x

def predict_sales_price(data_json,model):
    """This function will output the prediction"""
    reshaped_value=reshape_json(data_json=data_json)
    prediction = model.predict(reshaped_value)
    return prediction