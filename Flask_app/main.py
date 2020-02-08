"""Entry point for the flask app"""

from flask import Flask,request,jsonify
import config
from services import model
from services.predict import reshape_json,predict_sales_price
app = Flask(__name__)


# training the model when application starts using functions
# train_X, train_y = model.create_test_train(file_path=config.file_path,relevant_columns=config.important_columns)
# logistic_regression_model = model.train_model(train_X=train_X,train_y=train_y)


# training the model using oops
model_object = model.MODEL(file_path=config.file_path,relevant_columns=config.important_columns)
model_object.create_test_train()
logistic_regression_model=model_object.train_model()


# define paths
@app.route('/')
def hello_world():
    """what do we wanna show on the homepage"""
    return 'Hello, IISER!'


@app.route('/predict', methods=["POST"])
def predict():
    """Predicting the sales price given the said parameters"""
    request_json = request.get_json()
    prediction = predict_sales_price(data_json=request_json,model=logistic_regression_model)
    prediction=str(prediction[0][0])
    return jsonify({"Prediction": prediction})






if __name__=='__main__':
    app.run(port=8080, host='0.0.0.0')





