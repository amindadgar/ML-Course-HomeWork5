from logging import debug
from flask import Flask, jsonify, request
import pickle

app = Flask('customer-churn')


## load the model from file
with open('model1.bin','rb') as f_in:
    model = pickle.load(f_in)

with open('dv.bin','rb') as f_in:
    dv = pickle.load(f_in)

## Predict for a single customer
def predict_single(dv, model, customer):
    ## Apply one hot encoding
    custome_data = dv.transform([customer])
    
    prediction = model.predict_proba(custome_data)[:,1]  ## get the second prediction row
    
    return prediction[0]


@app.route('/predict',methods = ['POST'])
def predict():
    customer = request.get_json()

    prediction = predict_single(dv, model, customer)

    churn = prediction >= 0.5

    result = {
        'churn_probability': float(prediction),
        'churn': bool(churn)
    }

    return jsonify(result)

@app.route('/ping', methods = ['GET'])
def ping():
    return 'PONG'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)






