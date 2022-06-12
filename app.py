from flask import Flask,render_template,request
import numpy as np
import pickle

model=pickle.load(open('deep.pkl','rb'))

app=Flask(__name__)

@app.route("/")
def man():
      return render_template('front.html')
@app.route('/home')
def ef():
      return render_template('home.html')

@app.route('/predict',methods=['POST'])
def home():
    data1=request.form['q1']
    data2=request.form['q2']
    data3=request.form['q3']
    data4=request.form['q4']
    data5=request.form['q5']
    data6=request.form['q6']
    data7=request.form['q7']
    data8=request.form['q8']
    data9=request.form['q9']
    data10=request.form['q10']
    arr=np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10]])
    pred=model.predict(arr);
    output = round(pred[0], 2)
  
    return render_template('after.html',data=output)
    
if __name__=="__main__":
        app.run(debug=True)