from flask import Flask, render_template, request, flash
from forms import dataForm
from flask_wtf import CSRFProtect
from keras.models import load_model
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
#csrf = CSRFProtect(app)
app.config['SECRET_KEY'] = 'seckey'

model = load_model('my_model.h5')
graph = tf.get_default_graph()
dataset = pd.read_csv('dataset6.csv', delimiter=",")
X = dataset.iloc[:,:-1].values
st_sc = StandardScaler()
X = st_sc.fit_transform(X)

#print(model.predict(np.array([[18, 1, 0.4, 0.1, 99, 53, 103, 7.6, 4.3, 1.3]]))[0][0])
#print(model.predict(np.array(st_sc.fit_transform([[18, 1, 0.4, 0.1, 99, 53, 103, 7.6, 4.3, 1.3]],dataset.iloc[:,:-1].values))))


@app.route('/', methods=['GET', 'POST'])
def test():
    global graph
    with graph.as_default():
        result = ''
        scroll = False
        testscroll = ''
        test = True
        scrolly = ''
        form = dataForm(request.form)
        if form.is_submitted():
            print(model.predict(np.array(st_sc.transform([[18, 1, 0.4, 0.1, 99, 53, 103, 7.6, 4.3, 1.3]]))))
            print(form.age.data)
            print(type(form.age.data))
            #scrolly = '/#result'
            #print(model.predict(np.array([[form.age.data, 1, 0.4, 0.1, 99, 53, 103, 7.6, 4.3, 1.3]]))[0][0])
            for x in form:
                if x.data is None:
                    test = False

            if test:
                result = model.predict(np.array(st_sc.transform([[float(form.age.data), float(form.gender.data),
                                                                  float(form.total_bilirubin.data), float(form.direct_bilirubin.data),
                                                                  float(form.alkaline_phosphate.data), float(form.alamine_aminotransferase.data),
                                                                  float(form.aspartate_aminotransferase.data), float(form.total_proteins.data),
                                                                  float(form.albumin.data), float(form.albumin_and_globulin_ratio.data)]]
                                                                    )))[0][0]
            else:
                result = 'Input correct field parameters!'
            print(result)
            #form.ast.data
            test = True
            testscroll = True
            print(testscroll)
        return render_template('index2.html' + scrolly, form=form, result=result, scroll=testscroll)

@app.route('/tos', methods=['GET'])
def tos():
    return render_template('tos.html')

if __name__ == '__main__':
    app.run(debug=True)

