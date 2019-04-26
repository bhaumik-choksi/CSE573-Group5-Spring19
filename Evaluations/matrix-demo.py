import pickle
import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD, evaluate
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField


def predict_movies(USERID):
    svd, df, df_title = pickle.load(open('matrix-data.p', "rb"))
    df_current_best = df[(df['uid'] == USERID) & (df['rating'] >= 4)]
    df_current_best = df_current_best.set_index('iid')
    df_current_best = df_current_best.join(df_title)

    current = df_current_best['Name'].tolist()

    df_title['pred'] = df_title['Movie_Id'].apply(lambda x: svd.predict(USERID, x).est)
    df_title = df_title.sort_values('pred', ascending=False)
    predicted = df_title.head(10)['Name'].tolist()

    return [current, predicted]


# print(predict_movies(372233)[0])
DEBUG = True
app = Flask(__name__)


class ReusableForm(Form):
    name = TextField('Enter Valid User ID:')


@app.route("/", methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)
    if request.method == "POST":
        c, p = predict_movies(form.name.data)
        return render_template('result.html', c=c, p=p, u=form.name.data)

    return render_template('index.html', form=form)


if __name__ == "__main__":
    app.run(debug=DEBUG)
