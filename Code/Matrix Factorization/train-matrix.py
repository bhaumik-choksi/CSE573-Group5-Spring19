import pandas as pd
import pickle
import seaborn as sns
from surprise import Reader, Dataset, SVD, evaluate

sns.set_style("darkgrid")

df1 = pd.read_csv('../../Data/combined_data_1.txt', header=None, usecols=[0, 1], names=['uid', 'rating'])
df1['rating'] = df1['rating'].astype(float).fillna(1.0)
df1['iid'] = pd.DataFrame(list(range(len(df1))))

df = df1.head(100000)
df = df[['uid', 'iid', 'rating']]

df_title = pd.read_csv('../../Data/movie_titles.csv', encoding="ISO-8859-1", header=None,
                       names=['Movie_Id', 'Year', 'Name'])

USERID = '822109'

reader = Reader()
data = Dataset.load_from_df(df, reader)
alg = SVD()
output = alg.fit(data.build_full_trainset())
evaluate(alg, data)

pickle.dump([alg, df, df_title], open('../../Evaluations/matrix-data.p', "wb"))

print(df[df['rating'] == 5]['uid'])
