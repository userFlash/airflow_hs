# <YOUR_IMPORTS>
import glob
import os
from datetime import datetime

import dill
import json
import pandas as pd


path = os.environ.get('$PROJECT_PATH', '')
def predict():
    mod = sorted(os.listdir(f'{path}/data/models/'))

    with open(f'{path}/data/models/{mod[-1]}', 'rb') as file:
        model = dill.load(file)

        preds = pd.DataFrame(columns=['car_id', 'pred'])

        for file_j in glob.glob(f'{path}/data/test/*.json'):
            with open(file_j) as fin:
                form = json.load(fin)
                df = pd.DataFrame.from_dict([form])
                y = model.predict(df)
                X = {'car_id': df.id, 'pred': y}
                df1 = pd.DataFrame(X)
                preds = pd.concat([preds, df1], axis=0)
        print(preds)

        preds.to_csv(f'{path}/data/predictions/pred_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)

if __name__ == '__main__':
    predict()
