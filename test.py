from predict import Predict
from config import root_path
import json

if __name__ == '__main__':
    text = '装帧全部采用日本进口竹尾纸，专为读书人打造奢华手感 ◆ 畅销100万册，独占同名书市场七成份额...'
    predict = Predict()
    label, score = predict.predict(text)
    print('label:{}'.format(label))
    print('score:{}'.format(score))
    with open(root_path + '/data/label2id.json', 'r') as f:
        label2id = json.load(f)
    print(list(label2id.keys())[list(label2id.values()).index(label)])


