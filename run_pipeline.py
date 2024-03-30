import json
import argparse
from preprocessor import Preprocessor
from model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename')
    parser.add_argument('--mode', default='training')
    args = parser.parse_args()
    path = args.filename
    mode = args.mode

    if mode == 'training':
        preprocess = Preprocessor(path)
        X, y = preprocess.fit()
        model = Model()
        model.fit(X, y)

    elif mode == 'testing':
        preprocess = Preprocessor(path)
        X = preprocess.transform()
        model = Model()
        y_probs = list(model.predict(X))
        predictions = {'threshold': 0.27, "predict_probas": y_probs}
        with open("predictions.json", "w") as outfile:
            json.dump(predictions, outfile)

    else:
        raise Exception(f"Mode should be 'training' or 'testing' instead of {mode}.")


if __name__ == '__main__':
    main()