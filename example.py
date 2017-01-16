import argparse

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score

from newtonfm import FactorizationMachineClassifier


def load_example_data(train_path, test_path):

    X, y = load_svmlight_file(train_path)
    X_test, y_test = load_svmlight_file(test_path, n_features=X.shape[1])
    y = np.expand_dims(y, axis=1)
    y_test = np.expand_dims(y_test, axis=1)
    n = np.max((X.shape[1], X_test.shape[1]))
    X = X.tocsr()
    X_test = X_test.tocsr()

    return X, X_test, y, y_test


def main(train_path, test_path):
    np.seterr(all='raise')
    X_train, X_test, y_train, y_test = load_example_data(train_path, test_path)
    fm = FactorizationMachineClassifier(
        lambda_w=0.0625,
        lambda_U=0.0625,
        lambda_V=0.0625,
        d=4,
        epsilon=0.01,
        do_pcond=True,
        sub_rate=0.1,
        max_iter=100,
        random_seed=0,
        verbose=True
    )
    fm.fit(X_train, y_train)
    y_preds = fm.predict(X_test)
    print('test accuracy: {}'.format(accuracy_score(y_test>0, y_preds>0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='newtonfm example on test data')
    parser.add_argument('--train',
                        default='./test_data/fourclass_scale.tr',
                        type=str,
                        help='Training data.')
    parser.add_argument('--test',
                        default='./test_data/fourclass_scale.te',
                        type=str,
                        help='Test data.')

    args = parser.parse_args()
    main(args.train, args.test)
