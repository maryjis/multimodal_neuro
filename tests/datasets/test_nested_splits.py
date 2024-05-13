import random

import numpy as np
from neurograph.unidata.utils import get_nested_splits


def test_nested_splits():
    idx = np.arange(127)
    y = np.array(
        random.choices(
            population=[0, 1],
            weights=[4, 10],
            k=127,
        )
    )

    splits = get_nested_splits(idx, y, num_folds=10, random_state=1380)

    all_train = set()
    all_valid = set()
    all_test = set()

    print("\n")
    for i, (train_idx, valid_idx, test_idx) in enumerate(splits):
        train = set(train_idx)
        valid = set(valid_idx)
        test = set(test_idx)

        y_train = y[train_idx]
        y_valid = y[valid_idx]
        y_test = y[test_idx]

        all_train |= train
        all_valid |= valid
        all_test |= test

        assert train & valid == set()
        assert train & test == set()
        assert valid & test == set()

        print(
            "class 1 percentage: "
            f"fold: {i};"
            f" all data: {y.mean():.2f},"
            f" train {y_train.mean():.2f},"
            f" valid {y_valid.mean():.2f},"
            f" test {y_test.mean():.2f}"
        )

    assert all_train == all_valid == all_test
