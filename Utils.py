import numpy as np


def compute_error(output, target):
    assert output.shape[1] == 1
    assert target.shape[1] == 1
    assert output.shape[0] == target.shape[0]

    error = np.linalg.norm(target - output, 2) ** 2

    return 0.5 * error


def compute_accuracy(output, target, threshold=0.5):
    assert output.shape[1] == 1
    assert output.shape[0] == target.shape[0] == 1

    out = 0

    # print("Output = %s Target = %s" %(output[0],target[0]))

    if output[0] >= threshold:
        out = 1

    if out == target[0]:
        return 1

    return 0


def majority_vote(outputs):

    zero_prediction= 0
    one_prediction = 0

    for out in outputs:
        if out == 0:
            zero_prediction += 1
        else:
            one_prediction += 1

    if one_prediction >= zero_prediction:
        return 1

    return 0


def compute_class(output,threshold=0.5):
    assert output.shape[1] == 1
    assert output.shape[0] == 1

    prediction = 0
    if output >= threshold:
        prediction = 1

    return prediction
