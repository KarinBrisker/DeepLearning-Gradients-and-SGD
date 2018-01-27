import loglinear as ll
import random
import utils as ut
from collections import defaultdict



# from sentence to vector of counters
def feats_to_vec(features):
    counters = defaultdict(int)
    bigrams = ut.text_to_bigrams(features)
    for bigram in bigrams:
        id = ut.F2I(bigram)
        counters[id] += 1
    # Should return a numpy vector of features.
    return counters


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        prediction = ll.predict(features, params)
        if label == prediction:
            good += 1
        else:
            bad += 1
        pass
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = label                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss
            params = params - learning_rate * grads
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params

if __name__ == '__main__':
    TRAIN = [(l, ut.text_to_bigrams(t)) for l, t in ut.read_data("train")]
    DEV = [(l, ut.text_to_bigrams(t)) for l, t in ut.read_data("dev")]

    from collections import Counter

    fc = Counter()
    for l, feats in TRAIN:
        fc.update(feats)

    params = ll.create_classifier(ut.vocab, 12)
    trained_params = train_classifier(TRAIN, DEV, 10000, 0.5, params)

