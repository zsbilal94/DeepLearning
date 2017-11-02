import os
import tensorflow as tf
import pandas as pd
import numpy as np


cancer_training = "cancertrain.csv"
cancer_test = "cancertest.csv"

def main():
    # Loading in the datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=cancer_training,
                                                        target_dtype=np.int,
                                                        features_dtype=np.float32)
                                                        
                                                        
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=cancer_test,
                                                    target_dtype=np.int,
                                                    features_dtype=np.float32)                                       

    #features which have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("x", dimension=13)]

    # Build 3 layer DNN with 15, 30, 10 nodes
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[15, 30, 10],
                                                n_classes=4,
                                                model_dir="/tmp/cancer_model")

    # Train model
    classifier.train(input_fn=train_input_fn, steps=2000)

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False)

    accuracy = classifier.evaluate(x = test_set.data,
                                        y = test_set.target)["accuracy"]
    print('Accuracy: {0:f}'.format(accuracy))

if __name__ == "__main__":
    main()