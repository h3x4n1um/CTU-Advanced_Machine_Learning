# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.preprocessing
import tensorflow as tf
import tensorflow.keras as keras
import transformers

def create_dataset_from_dir(dataset_dir: str,
                            le: sklearn.preprocessing.LabelEncoder,
                            tokenizer: transformers.AutoTokenizer) -> tf.data.Dataset:
    labels = tf.io.gfile.listdir(dataset_dir)
    print(labels)

    input_ids       = list()
    attention_mask  = list()
    y               = list()
    for label in labels:
        samples = tf.io.gfile.listdir("{}/{}".format(dataset_dir, label))
        print("{}: {}".format(label, len(samples)))
        for sample in samples:
            with open("{}/{}/{}".format(dataset_dir, label, sample), 'r', encoding="utf8", errors="ignore") as f:
                encoded_input = tokenizer(f.read(),
                                          padding="max_length",
                                          truncation=True,
                                          return_tensors="tf")
                input_ids.append(encoded_input["input_ids"][0])
                attention_mask.append(encoded_input["attention_mask"][0])
            y.append(label.lower())
    return tf.data.Dataset.from_tensor_slices(((input_ids, attention_mask), le.transform(y))).shuffle(10000).batch(128)

def main() -> None:
    # const
    VB2000_path = "VB2000"
    test_set    = "test-500"

    # load PhoBERT from transformers
    PhoBERT_name        = "vinai/phobert-base"
    PhoBERT_tokenizer   = transformers.AutoTokenizer.from_pretrained(PhoBERT_name,
                                                                     use_fast=False)

    # label encoder
    le = sklearn.preprocessing.LabelEncoder()
    le.fit([label.lower() for label in tf.io.gfile.listdir("{}/{}".format(VB2000_path, test_set))])

    # load n preprocess data
    test_data   = create_dataset_from_dir("{}/{}".format(VB2000_path, test_set) , le, PhoBERT_tokenizer)

    # get x_test, y_test
    x_test_input_ids        = None
    x_test_attention_mask   = None
    y_test                  = None
    for datas, labels in test_data.as_numpy_iterator():
        datas_input_ids, datas_attention_mask = datas

        if x_test_input_ids is None:
            x_test_input_ids = datas_input_ids
        else:
            x_test_input_ids = np.vstack((x_test_input_ids, datas_input_ids))

        if x_test_attention_mask is None:
            x_test_attention_mask = datas_attention_mask
        else:
            x_test_attention_mask = np.vstack((x_test_attention_mask, datas_attention_mask))

        if y_test is None:
            y_test = labels
        else:
            y_test = np.hstack((y_test, labels))
    print(x_test_input_ids.shape)
    print(x_test_attention_mask.shape)
    print(y_test.shape)

    save_path = "best_model"

    # load best model n plot confusion matrix
    best_model  = keras.models.load_model("{}/model".format(save_path))
    y_pred      = best_model.predict([x_test_input_ids, x_test_attention_mask])

    #print(y_pred)
    print(type(y_pred))
    sns.heatmap(tf.math.confusion_matrix(y_test, y_pred.argmax(axis=1)),
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                annot=True,
                fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.savefig("{}/confusion_matrix.png".format(save_path), bbox_inches='tight')

if __name__ == "__main__":
    main()