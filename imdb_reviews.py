from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import losses

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

def main():
    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews",
        split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True
    )

    train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
    print(train_examples_batch)
    print(train_labels_batch)

    embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    hub_layer = hub.KerasLayer(
        embedding,
        input_shape=[],
        dtype=tf.string,
        trainable=True
    )
    print(hub_layer(train_examples_batch[:3]))

    model = Sequential([
        hub_layer,
        layers.Dense(8, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(1)
    ])
    model.summary()
    model.compile(
        optimizer="adam",
        loss=losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    
    history = model.fit(
        train_data.shuffle(100000).batch(512),
        epochs=20,
        validation_data=validation_data.batch(512)
    )

    results = model.evaluate(test_data.batch(512))
    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))



if __name__ == "__main__":
    main()