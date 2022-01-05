import tensorflow as tf

W1 = tf.Variable(tf.eye(2, 2))
b1 = tf.Variable(tf.zeros((2,)))

W2 = tf.Variable(tf.eye(2, 1))
b2 = tf.Variable(tf.zeros((1,)))

def h(x: tf.Tensor) -> tf.Tensor:
    return tf.nn.relu(tf.matmul(x, W1) + b1)

def predict(x: tf.Tensor) -> tf.Tensor:
    return tf.sigmoid(tf.matmul(h(x), W2) + b2)

def loss(y_true, y_pred) -> tf.Tensor:
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))

def train(x_train, y_train, lr: float = 0.001) -> tf.Tensor:
    with tf.GradientTape() as tape:
        train_loss = loss(y_train, predict(x_train))

    dW1, db1, dW2, db2 = tape.gradient(train_loss, [W1, b1, W2, b2])

    W1.assign_sub(lr*dW1)
    b1.assign_sub(lr*db1)

    W2.assign_sub(lr*dW2)
    b2.assign_sub(lr*db2)

    return train_loss

def main() -> None:
    X = tf.Variable([[0.0, 0], 
                     [0, 1],
                     [1, 0],
                     [1, 1]])
    Y = tf.Variable([[0.0],
                     [1],
                     [1],
                     [0]])
    with open("xor_loss.txt", 'w') as f:
        for epoch in range(100000):
            train_loss = float(train(X, Y, 0.01))
            print(train_loss)
            f.write("{}\n".format(train_loss))
    
    print(W1)
    print(b1)

    print(W2)
    print(b2)

    print(predict(X))

if __name__ == "__main__":
    main()