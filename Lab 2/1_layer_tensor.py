from threading import current_thread, currentThread
import tensorflow as tf

W = tf.Variable([[1.0],
                  [2]])
b = tf.Variable(3.0)

def h(x) -> tf.Tensor:
    return tf.nn.relu(tf.matmul(x, W) + b)

def predict(x: tf.Variable) -> tf.Tensor:
    return h(x)

def loss(y_true, y_pred) -> tf.Tensor:
    return tf.reduce_mean(tf.square(y_true - y_pred))

def train(x, y, lr: float = 0.01) -> None:
    with tf.GradientTape() as g:
        current_loss = loss(y, predict(x))

    print(current_loss)

    dW, db = g.gradient(current_loss, [W, b])

    print(dW)
    print(db)

    W.assign_sub(lr*dW)
    b.assign_sub(lr*db)

def main() -> None:
    X = tf.Variable([[0.0, 0], 
                     [0, 1],
                     [1, 0],
                     [1, 1]])
    Y = tf.Variable([[0.0],
                     [1],
                     [1],
                     [0]])
    for epoch in range(10000):
        train(X, Y)
    print(predict(X))

if __name__ == "__main__":
    main()