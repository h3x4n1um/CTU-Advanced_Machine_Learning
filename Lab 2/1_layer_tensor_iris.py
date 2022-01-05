from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

W = tf.Variable(tf.eye(4, 3))
b = tf.Variable(tf.zeros((3,)))

def predict(x: tf.Tensor) -> tf.Tensor:
	return tf.nn.softmax(tf.matmul(x, W) + b)

def loss(y_true, y_pred) -> tf.Tensor:
	return tf.reduce_mean(-tf.reduce_sum(y_true*tf.math.log(y_pred), 1))

def train(x_train, y_train, lr: float = 0.001) -> None:
	with tf.GradientTape() as tape:
		train_loss = loss(y_train, predict(x_train))

	dW, db = tape.gradient(train_loss, [W, b])

	W.assign_sub(lr*dW)
	b.assign_sub(lr*db)

def main() -> None:
	col_names = ["sepal length in cm",
				 "sepal width in cm",
				 "petal length in cm",
				 "petal width in cm",
				 "class"]

	df = pd.read_csv("iris.data",
					 names=col_names)

	df_x = df.drop("class", axis=1)

	enc_y	= LabelEncoder()
	df_y	= enc_y.fit_transform(df["class"])

	x_train, x_test, y_train, y_test = train_test_split(df_x,
														df_y,
														test_size=0.2,
														random_state=1911)

	x_train	= tf.Variable(x_train	, dtype=tf.float32)
	x_test	= tf.Variable(x_test	, dtype=tf.float32)

	y_train	= tf.one_hot(y_train, len(enc_y.classes_))
	y_test	= tf.one_hot(y_test	, len(enc_y.classes_))

	f_loss_train	= open("iris_loss_train.txt", 'w')
	f_loss_test		= open("iris_loss_test.txt"	, 'w')

	f_accurary_train	= open("iris_accurary_train.txt", 'w')
	f_accurary_test		= open("iris_accurary_test.txt"	, 'w')

	for epoch in range(100000):
		train(x_train, y_train, 0.01)

		y_true_train	= tf.argmax(y_train	, 1)
		y_true_test		= tf.argmax(y_test	, 1)

		y_pred_train	= predict(x_train)
		y_pred_test		= predict(x_test)

		train_loss 	= float(loss(y_train, y_pred_train))
		test_loss 	= float(loss(y_test	, y_pred_test))

		train_accuracy	= accuracy_score(y_true_train	, tf.argmax(y_pred_train, 1))
		test_accuracy	= accuracy_score(y_true_test	, tf.argmax(y_pred_test	, 1))
		
		print("Epoch {}:".format(epoch))
		print("\ttrain loss:\t{}\ttrain accuracy:\t{}"	.format(train_loss	, train_accuracy))
		print("\ttest loss:\t{}\ttest accuracy:\t{}"	.format(test_loss	, test_accuracy))

		f_loss_train.write("{}\n".format(train_loss))
		f_loss_test .write("{}\n".format(test_loss))

		f_accurary_train.write("{}\n".format(train_accuracy))
		f_accurary_test .write("{}\n".format(test_accuracy))

	f_loss_train.close()
	f_loss_test .close()

	f_accurary_train.close()
	f_accurary_test .close()

	disp = ConfusionMatrixDisplay(confusion_matrix(tf.argmax(y_test			, 1),
												   tf.argmax(predict(x_test), 1)),
								  display_labels=enc_y.classes_)
	disp.plot()

	plt.show()

if __name__ == "__main__":
	main()
