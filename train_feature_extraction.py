import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
import time

nb_classes=43
learning_rate = 0.001
EPOCHS = 2
BATCH_SIZE = 128

# TODO: Load traffic signs data.
training_file = '../CarND-Traffic-Signs/train.p'
testing_file = '../CarND-Traffic-Signs/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


# TODO: Split data into training and validation sets.
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
resized = tf.image.resize_images(x, [227,227])

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
# Hyperparameters
mu = 0
sigma = 0.1
# Fully connected layer
fc8_W = tf.Variable(tf.truncated_normal(shape, mean = mu, stddev = sigma*sigma))
fc8_b = tf.Variable(tf.zeros(nb_classes))
logits = tf.matmul(fc7, fc8_W) + fc8_b

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

one_hot_y = tf.one_hot(y, nb_classes)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)

loss_op = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

init_op = tf.initialize_all_variables()

# MODEL EVALUTATION
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def eval_data(X_data, y_data):
    num_examples = len(X_data)
    total_acc, total_loss = 0, 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        loss, acc = sess.run([loss_op, accuracy_op],
                             feed_dict={x: batch_x, y: batch_y})
        total_acc += (acc * len(batch_x))
        total_loss += (loss * len(batch_x))

    return total_loss / num_examples, total_acc / num_examples

# TODO: Train and evaluate the feature extraction model.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    num_examples = len(X_train)
    print("Training...\n \n")

    for epoch in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            loss, accur = sess.run([train_op, accuracy_op],
                                   feed_dict={x: batch_x, y: batch_y})

        loss_val, acc_val = eval_data(X_validation, y_validation)

        print("Epoch", epoch+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Training Loss =", loss)
        print("Training Accuracy =", accur)
        print("Validation Loss =", loss_val)
        print("Validation Accuracy =", acc_val)
        print("")
