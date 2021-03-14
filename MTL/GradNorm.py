import tensorflow as tf
import matplotlib.pyplot as plt
import math
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dense

seed = 42
random.seed(seed)


class multi_output_mlp(tf.keras.Model):
    def get_config(self):
        super(multi_output_mlp, self).get_config()

    def __init__(self):
        super(multi_output_mlp, self).__init__()
        self.fc_shared = Dense(31, activation=None, name='fc_shared')
        self.fc_1 = Dense(1, activation='sigmoid', name='fc_output1')
        self.fc_2 = Dense(1, activation='sigmoid', name='fc_output2')

    def call(self, x, **kwargs):
        output_shared = self.fc_shared(x)
        output1 = self.fc_1(output_shared)
        output2 = self.fc_2(output_shared)
        return output1, output2


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2,
                                                    random_state=27, stratify=data.target)
y_train = tf.cast(y_train.reshape([-1, 1]), dtype=tf.float32)
y_test = tf.cast(y_test.reshape([-1, 1]), dtype=tf.float32)
model = multi_output_mlp()

w0 = tf.Variable(initial_value=tf.constant(1.0), dtype=tf.float32, trainable=True)
w1 = tf.Variable(initial_value=tf.constant(1.0), dtype=tf.float32, trainable=True)

max_train_steps = 10
loss_initial = []
alpha = tf.constant(0.02, dtype=tf.float32)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

for i in range(max_train_steps):
    with tf.GradientTape(persistent=True) as tape:
        tape.reset()
        output1, output2 = model(X_train)
        loss1 = loss_object(output1, y_train)
        loss2 = loss_object(output2, y_train)
        loss = w0 * loss1 + w1 * loss2
        print("train_steps_{}: total_loss: {}, loss1: {}, loss2: {}".format(i, loss.numpy(), loss1.numpy(),
                                                                            loss2.numpy()))
        if i == 0:
            loss_initial.append(loss1)
            loss_initial.append(loss2)
            loss_initial[0].trainable = False
            loss_initial[1].trainable = False
        loss1_div_initial = tf.divide(loss1, loss_initial[0])
        loss2_div_initial = tf.divide(loss2, loss_initial[1])

        print(loss1_div_initial)
        print(loss2_div_initial)
        ##E_task_L~
        weighted_avg_loss = tf.divide(w0 * loss1_div_initial + w1 * loss2_div_initial, w0 + w1)


        ## r_i_t
        r0t = tf.divide(loss1_div_initial, weighted_avg_loss)
        r1t = tf.divide(loss2_div_initial, weighted_avg_loss)

        ## G_W_i_t
        grad0 = tape.gradient(w0 * loss1, model.trainable_variables[0])
        grad1 = tape.gradient(w1 * loss2, model.trainable_variables[0])
        grad0_norm = tf.norm(grad0, ord=2)
        grad1_norm = tf.norm(grad1, ord=2)

        ## G_W_t_~
        avg_grad_norm = tf.divide(w0 * grad0 + w1 * grad1, w0 + w1)
        # print(avg_grad_norm)
        ## L_grad
        L_grad_tmp0 = tf.multiply(avg_grad_norm, (tf.pow(r0t, alpha)))
        L_grad_tmp1 = tf.multiply(avg_grad_norm, (tf.pow(r1t, alpha)))
        L_grad = tf.abs(grad0_norm - L_grad_tmp0) + tf.abs(grad1_norm - L_grad_tmp1)

        standard_gradients = tape.gradient(loss, model.trainable_variables)
        w0_grad = tape.gradient(L_grad, w0)
        w1_grad = tape.gradient(L_grad, w1)
        ## 更新
        optimizer.apply_gradients(zip(standard_gradients, model.trainable_variables))
        optimizer.apply_gradients(zip([w0_grad], [w0]))
        optimizer.apply_gradients(zip([w1_grad], [w1]))
