import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

K = tf.keras.backend


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, field_dict, field_dim, input_dim, output_dim=30, **kwargs):
        self.field_dict = field_dict
        self.field_dim = field_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.field_weight_matrix = self.add_weight(name='field_weight_matrix',shape=(self.field_dim,self.field_dim),
                                                   initializer='glorot_uniform',trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        self.field_cross = tf.zeros(shape=[1])
        for i in range(self.input_dim):
            for j in range(i + 1, self.input_dim):
                weight = tf.reduce_sum(
                    tf.multiply(self.kernel[i], self.kernel[j]))
                weight = tf.multiply(weight, tf.multiply(x[:, i], x[:, j]))
                value = tf.multiply(weight,self.field_weight_matrix[self.field_dict[i],self.field_dict[j]])
                self.field_cross = tf.add(self.field_cross, value)
        return self.field_cross

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1


def FFM(feature_dim, field_dict, field_dim, output_dim=30):
    inputs = tf.keras.Input((feature_dim,))
    liner = tf.keras.layers.Dense(1)(inputs)
    cross = MyLayer(field_dict, field_dim, feature_dim, output_dim)(inputs)
    cross = tf.keras.layers.Reshape((1,))(cross)
    add = tf.keras.layers.Add()([liner, cross])
    predictions = tf.keras.layers.Activation('sigmoid')(add)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model


def train():
    field_dict = {i: i // 5 for i in range(30)}
    ffm = FFM(30, field_dict, 6, 30)
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2,
                                                        random_state=27, stratify=data.target)
    ffm.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))
    return ffm


if __name__ == '__main__':
    ffm = train()
