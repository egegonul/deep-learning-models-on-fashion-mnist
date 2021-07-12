import tensorflow as tf

mlp_1 = tf.keras.Sequential([             #models are defined as given in the manual
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5,activation='softmax')
])

mlp_2 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu',use_bias=False),
    tf.keras.layers.Dense(5,activation='softmax')
])

cnn_3 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28,1)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', activation='relu'),
    tf.keras.layers.Conv2D(filters=8, kernel_size=7, padding='valid', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='valid'),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, padding='valid'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='valid'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units=5, activation='softmax')
])

cnn_3_2 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', activation='relu'),
    tf.keras.layers.Conv2D(filters=8, kernel_size=5, padding='valid', activation='relu'),
    tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='valid', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='valid'),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, padding='valid', activation ='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='valid'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units=5, activation='softmax')
])

cnn_5 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', activation='relu'),
    tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='valid', activation='relu'),
    tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='valid', activation='relu'),
    tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='valid', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='valid'),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', activation ='relu'),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', activation ='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='valid'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units=5, activation='softmax')
])

Models={
    "mlp_1":mlp_1,
    "mlp_2":mlp_2,
    "cnn_3":cnn_3,
    "cnn_3_2":cnn_3_2,
    "cnn_5":cnn_5
}