# Pupil Finder: Models

## Model_1.h5
These are the details of this model implementation:
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        strides=(2, 2),
        activation="relu",
        input_shape=(self.img_width, self.img_height, 1)
    ),
    tf.keras.layers.Conv2D(64, (3, 3), strides=(3, 3), activation="relu"),
    tf.keras.layers.Conv2D(128, (3, 3), strides=(3, 3), activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1500, activation="relu"),
    tf.keras.layers.Dense(500, activation="relu"),
    tf.keras.layers.Dense(500, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(2)
])

model.compile(
    optimizer="rmsprop",
    loss="mean_squared_error",
    metrics=["accuracy"]
)
```

And these are the results obtained after 50 epochs:
```
Epoch 50/50
9064/9064 [==============================] - 8s 880us/sample - loss: 3.1266 - accuracy: 0.8917
6043/6043 - 2s - loss: 5.1154 - accuracy: 0.8751
INFO -- Model saved to models/model_1.h5.
```

To test this trained model with a camera you can use this command:
```
$ python test_model.py models/model_1.h5
```