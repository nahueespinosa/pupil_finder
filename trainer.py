import cv2
import csv
import os

import numpy as np
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


class Trainer:
    def __init__(self, config, log):
        self.config = config
        self.log = log
        self.images = list()
        self.labels = list()

    def load_data(self, labels_file, images_dir, limit):
        """
            Loads data resizing the images and adjusting the pupil position to
            match the new size.
        """
        with open(labels_file, 'r') as file:
            reader = csv.reader(file, delimiter=' ')
            for index, row in enumerate(reader):
                if 0 < limit <= index:
                    break

                self.log.debug(f"Loading {', '.join(row)}")
                image = cv2.imread(os.path.join(images_dir, row[0]), cv2.IMREAD_COLOR)

                if image is None:
                    self.log.error(f"Could not read image '{row[0]}'")
                    raise FileNotFoundError(f"No such file or directory: '{row[0]}'")

                height, width, layers = image.shape
                image = cv2.resize(
                    image,
                    (self.config['training']['img_width'], self.config['training']['img_height']),
                    interpolation=cv2.INTER_AREA
                )
                y_ratio = image.shape[0] / height
                x_ratio = image.shape[1] / width
                pupil = (round(int(row[1]) * y_ratio), round(int(row[2]) * x_ratio))
                self.images.append(image)
                self.labels.append(pupil)

    def show_data(self):
        self.log.info("Showing images... Press 'q' to quit.")
        for image, label in zip(self.images, self.labels):
            cv2.circle(image, label, 2, (255, 0, 0), 2)
            cv2.imshow("Pupils", image)

            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()

    def get_model(self):
        """
            Returns a compiled convolutional neural network model. Assume that the
            `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
        """
        # Create a convolutional neural network
        model = tf.keras.models.Sequential([

            # Convolutional layer. Learn 24 filters using a 3x3 kernel
            tf.keras.layers.Conv2D(
                24,
                (3, 3),
                activation="relu",
                input_shape=(self.config['training']['img_width'], self.config['training']['img_height'], 3)
            ),

            # Max-pooling layer, using 2x2 pool size
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # Convolutional layer. Learn 36 filters using a 3x3 kernel
            tf.keras.layers.Conv2D(36, (3, 3), activation="relu"),

            # Max-pooling layer, using 2x2 pool size
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # Convolutional layer. Learn 48 filters using a 3x3 kernel
            tf.keras.layers.Conv2D(48, (3, 3), activation="relu"),

            # Max-pooling layer, using 2x2 pool size
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # Flatten units
            tf.keras.layers.Flatten(),

            # Add a hidden layer
            tf.keras.layers.Dense(1000, activation="relu"),

            # Add a hidden layer
            tf.keras.layers.Dense(500, activation="relu"),

            # Add a hidden layer
            tf.keras.layers.Dense(20, activation="relu"),

            # Add an output layer with output units for both axis
            tf.keras.layers.Dense(2)
        ])

        # Train neural network
        model.compile(
            optimizer="rmsprop",
            loss="mean_squared_error",
            metrics=["accuracy"]
        )

        return model

    def train(self, filename):

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(self.images), np.array(self.labels), test_size=self.config['training']['test_size']
        )

        # Get a compiled neural network
        model = self.get_model()

        # Fit model on training data
        model.fit(x_train, y_train, epochs=self.config['training']['epochs'])

        # Evaluate neural network performance
        model.evaluate(x_test, y_test, verbose=2)

        # Save model to file
        model.save(filename)
        self.log.info(f"Model saved to {filename}.")


