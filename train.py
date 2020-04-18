import argparse
import cv2
import csv
import logging.config
import yaml
import os

import numpy as np
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def main(args):
    log.setLevel(logging.DEBUG if args.verbose > 0 else logging.INFO)
    log.info("Loading data...")

    # Get image arrays and labels for all image files
    images, labels = load_data('dataset/pupil.txt', 'dataset', args.limit)

    if args.show:
        log.info("Showing images... Press 'q' to quit.")
        for image, label in zip(images, labels):
            cv2.circle(image, label, 2, (255, 0, 0), 2)
            cv2.imshow("Pupils", image)

            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=config['general']['test_size']
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=config['general']['epochs'])

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    model.save(args.filename)
    print(f"Model saved to {args.filename}.")


def load_config(path, default_level=logging.INFO):
    """
    Load logging configuration.
    :param path: yaml configuration file
    :param default_level: level of debugging
    :return: None
    """
    if os.path.exists(path):
        with open(path) as file:
            config_dict = yaml.safe_load(file.read())
        logging.config.dictConfig(config_dict['logging'])
    else:
        log.error(f"Could not find file '{path}'. Using basic logging.")
        raise FileNotFoundError

    return config_dict


def load_data(labels_file, img_dir, limit):
    """
    Loads the dataset resizing the images and adjusting the pupil position to
    match the new size.
    :param labels_file: a file listing all images and pupil position in the
        format "subdir/image_name.png 36 42"
    :param img_dir: directory containing all images
    :param limit: limit the number of images to load
    :return: a tuple containing a list of images and a list of their
        corresponding labels.
    """
    images = list()
    labels = list()

    with open(labels_file, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for index, row in enumerate(reader):
            if 0 < limit <= index:
                break

            log.debug(f"Loading {', '.join(row)}")
            image = cv2.imread(os.path.join(img_dir, row[0]), cv2.IMREAD_COLOR)
            height, width, layers = image.shape
            image = cv2.resize(
                image,
                (config['general']['img_width'], config['general']['img_height']),
                interpolation=cv2.INTER_AREA
            )
            y_ratio = image.shape[0] / height
            x_ratio = image.shape[1] / width
            pupil = (round(int(row[1]) * y_ratio), round(int(row[2]) * x_ratio))
            images.append(image)
            labels.append(pupil)

    return images, labels


def get_model():
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
            input_shape=(config['general']['img_width'], config['general']['img_height'], 3)
        ),

        # Max-pooling layer, using 3x3 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        # Convolutional layer. Learn 36 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(36, (3, 3), activation="relu"),

        # Max-pooling layer, using 3x3 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        # Convolutional layer. Learn 48 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(48, (3, 3), activation="relu"),

        # Max-pooling layer, using 3x3 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer
        tf.keras.layers.Dense(4000, activation="relu"),

        # Add a hidden layer
        tf.keras.layers.Dense(2000, activation="relu"),

        # Add a hidden layer
        tf.keras.layers.Dense(500, activation="relu"),

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


log = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN to identify pupil position.')
    parser.add_argument('filename', help='name of the output file')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='increase verbosity level')
    parser.add_argument('-s', '--show-database', dest='show', action='store_true', default=False,
                        help='show database images before processing')
    parser.add_argument('-l', dest='limit', type=int, default=0, help='limit the number of images to load')
    config = load_config('configuration.yml')
    main(parser.parse_args())

