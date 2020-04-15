import argparse
import cv2
import csv
import logging.config
import yaml
import os

log = logging.getLogger(__name__)


def main(args):
    load_config(args.config)

    log.setLevel(logging.DEBUG if args.verbose > 0 else logging.INFO)
    log.info("Loading data...")
    images, labels = load_data(args.labels, args.dataset)

    log.info("Showing images... Press 'q' to quit.")
    for image, label in zip(images, labels):
        cv2.circle(image, label, 2, (255, 0, 0), 2)
        cv2.imshow("Pupils", image)

        if cv2.waitKey(1) == ord('q'):
            break


def load_config(path, default_level=logging.INFO):
    """
    Load logging configuration.
    :param path: yaml configuration file
    :param default_level: level of debugging
    :return: None
    """
    if os.path.exists(path):
        with open(path) as file:
            config = yaml.safe_load(file.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        log.warning(f"Could not find file '{path}'. Using basic logging.")


def load_data(labels_file, img_dir):
    """
    Loads the dataset resizing the images and adjusting the pupil position to
    match the new size.
    :param labels_file: a file listing all images and pupil position in the
        format "subdir/image_name.png 36 42"
    :param img_dir: directory containing all images
    :return: a tuple containing a list of images and a list of their
        corresponding labels.
    """
    images = list()
    labels = list()

    with open(labels_file, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            log.debug(f"Loading {', '.join(row)}")
            image = cv2.imread(os.path.join(img_dir, row[0]), cv2.IMREAD_COLOR)
            height, width, layers = image.shape
            image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_AREA)
            y_ratio = image.shape[0] / height
            x_ratio = image.shape[1] / width
            pupil = (round(int(row[1]) * y_ratio), round(int(row[2]) * x_ratio))
            images.append(image)
            labels.append(pupil)

    return images, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ANN to identify pupil position.')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='increase verbosity level')
    parser.add_argument('-c', dest='config', default='configuration.yml', help='configuration file')
    parser.add_argument('-l', dest='labels', default='dataset/pupil.txt', help='labels file')
    parser.add_argument('-d', dest='dataset', default='dataset', help='image dataset directory')
    main(parser.parse_args())

