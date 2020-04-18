import argparse
import logging.config
import yaml
import os
from trainer import Trainer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a CNN to identify pupil position.')
    parser.add_argument('filename', type=str, help='name of the output file')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='increase verbosity level')
    parser.add_argument('-l', dest='limit', type=int, default=0, help='limit the number of images to load')
    parser.add_argument('-s', '--show-database', dest='show', action='store_true', default=False,
                        help='show database images before processing')
    args = parser.parse_args()

    # Load configuration file
    with open('configuration.yml') as file:
        config = yaml.safe_load(file.read())

    # Load logging file
    os.makedirs('log', exist_ok=True)
    logging.config.dictConfig(config['logging'])
    log = logging.getLogger()
    log.setLevel(logging.DEBUG if args.verbose > 0 else logging.INFO)

    trainer = Trainer(config, log)

    log.info("Loading data...")
    trainer.load_data('dataset/pupil.txt', 'dataset', args.limit)

    if args.show:
        trainer.show_data()

    trainer.train(args.filename)


if __name__ == '__main__':
    main()
