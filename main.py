import argparse
import sys
import os

from torch import optim
from torch import nn

from utils.datasets import get_dataloaders, get_num_classes, get_class_labels, DATASETS
from utils.helpers import get_config_section, FormatterNoDuplicate, set_seed, create_safe_directory
from semseg.models.segnet import init_specific_model
from semseg.training import Trainer
from semseg.evaluate import Evaluator
from semseg.utils.modelIO import save_model, load_model, load_metadata

from semseg.models.segnet import MODELS

CONFIG_FILE = "hyperparams.ini"
RES_DIR = "results"

def parse_arguments(args_to_parse):
    """Parse the command line arguments.
        Parameters
        ----------
        args_to_parse: list of str
            Arguments to parse (split on whitespaces).
    """
    description = "PyTorch implementation of convolutional neural network for image classification"
    default_config = get_config_section([CONFIG_FILE], "Preset")
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('-d', '--dataset', help="Path to training data.",
                          default=default_config['dataset'], choices=DATASETS)
    training.add_argument('-i', '--img-size', type=tuple,
                          default=default_config['img_size'],
                          help='Size to transform all images to')
    training.add_argument('-b', '--batch-size', type=int,
                          default=default_config['batch_size'],
                          help='Batch size for training.')
    training.add_argument('--lr', type=float, default=default_config['lr'],
                          help='Learning rate.')
    training.add_argument('-e', '--epochs', type=int,
                          default=default_config['epochs'],
                          help='Maximum number of epochs to run for.')

    # Model Options
    model = parser.add_argument_group('Model specific options')
    model.add_argument('-m', '--model-type',
                       default=default_config['model'], choices=MODELS,
                       help='Type of encoder to use.')

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('-n', '--name', type=str, default=default_config['name'],
                         help="Name of the model for storing and loading purposes.")

    # Evaluation options
    evaluation = parser.add_argument_group('Evaluation specific options')
    evaluation.add_argument('--is-eval-only', action='store_true',
                            default=default_config['is_eval_only'],
                            help='Whether to only evaluate using precomputed model `name`.')
    evaluation.add_argument('--no-test', action='store_true',
                            default=default_config['no_test'],
                            help="Whether or not to compute the test losses.`")

    args = parser.parse_args(args_to_parse)

    return args


def main(args):
    """Main train and evaluation function.
        Parameters
        ----------
        args: argparse.Namespace
            Arguments
    """
    seed = 2 if args.dataset == 'fashion' else 4 # TODO: fix initialisation bug (you may need to change the seed for your data)
    set_seed(seed)
    exp_dir = os.path.join(RES_DIR, args.name)

    if not args.is_eval_only:
        # Create directory (if same name exists, archive the old one)
        create_safe_directory(exp_dir)

        # PREPARES DATA
        train_loader = get_dataloaders(args.dataset,
                                       image_set='train',
                                       image_size = args.img_size,
                                       batch_size=args.batch_size)

         # prepares model
        args.num_classes = get_num_classes(args.dataset)
        model = init_specific_model(args.model_type,
                                    args.img_size,
                                    args.num_classes,
                                    init_from_pretrained=True,
                                    pretrained_type='VGG16')
        # TRAINS
        print('***************************************************')
        print('*                 Training Model                  *')
        print('***************************************************')
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion)
        trainer(train_loader, args.epochs)

        # SAVE MODEL AND EXPERIMENT INFORMATION
        save_model(trainer.model, exp_dir, metadata=vars(args))

    if not args.no_test:
        # LOADS MODEL
        model = load_model(exp_dir, is_gpu=False)
        metadata = load_metadata(exp_dir)
        args.num_classes = get_num_classes(args.dataset)

        # GET TRAIN AND TEST DATA
        train_loader = get_dataloaders(metadata["dataset"],
                                       is_train=True,
                                       batch_size=args.batch_size)

        test_loader = get_dataloaders(metadata["dataset"],
                                      is_train=False,
                                      batch_size=args.batch_size)

        # EVALUATE FOR TRAIN AND TEST
        class_labels = get_class_labels(args.dataset)
        evaluate = Evaluator(model, args.num_classes)

        print('***************************************************')
        print('*            Evaluating Train Accuracy            *')
        print('***************************************************')
        train_accuracy, class_train_accuracy = evaluate(train_loader)
        print('Train accuracy of the network on the 60000 test images: %d %%' % train_accuracy)
        for i in range(args.num_classes):
            print('Accuracy of %5s : %2d %%' % (class_labels[i], class_train_accuracy[i]))

        print('***************************************************')
        print('*            Evaluating Test Accuracy             *')
        print('***************************************************')
        test_accuracy, class_test_accuracy = evaluate(test_loader)
        print('Test accuracy of the network on the 10000 test images: %d %%' % test_accuracy)
        for i in range(args.num_classes):
            print('Accuracy of %5s : %2d %%' % (class_labels[i], class_test_accuracy[i]))

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)