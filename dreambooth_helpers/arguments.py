import argparse

from dreambooth_helpers.joepenna_dreambooth_config import JoePennaDreamboothConfigSchemaV1


def parse_arguments() -> JoePennaDreamboothConfigSchemaV1:
    def _get_parser(**parser_kwargs):
        def str2bool(v):
            if isinstance(v, bool):
                return v
            if v.lower() in ("yes", "true", "t", "y", "1"):
                return True
            elif v.lower() in ("no", "false", "f", "n", "0"):
                return False
            else:
                raise argparse.ArgumentTypeError("Boolean value expected.")

        parser = argparse.ArgumentParser(**parser_kwargs)

        parser.add_argument(
            "--config_file_path",
            type=str,
            required=False,
            default=None,
            help="A config file containing all of your variables"
        )

        parser.add_argument(
            "--project_name",
            type=str,
            required=True,
            help="Name of the project"
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug logging",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=1337,
            help="seed for seed_everything",
        )
        parser.add_argument(
            "--token",
            type=str,
            required=True,
            help="Unique token you want to represent your trained model. Ex: firstNameLastName."
        )

        parser.add_argument(
            "--token_only",
            action="store_true",
            help="Train only using the token and no class."
        )

        parser.add_argument(
            "--training_model",
            type=str,
            required=True,
            help="Path to model to train (model.ckpt)"
        )

        parser.add_argument(
            "--training_images",
            type=str,
            required=True,
            help="Path to training images directory"
        )

        parser.add_argument(
            "--regularization_images",
            type=str,
            required=False,
            help="Path to directory with regularization images"
        )
        parser.add_argument(
            "--class_word",
            type=str,
            required=False,
            help="Match class_word to the category of images you want to train. Example: 'man', 'woman', 'dog', or 'artstyle'."
        )

        parser.add_argument(
            "--mirror_probability",
            type=float,
            required=False,
            default=0.25,
            help="mirror Percentage "
                 "Example: if set to 0.5, will flip (mirror) your training images 50% of the time."
                 "This helps expand your dataset without needing to include more training images."
                 "This can lead to worse results for face training since most people's faces are not perfectly symmetrical."
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            required=False,
            default=3e-06,
            help="Set the learning rate. Defaults to 1.0e-06 (0.000001).  Accepts scientific notation."
        )
        parser.add_argument(
            "--save_every_x_steps",
            type=int,
            required=False,
            default=0,
            help="Saves a checkpoint every x steps"
        )
        parser.add_argument(
            "--gpu",
            type=int,
            default=0,
            required=False,
            help="Specify a GPU other than 0 to use for training.  Multi-GPU support is not currently implemented."
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            required=False,
            default=4,
            help="image batch size and number of epochs to perform for iterable datasets"
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            required=False,
            default=2,
            help="number of workers to deploy for data preprocessing"
        )
        parser.add_argument(
            "--epochs",
            type=int,
            required=False,
            default=100,
            help="total number of training iterations to perform per image in your dataset duirng training phase split"
        )
        parser.add_argument(
            "--validation_iterations",
            type=int,
            required=False,
            default=20,
            help="number of validating iterations to perform per image duirng validation split"
        )
        parser.add_argument(
            "--regularization_iterations",
            type=int,
            required=False,
            default=5,
            help="number of regularizing iterations to perform per image during training phase split"
        )
        parser.add_argument(
            "--resolution",
            type=int,
            required=False,
            default=512,
            help="image resolution as N for N^2 in (W * H)"
        )
        parser.add_argument(
            "--resampler",
            type=str,
            required=False,
            choices=["bilinear", "bicubic", "lanczos"],
            default="lanczos"
        )
        parser.add_argument(
            "--center_crop",
            type=bool,
            required=False,
            default=True,
            help="makes ANY polygon your new favorite rhomboid!!!11!1"
        )
        parser.add_argument(
            "--test",
            const="test",
            nargs="?",
            required=False,
            default=None
        )
        parser.add_argument(
            "--accumulated_gradients",
            type=int,
            required=False,
            default=1,
            help="Number of forward pass gradient iterations to process as a single 'batched' iteration during backward propegation: 1 global training step = (1 x accum_grad) iterations"
        )

        return parser

    parser = _get_parser()
    opt, unknown = parser.parse_known_args()

    config = JoePennaDreamboothConfigSchemaV1()

    if opt.config_file_path is not None:
        config.saturate_from_file(config_file_path=opt.config_file_path)
    else:
        config.saturate(
            project_name=opt.project_name,
            seed=opt.seed,
            debug=opt.debug,
            gpu=opt.gpu,
            save_every_x_steps=opt.save_every_x_steps,
            training_images_folder_path=opt.training_images,
            regularization_images_folder_path=opt.regularization_images,
            token=opt.token,
            token_only=opt.token_only,
            class_word=opt.class_word,
            mirror_probability=opt.mirror_probability,
            learning_rate=opt.learning_rate,
            model_repo_id='',
            model_path=opt.training_model,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            epochs=opt.epochs,
            validation_iterations=opt.validation_iterations,
            regularization_iterations=opt.regularization_iterations,
            resolution=opt.resolution,
            resampler=opt.resampler,
            center_crop=opt.center_crop,
            test=opt.test,
            accumulated_gradients=opt.accumulated_gradients
        )

    return config
