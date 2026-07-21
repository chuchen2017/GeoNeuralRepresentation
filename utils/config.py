import os

import yaml


def load_config(parser, default_config=None):
    """Layer a YAML config file on top of an argparse parser's built-in defaults.

    Adds a `--config` flag to `parser` (falling back to `default_config` if not passed
    on the command line). Any key found in the YAML file overrides that argument's
    hardcoded default, but an explicit CLI flag still wins over both.

    Precedence (low -> high): argparse defaults  <  config file  <  CLI flags.

    Args:
        parser (argparse.ArgumentParser): parser with all --arguments already added.
        default_config (str | None): path used when --config isn't passed explicitly.

    Returns:
        argparse.Namespace: the final, merged arguments.
    """
    parser.add_argument('--config', type=str, default=default_config,
                         help='Path to a YAML file overriding the defaults above.')
    known_args, _ = parser.parse_known_args()
    if known_args.config is not None and os.path.isfile(known_args.config):
        with open(known_args.config, 'r') as f:
            config = yaml.safe_load(f) or {}
        parser.set_defaults(**config)
    return parser.parse_args()
