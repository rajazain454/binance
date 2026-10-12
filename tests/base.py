"""Base test case providing isolated temporary directories and logging for tests."""

import logging
import os
import shutil
import tempfile
import unittest

from utils import close_logger_handlers, setup_logger


class BaseTestCase(unittest.TestCase):
    """Base class for all QuantLab tests.

    Ensures each test execution gets an isolated temporary directory
    for state_file, trade_ledger, logs_dir, and data_dir, and redirects
    logging away from repo root and logs/ directory to a temp log file.
    """

    def setUp(self):
        super().setUp()
        self.test_dir = tempfile.mkdtemp(prefix="quantlab_test_")
        self.logs_dir = os.path.join(self.test_dir, "logs")
        self.data_dir = os.path.join(self.test_dir, "data")
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        self.state_file = os.path.join(self.test_dir, "test_state.json")
        self.trade_ledger = os.path.join(self.logs_dir, "test_trades.csv")
        self.log_file = os.path.join(self.logs_dir, "test.log")

        # Point logger to the isolated temporary log file
        setup_logger(self.log_file)

    def tearDown(self):
        # Close file handlers first so Windows file lock is released
        close_logger_handlers("quantlab")
        shutil.rmtree(self.test_dir, ignore_errors=True)
        super().tearDown()

    def get_test_paths(self):
        """Returns a paths dictionary with all paths pointing into the temp directory."""
        return {
            "state_file": self.state_file,
            "trade_ledger": self.trade_ledger,
            "logs_dir": self.logs_dir,
            "data_dir": self.data_dir,
            "models_dir": "models",
        }

    def override_config_paths(self, config):
        """Overrides config['paths'] with isolated temporary directory paths."""
        if "paths" not in config:
            config["paths"] = {}
        config["paths"]["state_file"] = self.state_file
        config["paths"]["trade_ledger"] = self.trade_ledger
        config["paths"]["logs_dir"] = self.logs_dir
        config["paths"]["data_dir"] = self.data_dir
        if "models_dir" not in config["paths"]:
            config["paths"]["models_dir"] = "models"
        return config
