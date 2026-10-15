import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solve import cli

if __name__ == "__main__":
    cli.main_run_examples()
