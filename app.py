import os
import sys
from vmpred.exception import vmException
from vmpred.logger import logging
from vmpred.pipeline.pipeline import Pipeline
from vmpred.config.configuration import Configuration



def main():
    try:
        config = Configuration()
        pipeline = Pipeline(config = config)
        pipeline.run()
        print("MODEL RUN COMPLETED")

    except Exception as e:
        raise vmException(e,sys) from e


if __name__ == "__main__":
    main()
