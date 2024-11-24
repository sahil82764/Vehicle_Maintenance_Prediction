from vmpred.logger import logging
from vmpred.exception import vmException
from vmpred.entity.configEntity import DataValidationConfig
from vmpred.entity.artifactEntity import DataIngestionArtifact, DataValidationArtifact
import os, sys
import pandas as pd

# from evidently.model_profile import Profile
# from evidently.model_profile.sections import DataDriftProfileSection
# from evidently.dashboard import Dashboard
# from evidently.dashboard.tabs import DataDriftTab
# import json