# 2025 DEPA Foundation
#
# This work is dedicated to the public domain under the CC0 1.0 Universal license.
# To the extent possible under law, DEPA Foundation has waived all copyright and 
# related or neighboring rights to this work. 
# CC0 1.0 Universal (https://creativecommons.org/publicdomain/zero/1.0/)
#
# This software is provided "as is", without warranty of any kind, express or implied,
# including but not limited to the warranties of merchantability, fitness for a 
# particular purpose and noninfringement. In no event shall the authors or copyright
# holders be liable for any claim, damages or other liability, whether in an action
# of contract, tort or otherwise, arising from, out of or in connection with the
# software or the use or other dealings in the software.
#
# For more information about this framework, please visit:
# https://depa.world/training/depa_training_framework/

# given the filepath to a python script, and command line arguments, execute it and return the result

import os
import sys
import importlib.util
from .task_base import TaskBase

class FeatureEnggCustom(TaskBase):
    def execute(self, config):
        """Execute the feature engineering custom script.
        execution command:feature_engineering.py /mnt/remote/genomics_lab /mnt/remote/pharmaceutical_company /mnt/remote/computational_biology_lab /mnt/remote/cancer_institute /tmp/"""
        script_path = config["feature_engineering"]["feat_engg_script"]
        data_dirs = []
        for dataset in config.get("datasets", []):
            data_dirs.append(dataset["mount_path"])
        output_path = config["feature_engineering"]["output_path"] if "feature_engineering" in config else config["output_path"]

        # Prepare arguments: [data_dir1, data_dir2, data_dir3, data_dir4, output_path]
        args = data_dirs + [output_path]

        # Save current sys.argv
        old_argv = sys.argv.copy()
        sys.argv = [script_path] + args

        # Import script
        spec = importlib.util.spec_from_file_location("user_feat_engg_mod", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Execute main (expects main() pattern)
        result = module.main()

        # Restore sys.argv
        sys.argv = old_argv

        return result