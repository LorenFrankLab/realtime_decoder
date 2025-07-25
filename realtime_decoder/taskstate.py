import os
from realtime_decoder import logging_base, utils

"""Contains objects for dealing with the task state"""

class TaskStateHandler(logging_base.LoggingClass):
    """An object that gets the current task state"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # defaults
        self._switch_time = 0
        self._taskstate_file = None
        self._is_switch_logged = False

        source = self.config['datasource']
        if source == 'trodes':
            self._taskstate_file = self.config[source]['taskstate_file']
        elif source == 'trodes_file_simulator':
            statescript_log_file = utils.find_unique_file(
                os.path.join(
                    self.config[source]['raw_dir'], 
                    "*.stateScriptLog"
                ),
                "state script log"
            )
            t = utils.get_switch_time(statescript_log_file)

            trodesconf = self.config[source]['config_file']
            root = utils.get_xml_root(trodesconf)
            sr = int(root.find("HardwareConfiguration").attrib['samplingRate'])

            # convert milliseconds to sample number
            self._switch_time = int(t / 1000 * sr)
        else:
            pass

    def get_task_state(self, timestamp):

        """Determine the current task state. If a data source other
        than Trodes is being used, this method always returns a task
        state of 1 (this is to support compatibility with other data
        acquisition systems)"""

        source = self.config['datasource']
        if source == 'trodes':
            return utils.get_last_num(self._taskstate_file)
        elif source == 'trodes_file_simulator':
            taskstate = 1 if timestamp < self._switch_time else 2
            if taskstate == 2 and not self._is_switch_logged:
                self.class_log.info(f"Task state switched to 2")
                self._is_switch_logged = True
            return taskstate    
        else:
            return 1
