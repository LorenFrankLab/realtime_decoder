
import time
import logging
import numpy as np
import seaborn as sns
import pyqtgraph as pg

from collections import OrderedDict
from mpi4py import MPI
from matplotlib import cm
from PyQt5.QtCore import (
    Qt, pyqtSignal, QThread, QTimer, QElapsedTimer
)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout,
    QLineEdit, QGroupBox, QHBoxLayout, QDialog, QTabWidget,
    QPushButton, QLabel, QSpinBox, QSlider, QStatusBar,
    QFileDialog, QMessageBox, QRadioButton, QTextEdit, QStatusBar
)

from realtime_decoder import base, messages

_DEFAULT_GUI_PARAMS = {
    "colormap" : "rocket"
}

def _show_message(parent, text, *, kind=None):
    """Show a pop up message"""

    if kind is None:
        kind = QMessageBox.NoIcon
    elif kind == "question":
        kind = QMessageBox.Question
    elif kind == "information":
        kind = QMessageBox.Information
    elif kind == "warning":
        kind = QMessageBox.Warning
    elif kind == "critical":
        kind = QMessageBox.Critical
    else:
        msg = QMessageBox(parent)
        msg.setText(f"Invalid message kind '{kind}' specified")
        msg.setIcon(QMessageBox.Critical)
        msg.addButton(QMessageBox.Ok)
        msg.exec_()
        return

    msg = QMessageBox(parent)
    msg.setText(text)
    msg.setIcon(kind)
    msg.addButton(QMessageBox.Ok)
    msg.exec_()

####################################################################################
# Interfaces
####################################################################################

class DialogSendInterface(base.StandardMPISendInterface):
    """Sending interface object for GUI process"""

    def __init__(self, comm, rank, config):
        super().__init__(comm, rank, config)

    def send_main_params(self, params):
        self.comm.send(
            obj=params,
            dest=self.config['rank']['supervisor'][0],
            tag=messages.MPIMessageTag.GUI_PARAMETERS
        )

    def send_ripple_params(self, params):
        """Send parameters relevant for ripple detection"""

        for rank in self.config['rank']['ripples']:
            self.comm.send(
                obj=params, dest=rank,
                tag=messages.MPIMessageTag.GUI_PARAMETERS
            )

    def send_encoding_model_params(self, params):
        """Send parameters relevant for encoding models"""

        ranks = (
            self.config['rank']['encoders'] +
            self.config['rank']['decoders']
        )
        for rank in ranks:
            self.comm.send(
                obj=params, dest=rank,
                tag=messages.MPIMessageTag.GUI_PARAMETERS
            )

    def send_startup(self):
        """Send message telling MainProcess to initiate startup
        sequence"""

        self.comm.send(
            obj=messages.StartupSignal(),
            dest=self.config['rank']['supervisor'][0],
            tag=messages.MPIMessageTag.COMMAND_MESSAGE
        )

    def send_shutdown(self):
        """Send message telling MainProcess to initiate
        shutdown sequence"""

        self.comm.send(
            obj=messages.TerminateSignal(),
            dest=self.config['rank']['supervisor'][0],
            tag=messages.MPIMessageTag.COMMAND_MESSAGE
        )

class GenericGuiRecvInterface(base.MPIRecvInterface):
    """Interface object for receiving a non neural data (MPI)
    message"""

    def __init__(
        self, comm, rank, config,
        msg_dtype, msg_tag, msg_handler
    ):
        super().__init__(comm, rank, config)
        self._msg_dtype = msg_dtype
        self._msg_tag = msg_tag
        self._msg_handler = msg_handler

        self._msg_buffer = bytearray(self._msg_dtype.itemsize)
        self._mpi_status = MPI.Status()
        self._req = self.comm.Irecv(
            buf=self._msg_buffer,
            tag=self._msg_tag
        )

    def receive(self):
        """Test whether a message is available, and if so, process it"""

        rdy = self._req.Test(status=self._mpi_status)
        if rdy:
            msg = np.frombuffer(self._msg_buffer, dtype=self._msg_dtype)
            self._msg_handler.handle_message(msg, self._mpi_status)
            self._req = self.comm.Irecv(
                buf=self._msg_buffer,
                tag=self._msg_tag
            )

class ArmEventsRecvInterface(base.MPIRecvInterface):
    """Interface object for receiving data about the number of
    rewarded arms"""

    def __init__(self, comm, rank, config, msg_handler):
        super().__init__(comm, rank, config)
        self._msg_handler = msg_handler
        self._msg_buffer = np.zeros(
            len(config['encoder']['position']['arm_coords']),
            dtype='=i4'
        )
        self._mpi_status = MPI.Status()
        self._req = self.comm.Irecv(
            buf=self._msg_buffer,
            tag=messages.MPIMessageTag.ARM_EVENTS
        )

    def receive(self):
        """Test whether a message is available, and if so, process it"""

        rdy = self._req.Test(status=self._mpi_status)
        if rdy:
            self._msg_handler.handle_message(
                self._msg_buffer, self._mpi_status
            )
            self._req = self.comm.Irecv(
                buf=self._msg_buffer,
                tag=messages.MPIMessageTag.ARM_EVENTS
            )


####################################################################################
# Qt Classes
####################################################################################

class TabbedDialog(QDialog):
    """Dialog window used for control and modifying parameters"""

    def __init__(self, parent, comm, rank, config):
        super().__init__(parent)
        self._rank = rank
        self._config = config
        self.setWindowTitle("Parameters/Control")

        self._send_interface = DialogSendInterface(
            comm, rank, config
        )
        self._class_log = logging.getLogger(
            name=f'{self.__class__.__name__}'
        )

        self._main_params = messages.GuiMainParameters()
        self._ripple_params = messages.GuiRippleParameters()
        self._model_params = messages.GuiEncodingModelParameters()

        self._timer = None
        timer_interval = self._config['gui']['send_interval']
        if timer_interval > 0:
            self._timer = QTimer()
            self._timer.setInterval(timer_interval*1000)
            self._timer.timeout.connect(self._send_all_params)

        self._tab_widget = QTabWidget()
        dialog_layout = QVBoxLayout(self)
        dialog_layout.addWidget(self._tab_widget)

        self._setup_params_tab()
        self._setup_control_tab()

    def _setup_params_tab(self):
        """Set up the parameters tab"""

        self._params_tab = QWidget()

        layout = QGridLayout(self._params_tab)

        # add helpful tool tips!
        self._setup_stim_params(layout)
        self._setup_ripple_params(layout)
        self._setup_model_params(layout)

        self._tab_widget.addTab(self._params_tab, self.tr("Parameters"))

    def _setup_stim_params(self, layout):
        """Set up the stimulation parameters tab"""

        self._stim_label = QLabel(self.tr("Stimulation"))
        self._stim_label.setStyleSheet("font-weight: bold")
        layout.addWidget(self._stim_label, 0, 0)

        self._setup_instructive_task(layout)
        self._setup_target_arm(layout)
        self._setup_post_thresh(layout)
        self._setup_max_center_well_distance(layout)
        self._setup_num_above_thresh(layout)
        self._setup_min_duration(layout)
        self._setup_well_angle_range(layout)
        self._setup_within_angle_range(layout)
        self._setup_rotate_180(layout)
        self._setup_replay_stim(layout)
        self._setup_ripple_stim(layout)
        self._setup_head_direction_stim(layout)

    def _setup_instructive_task(self, layout):
        """Set up the label showing whether this is an instructive
        task or not"""

        text = 'Task: '
        if self._config['stimulation']['instructive']:
            text += 'Instructive'
        else:
            text += 'Non-instructive'

        self._instructive_task_label = QLabel(self.tr(text))
        layout.addWidget(self._instructive_task_label, 1, 0)

    def _setup_target_arm(self, layout):
        """Set up widgets and data related to the replay target arm"""

        self._target_arm_label = QLabel(self.tr("Replay target arm"))
        layout.addWidget(self._target_arm_label, 2, 0)

        self._target_arm_edit = QLineEdit()
        layout.addWidget(self._target_arm_edit, 2, 1)

        self._target_arm_button = QPushButton(self.tr("Update"))
        self._target_arm_button.pressed.connect(self._check_target_arm)
        layout.addWidget(self._target_arm_button, 2, 2)

        arm = self._config['stimulation']['replay']['target_arm']

        if self._config['stimulation']['instructive']:
            value = "Handled by stim decider"
            self._target_arm_button.setEnabled(False)
        else:
            value = str(arm)

        self._target_arm_edit.setText(value)
        self._main_params.replay_target_arm = arm

    def _setup_post_thresh(self, layout):
        """Set up widgets and data related to posterior
        threshold (see documentation for more info)"""

        self._post_label = QLabel(self.tr("Posterior threshold"))
        self._post_label.setToolTip("Just a helpful tool tip")
        layout.addWidget(self._post_label, 3, 0)

        self._post_edit = QLineEdit()
        layout.addWidget(self._post_edit, 3, 1)

        self._post_thresh_button = QPushButton(self.tr("Update"))
        self._post_thresh_button.pressed.connect(self._check_post_thresh)
        layout.addWidget(self._post_thresh_button, 3, 2)

        value = float(
            self._config['stimulation']['replay']['primary_arm_threshold']
        )
        self._post_edit.setText(str(value))
        self._main_params.posterior_threshold = value

    def _setup_max_center_well_distance(self, layout):
        """Set up widgets and data related to the maximum distance from
        the center well that is allowed to qualify a replay event"""

        self._max_center_well_label = QLabel(self.tr("Max center well distance"))
        layout.addWidget(self._max_center_well_label, 4, 0)

        self._max_center_well_edit = QLineEdit()
        layout.addWidget(self._max_center_well_edit, 4, 1)

        self._max_center_well_button = QPushButton(self.tr("Update"))
        self._max_center_well_button.pressed.connect(self._check_max_center_well)
        layout.addWidget(self._max_center_well_button, 4, 2)

        value = float(self._config['stimulation']['max_center_well_dist'])
        self._max_center_well_edit.setText(str(value))
        self._main_params.max_center_well_distance = value

    def _setup_num_above_thresh(self, layout):
        """Set up widgets and data related to number of electrode
        groups above threshold (see documentation)"""

        self._num_above_label = QLabel(self.tr("Num. trodes above threshold"))
        layout.addWidget(self._num_above_label, 5, 0)

        self._num_above_edit = QLineEdit()
        layout.addWidget(self._num_above_edit, 5, 1)

        self._num_above_button = QPushButton(self.tr("Update"))
        self._num_above_button.pressed.connect(self._check_num_above)
        layout.addWidget(self._num_above_button, 5, 2)

        value = int(self._config['stimulation']['ripples']['num_above_thresh'])
        self._num_above_edit.setText(str(value))
        self._main_params.num_above_threshold = value

    def _setup_min_duration(self, layout):
        """Set up widgets and data related to the minimum amount of time
        needed to qualify as a head direction event"""

        self._min_duration_label = QLabel(self.tr("Min duration head angle"))
        layout.addWidget(self._min_duration_label, 6, 0)

        self._min_duration_edit = QLineEdit()
        layout.addWidget(self._min_duration_edit, 6, 1)

        self._min_duration_button = QPushButton(self.tr("Update"))
        self._min_duration_button.pressed.connect(self._check_min_duration)
        layout.addWidget(self._min_duration_button, 6, 2)

        value = float(
            self._config['stimulation']['head_direction']['min_duration']
        )
        self._min_duration_edit.setText(str(value))
        self._main_params.min_duration = value

    def _setup_well_angle_range(self, layout):
        """Set up widgets and data related to the well angle range
        (see documentation)"""

        self._well_angle_range_label = QLabel(self.tr("Well angle range"))
        layout.addWidget(self._well_angle_range_label, 7, 0)

        self._well_angle_range_edit = QLineEdit()
        layout.addWidget(self._well_angle_range_edit, 7, 1)

        self._well_angle_range_button = QPushButton(self.tr("Update"))
        self._well_angle_range_button.pressed.connect(self._check_well_angle_range)
        layout.addWidget(self._well_angle_range_button, 7, 2)

        value = float(
            self._config['stimulation']['head_direction']['well_angle_range']
        )
        self._well_angle_range_edit.setText(str(value))
        self._main_params.well_angle_range = value

    def _setup_within_angle_range(self, layout):
        """Set up widgets and data related to the within angle range
        (see documentation)"""

        self._within_angle_range_label = QLabel(self.tr("Within angle range"))
        layout.addWidget(self._within_angle_range_label, 8, 0)

        self._within_angle_range_edit = QLineEdit()
        layout.addWidget(self._within_angle_range_edit, 8, 1)

        self._within_angle_range_button = QPushButton(self.tr("Update"))
        self._within_angle_range_button.pressed.connect(self._check_within_angle_range)
        layout.addWidget(self._within_angle_range_button, 8, 2)

        value = float(
            self._config['stimulation']['head_direction']['within_angle_range']
        )
        self._within_angle_range_edit.setText(str(value))
        self._main_params.within_angle_range = value

    def _setup_rotate_180(self, layout):
        """Set up widgets and data related to whether or not to rotate
        the head angle vector by 180 degrees"""

        self._rotate_180_label = QLabel(self.tr("Rotate head vector 180"))
        layout.addWidget(self._rotate_180_label, 9, 0)

        self._rotate_180_yes = QRadioButton(self.tr("YES"))
        self._rotate_180_no = QRadioButton(self.tr("NO"))
        rotate_180_layout = QHBoxLayout()
        rotate_180_layout.addWidget(self._rotate_180_yes)
        rotate_180_layout.addWidget(self._rotate_180_no)
        rotate_180_group_box = QGroupBox()
        rotate_180_group_box.setLayout(rotate_180_layout)
        layout.addWidget(rotate_180_group_box, 9, 1)

        self._rotate_180_button = QPushButton(self.tr("Update"))
        self._rotate_180_button.pressed.connect(self._check_rotate_180)
        layout.addWidget(self._rotate_180_button, 9, 2)

        if self._config['stimulation']['head_direction']['rotate_180']:
            self._rotate_180_yes.setChecked(True)
            self._main_params.rotate_180 = True
        else:
            self._rotate_180_no.setChecked(True)
            self._main_params.rotate_180 = False

    def _setup_replay_stim(self, layout):
        """Set up widgets and data related to whether replay
        stimulation is enabled"""

        self._replay_stim_label = QLabel(self.tr("Replay stim"))
        layout.addWidget(self._replay_stim_label, 10, 0)

        self._replay_stim_on = QRadioButton(self.tr("ON"))
        self._replay_stim_off = QRadioButton(self.tr("OFF"))
        replay_layout = QHBoxLayout()
        replay_layout.addWidget(self._replay_stim_on)
        replay_layout.addWidget(self._replay_stim_off)
        replay_group_box = QGroupBox()
        replay_group_box.setLayout(replay_layout)
        layout.addWidget(replay_group_box, 10, 1)

        self._replay_stim_button = QPushButton(self.tr("Update"))
        self._replay_stim_button.pressed.connect(self._check_replay_stim)
        layout.addWidget(self._replay_stim_button, 10, 2)

        if self._config['stimulation']['replay']['enabled']:
            self._replay_stim_on.setChecked(True)
            self._main_params.replay_stim_enabled = True
        else:
            self._replay_stim_off.setChecked(True)
            self._main_params.replay_stim_enabled = False

    def _setup_ripple_stim(self, layout):
        """Set up widgets and data related to whether stimulation
        based on ripple events is enabled"""

        self._ripple_stim_label = QLabel(self.tr("Ripple stim"))
        layout.addWidget(self._ripple_stim_label, 11, 0)

        self._ripple_stim_on = QRadioButton(self.tr("ON"))
        self._ripple_stim_off = QRadioButton(self.tr("OFF"))
        ripple_layout = QHBoxLayout()
        ripple_layout.addWidget(self._ripple_stim_on)
        ripple_layout.addWidget(self._ripple_stim_off)
        ripple_group_box = QGroupBox()
        ripple_group_box.setLayout(ripple_layout)
        layout.addWidget(ripple_group_box, 11, 1)

        self._ripple_stim_button = QPushButton(self.tr("Update"))
        self._ripple_stim_button.pressed.connect(self._check_ripple_stim)
        layout.addWidget(self._ripple_stim_button, 11, 2)

        if self._config['stimulation']['ripples']['enabled']:
            self._ripple_stim_on.setChecked(True)
            self._main_params.ripple_stim_enabled = True
        else:
            self._ripple_stim_off.setChecked(True)
            self._main_params.ripple_stim_enabled = False

    def _setup_head_direction_stim(self, layout):
        """Set up widgets and data related to whether stimulation
        based on head direction to a reward well is enabled"""

        self._hdir_stim_label = QLabel(self.tr("Head direction stim"))
        layout.addWidget(self._hdir_stim_label, 12, 0)

        self._hdir_stim_on = QRadioButton(self.tr("ON"))
        self._hdir_stim_off = QRadioButton(self.tr("OFF"))
        hdir_layout = QHBoxLayout()
        hdir_layout.addWidget(self._hdir_stim_on)
        hdir_layout.addWidget(self._hdir_stim_off)
        hdir_group_box = QGroupBox()
        hdir_group_box.setLayout(hdir_layout)
        layout.addWidget(hdir_group_box, 12, 1)

        self._hdir_stim_button = QPushButton(self.tr("Update"))
        self._hdir_stim_button.pressed.connect(self._check_hdir_stim)
        layout.addWidget(self._hdir_stim_button, 12, 2)

        if self._config['stimulation']['head_direction']['enabled']:
            self._hdir_stim_on.setChecked(True)
            self._main_params.head_direction_stim_enabled = True
        else:
            self._hdir_stim_off.setChecked(True)
            self._main_params.head_direction_stim_enabled = False

    def _setup_ripple_params(self, layout):
        """Set up the ripple parameters widgets"""

        self._ripple_label = QLabel(self.tr("Ripple"))
        self._ripple_label.setStyleSheet("font-weight: bold")
        layout.addWidget(self._ripple_label, 13, 0)

        self._setup_ripple_detect_vel(layout)
        self._setup_rip_thresh(layout)
        self._setup_cond_rip_thresh(layout)
        self._setup_content_rip_thresh(layout)
        self._setup_end_rip_thresh(layout)

    def _setup_ripple_detect_vel(self, layout):
        """Set up widgets and data for the maximum velocity allowed for a
        ripple event"""

        self._ripple_vel_thresh_label = QLabel(self.tr("Ripple velocity threshold"))
        layout.addWidget(self._ripple_vel_thresh_label, 14, 0)

        self._ripple_vel_thresh_edit = QLineEdit()
        layout.addWidget(self._ripple_vel_thresh_edit, 14, 1)

        self._ripple_vel_thresh_button = QPushButton(self.tr("Update"))
        self._ripple_vel_thresh_button.pressed.connect(self._check_ripple_vel_thresh)
        layout.addWidget(self._ripple_vel_thresh_button, 14, 2)

        value = float(self._config['ripples']['vel_thresh'])
        self._ripple_vel_thresh_edit.setText(str(value))
        self._ripple_params.velocity_threshold = value

    def _setup_rip_thresh(self, layout):
        """Set up widgets and data for standard ripple threshold"""

        self._rip_thresh_label = QLabel(self.tr("Ripple threshold"))
        layout.addWidget(self._rip_thresh_label, 15, 0)

        self._rip_thresh_edit = QLineEdit()
        layout.addWidget(self._rip_thresh_edit, 15, 1)

        self._rip_thresh_button = QPushButton(self.tr("Update"))
        self._rip_thresh_button.pressed.connect(self._check_rip_thresh)
        layout.addWidget(self._rip_thresh_button, 15, 2)

        value = float(self._config['ripples']['threshold']['standard'])
        self._rip_thresh_edit.setText(str(value))
        self._ripple_params.ripple_threshold = value

    def _setup_cond_rip_thresh(self, layout):
        """Set up widgets and data for conditioning ripple threshold"""

        self._cond_rip_thresh_label = QLabel(
            self.tr("Conditioning ripple threshold")
        )
        layout.addWidget(self._cond_rip_thresh_label, 16, 0)

        self._cond_rip_thresh_edit = QLineEdit()
        layout.addWidget(self._cond_rip_thresh_edit, 16, 1)

        self._cond_rip_thresh_button = QPushButton(self.tr("Update"))
        self._cond_rip_thresh_button.pressed.connect(
            self._check_cond_rip_thresh
        )
        layout.addWidget(self._cond_rip_thresh_button, 16, 2)

        value = float(self._config['ripples']['threshold']['conditioning'])
        self._cond_rip_thresh_edit.setText(str(value))
        self._ripple_params.conditioning_ripple_threshold = value

    def _setup_content_rip_thresh(self, layout):
        """Set up widgets and data for content ripple threshold"""

        self._content_rip_thresh_label = QLabel(
            self.tr("Content ripple threshold")
        )
        layout.addWidget(self._content_rip_thresh_label, 17, 0)

        self._content_rip_thresh_edit = QLineEdit()
        layout.addWidget(self._content_rip_thresh_edit, 17, 1)

        self._content_rip_thresh_button = QPushButton(self.tr("Update"))
        self._content_rip_thresh_button.pressed.connect(
            self._check_content_rip_thresh
        )
        layout.addWidget(self._content_rip_thresh_button, 17, 2)

        value = float(self._config['ripples']['threshold']['content'])
        self._content_rip_thresh_edit.setText(str(value))
        self._ripple_params.content_ripple_threshold = value

    def _setup_end_rip_thresh(self, layout):
        """Set up widgets and data for end of ripple threshold"""

        self._end_rip_thresh_label = QLabel(
            self.tr("End of ripple threshold")
        )
        layout.addWidget(self._end_rip_thresh_label, 18, 0)

        self._end_rip_thresh_edit = QLineEdit()
        layout.addWidget(self._end_rip_thresh_edit, 18, 1)

        self._end_rip_thresh_button = QPushButton(self.tr("Update"))
        self._end_rip_thresh_button.pressed.connect(
            self._check_end_rip_thresh
        )
        layout.addWidget(self._end_rip_thresh_button, 18, 2)

        value = float(self._config['ripples']['threshold']['end'])
        self._end_rip_thresh_edit.setText(str(value))
        self._ripple_params.end_ripple_threshold = value

    def _setup_model_params(self, layout):
        """Set up widgets for encoding model parameters"""

        self._model_label = QLabel(self.tr("Encoding Model"))
        self._model_label.setStyleSheet("font-weight: bold")
        layout.addWidget(self._model_label, 19, 0)

        self._setup_encoding_model_vel(layout)

    def _setup_encoding_model_vel(self, layout):
        """Set up widgets and data for minimum velocity for data to be
        added to encoding model"""

        self._encoding_vel_thresh_label = QLabel(self.tr("Encoding velocity threshold"))
        layout.addWidget(self._encoding_vel_thresh_label, 20, 0)

        self._encoding_vel_thresh_edit = QLineEdit()
        layout.addWidget(self._encoding_vel_thresh_edit, 20, 1)

        self._encoding_vel_thresh_button = QPushButton(self.tr("Update"))
        self._encoding_vel_thresh_button.pressed.connect(self._check_encoding_vel_thresh)
        layout.addWidget(self._encoding_vel_thresh_button, 20, 2)

        value = float(self._config['encoder']['vel_thresh'])
        self._encoding_vel_thresh_edit.setText(str(value))
        self._model_params.encoding_velocity_threshold = value

    def _setup_control_tab(self):
        """Set up decoder control tab"""

        self._control_tab = QWidget()
        layout = QGridLayout(self._control_tab)
        self._tab_widget.addTab(self._control_tab, self.tr("Control"))

        # add help tool tips!
        self._setup_general_control(layout)
        self._setup_ripple_control(layout)
        self._setup_encoding_control(layout)

        # No startup/shutdown allowed until ranks have finished sending
        # records to the main process
        self._startup_button.setEnabled(False)
        self._shutdown_button.setEnabled(False)

    def _setup_general_control(self, layout):
        """Set up control buttons"""
    
        self._general_control_label = QLabel(self.tr("General"))
        self._general_control_label.setStyleSheet("font-weight: bold")
        layout.addWidget(self._general_control_label, 0, 0)

        self._setup_startup(layout)
        self._setup_shutdown(layout)

        # self.update_start_buttons(False)

    def _setup_startup(self, layout):
        """Set up startup button"""

        self._startup_button = QPushButton(self.tr("Startup"))
        layout.addWidget(self._startup_button, 1, 0)

        self._startup_button.pressed.connect(
            self._initiate_startup
        )

    def _setup_shutdown(self, layout):
        """Set up shutdown button"""

        self._shutdown_button = QPushButton(self.tr("Shutdown"))
        layout.addWidget(self._shutdown_button, 1, 1)

        self._shutdown_button.pressed.connect(
            self._initiate_shutdown
        )

    def _setup_ripple_control(self, layout):
        """Set up widgets for controlling ripple statistics updating"""

        self._ripple_control_label = QLabel(self.tr("Ripple"))
        self._ripple_control_label.setStyleSheet("font-weight: bold")
        layout.addWidget(self._ripple_control_label, 2, 0)

        self._setup_ripple_freeze(layout)

    def _setup_ripple_freeze(self, layout):
        """Set up widgets and data for controlling ripple statistics updating"""

        self._is_ripple_stats_frozen = self._config['ripples']['freeze_stats']
        if self._is_ripple_stats_frozen:
            text = "Unfreeze ripple stats"
        else:
            text = "Freeze ripple stats"

        self._ripple_freeze_button = QPushButton(self.tr(text))
        layout.addWidget(self._ripple_freeze_button, 3, 0)

        self._ripple_freeze_button.pressed.connect(
            self._check_ripple_freeze
        )

        self._ripple_params.freeze_stats = self._is_ripple_stats_frozen

    def _setup_encoding_control(self, layout):
        """Set up widgets for controlling encoding model updating"""

        self._encoding_freeze_label = QLabel(self.tr("Encoding"))
        self._encoding_freeze_label.setStyleSheet("font-weight: bold")
        layout.addWidget(self._encoding_freeze_label, 4, 0)

        self._setup_encoding_freeze(layout)

    def _setup_encoding_freeze(self, layout):
        """Set up widgets and data for controlling encoding model updating"""

        self._is_encoding_model_frozen = self._config['frozen_model']
        if self._is_encoding_model_frozen:
            text = "Unfreeze encoding model"
        else:
            text = "Freeze encoding model"

        self._encoding_freeze_button = QPushButton(self.tr(text))
        layout.addWidget(self._encoding_freeze_button, 5, 0)

        self._encoding_freeze_button.pressed.connect(
            self._check_encoding_freeze
        )

        self._model_params.freeze_model = self._is_encoding_model_frozen

    def _check_target_arm(self):
        """Input validation for updating target arm parameter"""

        target_arm = self._target_arm_edit.text()

        try:
            num_arms = len(
                self._config['encoder']['position']['arm_coords']
            )
            target_arm = int(target_arm)
            if target_arm < 0:
                target_arm = 0
            elif target_arm >= num_arms:
                target_arm = num_arms - 1

            self._main_params.replay_target_arm = target_arm
            self._send_main_params()
            _show_message(
                self,
                f"Message sent - Replay target arm: {target_arm}"
            )
            self._target_arm_edit.setText(str(target_arm))

        except Exception as e:
            _show_message(
                self, "An unexpected exception occurred! " + e.args[0],
                kind='critical'
            )

    def _check_post_thresh(self):
        """Input validation for posterior threshold parameter"""

        post_thresh = self._post_edit.text()

        try:
            post_thresh = float(post_thresh)
            if post_thresh < 0:
                post_thresh = 0
            elif post_thresh >= 1:
                post_thresh = 1

            self._main_params.posterior_threshold = post_thresh
            self._send_main_params()
            _show_message(
                self,
                f"Message sent - Posterior threshold value: {post_thresh}",
                kind="information")
            self._post_edit.setText(str(post_thresh))

        except Exception as e:
            _show_message(
                self, "An unexpected exception occurred! " + e.args[0],
                kind='critical'
            )

    def _check_max_center_well(self):
        """Input validation for max center well distance parameter"""

        dist = self._max_center_well_edit.text()

        try:
            dist = float(dist)
            # any other restrictions?
            if dist < 0:
                dist = 0

            self._main_params.max_center_well_distance = dist
            self._send_main_params()
            # need to check in main process whether distance is converted
            # to cm
            _show_message(
                self,
                f"Message sent - Max center well distance (cm) value: {dist}",
                kind="information")
            self._max_center_well_edit.setText(str(dist))

        except Exception as e:
            _show_message(
                self, "An unexpected exception occurred! " + e.args[0],
                kind='critical'
            )

    def _check_num_above(self):
        """Input validation for number of electrodes above ripple threshold
        parameter"""

        max_n_above = len(self._config["trode_selection"]["ripples"])
        n_above = self._num_above_edit.text()

        try:
            n_above = float(n_above)
            _, rem = divmod(n_above, 1)
            show_error = False
            if rem != 0:
                show_error = True
            if n_above not in list(range(1, max_n_above + 1)):
                show_error = True

            if show_error:
                _show_message(
                    self,
                    "Number of tetrodes above threshold must be an INTEGER value "
                    f"(i.e. not a float) between 1 and {max_n_above}, inclusive",
                    kind="critical")
            else:
                n_above = int(n_above)
                self._main_params.num_above_threshold = n_above
                self._send_main_params()
                _show_message(
                    self,
                    f"Message sent - n above threshold value: {n_above}",
                    kind="information")
                self._num_above_edit.setText(str(n_above))

        except Exception as e:
            _show_message(
                self, "An unexpected exception occurred! " + e.args[0],
                kind='critical'
            )

    def _check_min_duration(self):
        """Input validation for minimum duration of head direction
        event parameter"""

        min_duration = self._min_duration_edit.text()

        try:
            min_duration = float(min_duration)
            # any other restrictions?
            if min_duration < 0:
                min_duration = 0

            self._main_params.min_duration = min_duration
            self._send_main_params()
            _show_message(
                self,
                f"Message sent - Min head direction duration value: {min_duration}",
                kind="information"
            )
            self._min_duration_edit.setText(str(min_duration))

        except Exception as e:
            _show_message(
                self, "An unexpected exception occurred! " + e.args[0],
                kind='critical'
            )

    def _check_well_angle_range(self):
        """Input validation for well angle range parameter"""

        well_angle_range = self._well_angle_range_edit.text()

        try:
            well_angle_range = float(well_angle_range)
            # any other restrictions?
            if well_angle_range < 0:
                well_angle_range = 0
            elif well_angle_range > 360:
                well_angle_range = 360

            self._main_params.well_angle_range = well_angle_range
            self._send_main_params()
            _show_message(
                self,
                f"Message sent - Well angle range value: {well_angle_range}",
                kind="information"
            )
            self._well_angle_range_edit.setText(str(well_angle_range))

        except Exception as e:
            _show_message(
                self, "An unexpected exception occurred! " + e.args[0],
                kind='critical'
            )

    def _check_within_angle_range(self):
        """Input validation for within-angle angle range parameter"""

        within_angle_range = self._within_angle_range_edit.text()

        try:
            within_angle_range = float(within_angle_range)
            # any other restrictions?
            if within_angle_range < 0:
                within_angle_range = 0
            elif within_angle_range > 360:
                within_angle_range = 360

            self._main_params.within_angle_range = within_angle_range
            self._send_main_params()
            _show_message(
                self,
                f"Message sent - Within angle range value: {within_angle_range}",
                kind="information"
            )
            self._within_angle_range_edit.setText(str(within_angle_range))

        except Exception as e:
            _show_message(
                self, "An unexpected exception occurred! " + e.args[0],
                kind='critical'
            )

    def _check_rotate_180(self):
        """Check whether the head direction vector is requested
        to be rotated 180 degrees and send corresponding message"""

        rotate_180 = self._rotate_180_yes.isChecked()
        try:
            if rotate_180:
                self._main_params.rotate_180 = True
                self._send_main_params()
                _show_message(
                    self,
                    "Message sent - Rotate 180 ON",
                    kind="information")
            else:
                self._main_params.rotate_180 = False
                self._send_main_params()
                _show_message(
                    self,
                    "Message sent - Rotate 180 OFF",
                    kind="information")
        except Exception as e:
            _show_message(
                self, "An unexpected exception occurred! " + e.args[0],
                kind='critical'
            )

    def _check_replay_stim(self):
        """Check whether stimulation based on replay events is
        requested to be enabled and send corresponding message"""

        replay_stim_on = self._replay_stim_on.isChecked()
        try:
            if replay_stim_on:
                self._main_params.replay_stim_enabled = True
                self._send_main_params()
                _show_message(
                    self,
                    "Message sent - Replay stim ON",
                    kind='information'
                )
            else:
                self._main_params.replay_stim_enabled = False
                self._send_main_params()
                _show_message(
                    self,
                    "Message sent - Replay stim OFF",
                    kind='information'
                )
        except Exception as e:
            _show_message(
                self, "An unexpected exception occurred! " + e.args[0],
                kind='critical'
            )

    def _check_ripple_stim(self):
        """Check whether stimulation based on ripple events
        requested to be enabled and send corresponding message"""

        ripple_stim_on = self._ripple_stim_on.isChecked()
        try:
            if ripple_stim_on:
                self._main_params.ripple_stim_enabled = True
                self._send_main_params()
                _show_message(
                    self,
                    "Message sent - Ripple stim ON",
                    kind='information'
                )
            else:
                self._main_params.ripple_stim_enabled = False
                self._send_main_params()
                _show_message(
                    self,
                    "Message sent - Ripple stim OFF",
                    kind='information'
                )
        except Exception as e:
            _show_message(
                self, "An unexpected exception occurred! " + e.args[0],
                kind='critical'
            )

    def _check_hdir_stim(self):
        """Check whether stimulation based on head direction events
        is requested to be enabled and send corresponding message"""

        hdir_stim_on = self._hdir_stim_on.isChecked()
        try:
            if hdir_stim_on:
                self._main_params.head_direction_stim_enabled = True
                self._send_main_params()
                _show_message(
                    self,
                    "Message sent - Head direction stim ON",
                    kind='information'
                )
            else:
                self._main_params.head_direction_stim_enabled = False
                self._send_main_params()
                _show_message(
                    self,
                    "Message sent - Head direction stim ON",
                    kind='information'
                )
        except Exception as e:
            _show_message(
                self, "An unexpected exception occurred! " + e.args[0],
                kind='critical'
            )

    def _check_ripple_vel_thresh(self):
        """Input validation for ripple detection velocity threshold parameter"""

        ripple_vel_thresh = self._ripple_vel_thresh_edit.text()

        try:
            ripple_vel_thresh = float(ripple_vel_thresh)
            if ripple_vel_thresh < 0:
                ripple_vel_thresh = 0

            self._ripple_params.velocity_threshold = ripple_vel_thresh
            self._send_ripple_params()
            _show_message(
                self,
                f"Message sent - Ripple velocity threshold value: {ripple_vel_thresh}",
                kind="information")
            self._ripple_vel_thresh_edit.setText(str(ripple_vel_thresh))

        except Exception as e:
            _show_message(
                self, "An unexpected exception occurred! " + e.args[0],
                kind='critical'
            )

    def _check_rip_thresh(self):
        """Input validation for standard ripple threshold parameter"""

        rip_thresh = self._rip_thresh_edit.text()
        try:
            rip_thresh = float(rip_thresh)
            # any other restrictions?
            if rip_thresh < 0:
                rip_thresh = 0

            self._ripple_params.ripple_threshold = rip_thresh
            self._send_ripple_params()
            _show_message(
                self,
                f"Message sent - Ripple threshold value: {rip_thresh}",
                kind="information")
            self._ripple_vel_thresh_edit.setText(str(rip_thresh))
        except Exception as e:
            _show_message(
                self, "An unexpected exception occurred! " + e.args[0],
                kind='critical'
            )

    def _check_cond_rip_thresh(self):
        """Input validation for conditioning ripple threshold parameter"""

        cond_rip_thresh = self._cond_rip_thresh_edit.text()
        try:
            cond_rip_thresh = float(cond_rip_thresh)
            # any other restrictions?
            if cond_rip_thresh < 0:
                cond_rip_thresh = 0

            self._ripple_params.conditioning_ripple_threshold = cond_rip_thresh
            self._send_ripple_params()
            _show_message(
                self,
                f"Message sent - Conditioning ripple threshold value: {cond_rip_thresh}",
                kind="information")
            self._cond_rip_thresh_edit.setText(str(cond_rip_thresh))
        except Exception as e:
            _show_message(
                self, "An unexpected exception occurred! " + e.args[0],
                kind='critical'
            )

    def _check_content_rip_thresh(self):
        """Input validation for content ripple threshold parameter"""

        content_rip_thresh = self._content_rip_thresh_edit.text()
        try:
            content_rip_thresh = float(content_rip_thresh)
            # any other restrictions?
            if content_rip_thresh < 0:
                content_rip_thresh = 0

            self._ripple_params.content_ripple_threshold = content_rip_thresh
            self._send_ripple_params()
            _show_message(
                self,
                "Message sent - Conditioning ripple threshold value: "
                f"{content_rip_thresh}", kind="information")
            self._content_rip_thresh_edit.setText(str(content_rip_thresh))
        except Exception as e:
            _show_message(
                self, "An unexpected exception occurred! " + e.args[0],
                kind='critical'
            )

    def _check_end_rip_thresh(self):
        """Input validation for end of ripple threshold parameter"""

        end_rip_thresh = self._end_rip_thresh_edit.text()
        try:
            end_rip_thresh = float(end_rip_thresh)
            # any other restrictions?
            if end_rip_thresh < 0:
                end_rip_thresh = 0

            self._ripple_params.end_ripple_threshold = end_rip_thresh
            self._send_ripple_params()
            _show_message(
                self,
                "Message sent - End of ripple threshold value: "
                f"{end_rip_thresh}", kind="information")
            self._end_rip_thresh_edit.setText(str(end_rip_thresh))
        except Exception as e:
            _show_message(
                self, "An unexpected exception occurred! " + e.args[0],
                kind='critical'
            )

    def _check_encoding_vel_thresh(self):
        """Input validation for encoding velocity threshold parameter"""

        encoding_vel_thresh = self._encoding_vel_thresh_edit.text()
        try:
            encoding_vel_thresh = float(encoding_vel_thresh)
            # any other restrictions?
            if encoding_vel_thresh < 0:
                encoding_vel_thresh = 0

            self._model_params.encoding_velocity_threshold = encoding_vel_thresh
            self._send_encoding_model_params()
            _show_message(
                self,
                f"Message sent - Encoding velocity threshold value: {encoding_vel_thresh}",
                kind="information")
            self._encoding_vel_thresh_edit.setText(str(encoding_vel_thresh))
        except Exception as e:
            _show_message(
                self, "An unexpected exception occurred! " + e.args[0],
                kind='critical'
            )

    def _initiate_startup(self):
        """Initiate startup sequence"""

        # send startup signal
        self._send_interface.send_startup()
        self._startup_button.setEnabled(False)

    def _initiate_shutdown(self):
        """Initiate shutdown sequence"""

        # send terminate signal
        self._send_interface.send_shutdown()
        self._shutdown_button.setEnabled(False)

    def _check_ripple_freeze(self):
        """Determine what to do when the freeze ripple states button
        is toggled"""

        if self._is_ripple_stats_frozen:
            self._is_ripple_stats_frozen = False
            self._ripple_freeze_button.setText(self.tr("Freeze ripple stats"))
        else:
            self._is_ripple_stats_frozen = True
            self._ripple_freeze_button.setText("Unfreeze ripple stats")

        self._ripple_params.freeze_stats = self._is_ripple_stats_frozen
        self._send_ripple_params()

    def _check_encoding_freeze(self):
        """Determine what to do when the freeze encoding model button
        is toggled"""

        if self._is_encoding_model_frozen:
            self._is_encoding_model_frozen = False
            self._encoding_freeze_button.setText(self.tr("Freeze encoding stats"))
        else:
            self._is_encoding_model_frozen = True
            self._encoding_freeze_button.setText(self.tr("Unfreeze encoding stats"))

        self._model_params.freeze_model = self._is_encoding_model_frozen
        self._send_encoding_model_params()

    def _send_main_params(self):
        """Send parameters to be received in main_process"""

        print(self._main_params)
        self._send_interface.send_main_params(
            self._main_params
        )

    def _send_ripple_params(self):
        """Send ripple detection parameters"""

        print(self._ripple_params)
        self._send_interface.send_ripple_params(
            self._ripple_params
        )

    def _send_encoding_model_params(self):
        """Send encoding model parameters"""

        print(self._model_params)
        self._send_interface.send_encoding_model_params(
            self._model_params
        )

    def _send_all_params(self):
        """Send all parameters"""

        self._send_main_params()
        self._send_ripple_params()
        self._send_encoding_model_params()

    def enable_general_control(self):
        """Allow user to start or shutdown the system"""

        self._startup_button.setEnabled(True)
        self._shutdown_button.setEnabled(True)

    def update_start_buttons(self):
        """Update the decoder control buttons"""

        self._startup_button.setEnabled(False)
        self._shutdown_button.setEnabled(True)

    def run(self):
        """Run the processing for the dialog window"""

        if self._timer is not None:
            self._timer.start()

    def closeEvent(self, event):
        """Method called when user attempts to close the
        dialog window"""

        _show_message(
            self,
            "Processes not finished running. Closing GUI dialog is disabled",
            kind="critical")
        event.ignore()

class DecodingResultsWindow(QMainWindow):
    """Window for visualizing decoding plots"""

    def __init__(self, comm, rank, config):
        super().__init__()

        self._rank = rank
        self._config = config

        self._send_interface = None
        self._command_interface = None
        self._posterior_interface = None
        self._arm_events_interface = None
        self._rewards_interface = None
        self._dropped_spikes_interface = None
        self._setup_interfaces(comm, rank, config)
        self._is_setup_complete = False

        self._class_log = logging.getLogger(
            name=f'{self.__class__.__name__}'
        )

        self.setWindowTitle("Decoder Output")
        self._graphics_widget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self._graphics_widget)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Status: NOT READY")

        # Status bar data
        self._sbdata = {}
        self._sbdata['arm_events'] = [0] * len(
            self._config['encoder']['position']['arm_coords']
        )
        self._sbdata['dropped_spikes'] = [0] * len(
            self._config['rank']['decoders']
        )
        self._sbdata['rewards_delivered'] = 0

        self._init_plots()
        self._init_colormap()

        self._dialog = TabbedDialog(self, comm, rank, config)
        self._dialog.move(
            self.pos().x() + self.frameGeometry().width() + 100, self.pos().y())

        self._timer = QTimer()
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._update)

        self._elapsed_timer = QElapsedTimer()
        self._refresh_msec = int(
            np.round(1/self._config['gui']['refresh_rate']*1000)
        )

        self._ok_to_terminate = False

    def _setup_interfaces(self, comm, rank, config):
        """Set up interfaces used to receive MPI messages"""

        self._send_interface = base.StandardMPISendInterface(
            comm, rank, config
        )
        self._command_interface = base.StandardMPIRecvInterface(
            comm, rank, config,
            messages.MPIMessageTag.COMMAND_MESSAGE, self
        )

        self._posterior_interface = GenericGuiRecvInterface(
            comm, rank, config,
            messages.get_dtype("Posterior", config=config),
            messages.MPIMessageTag.POSTERIOR, self
        )

        self._dropped_spikes_interface = GenericGuiRecvInterface(
            comm, rank, config,
            messages.get_dtype("DroppedSpikes", config=config),
            messages.MPIMessageTag.DROPPED_SPIKES, self
        )

        self._arm_events_interface = ArmEventsRecvInterface(
            comm, rank, config, self
        )

    def _init_plots(self):
        """Initialize plots"""

        self._decoder_rank_ind_map = OrderedDict()
        for ii, rank in enumerate(self._config["rank"]["decoders"]):
            self._decoder_rank_ind_map[rank] = ii

        dt = (
            self._config['decoder']['time_bin']['samples'] /
            self._config['sampling_rate']['spikes']
        )
        B = self._config['encoder']['position']['num_bins']
        N = int(
            np.round(
                self._config['gui']['trace_length'] / dt,
                decimals=0
            )
        )
        self._num_time_bins = N

        num_plots = len(self._config['rank']['decoders'])

        algorithm = self._config['algorithm']
        state_labels = self._config[algorithm]['state_labels']
        S = len(state_labels)
        self._num_states = S
        state_colors = self._config['gui']['state_colors']

        # plot handles
        self._plots = {}
        self._plots['lk'] = [None] * num_plots
        self._plots['post'] = [None] * num_plots
        self._plots['state'] = [None] * num_plots

        # plot items
        self._plot_items = {}
        self._plot_items['lk'] = {}
        self._plot_items['lk']['image'] = [None] * num_plots
        self._plot_items['post'] = {}
        self._plot_items['post']['image'] = [None] * num_plots
        self._plot_items['state'] = {}
        self._plot_items['state']['data'] = [ [] for _ in range(num_plots)]

        # data
        self._data = {}
        self._data['lk'] = [np.zeros((B, N)) for _ in range(num_plots)]
        self._data['post'] = [np.zeros((B, N)) for _ in range(num_plots)]
        self._data['state'] = [np.zeros((S, N)) for _ in range(num_plots)]
        self._data['ind'] = [0] * num_plots

        # used for likelihood/posterior plots
        bin_edges = np.linspace(
            self._config['encoder']['position']['lower'],
            self._config['encoder']['position']['upper'],
            self._config['encoder']['position']['num_bins'] + 1
        )
        arm_coords = self._config['encoder']['position']['arm_coords']

        self._setup_lk_plots(num_plots, bin_edges, arm_coords)
        self._setup_posterior_plots(num_plots, bin_edges, arm_coords)
        self._setup_state_prob_plots(num_plots, state_labels, state_colors)

        num_xticks = self._config['gui']['num_xticks']
        self._set_plot_ticks(num_plots, dt, N, num_xticks)

    def _init_colormap(self):
        """Initialize colormap for the plots"""

        try:
            cmap = sns.color_palette(
                self._config['gui']['colormap'], as_cmap=True)
        except:
            _show_message(
                self,
                f"Colormap {colormap} could not be found, using default",
                kind="information")
            cmap = sns.color_palette(
                _DEFAULT_GUI_PARAMS["colormap"], as_cmap=True)

        cmap._init()
        lut = (cmap._lut * 255).view(np.ndarray)

        for ii, (lk_image, post_image) in enumerate(
            zip(
                self._plot_items['lk']['image'],
                self._plot_items['post']['image']
            )
        ):
            lk_image.setLookupTable(lut)
            lk_image.setImage(self._data['lk'][ii].T)

            post_image.setLookupTable(lut)
            post_image.setImage(self._data['post'][ii].T)

    def _set_plot_ticks(self, num_plots, dt, num_time_bins, num_xticks):
        """Set the plot ticks at regularly spaced intervals"""

        for ii in range(num_plots):

            ticks = np.linspace(0, num_time_bins, num_xticks)
            tick_labels = [str(np.round(tick*dt, decimals=2)) for tick in ticks]
            tick_data = [[(tick, tick_label) for (tick, tick_label) in zip(ticks, tick_labels)]]

            self._plots['lk'][ii].getAxis("bottom").setTicks(tick_data)
            self._plots['post'][ii].getAxis("bottom").setTicks(tick_data)
            self._plots['state'][ii].getAxis("bottom").setTicks(tick_data)

    def _setup_lk_plots(self, num_plots, bin_edges, arm_coords):
        """Set up the plots visualizing likelihoods"""

        for ii in range(num_plots):

            dec_rank = self._config['rank']['decoders'][ii]

            # create plots/plot properties
            self._plots['lk'][ii] = self._graphics_widget.addPlot(
                0, ii, 1, 1,
                title=f'Likelihood (Rank {dec_rank})',
                labels={
                    'left': 'Position bin',
                    'bottom': 'Time (sec)',
                }
            )
            self._plots['lk'][ii].setMenuEnabled(False)

            # add plot items
            for lower_bin, upper_bin in arm_coords:
                lb = bin_edges[lower_bin]
                ub = bin_edges[upper_bin + 1]
                # Note: Not storing horizontal line data
                self._plots['lk'][ii].addItem(
                    pg.PlotDataItem(
                        np.ones(self._num_time_bins) * lb, pen='w', width=10
                    )
                )
                self._plots['lk'][ii].addItem(
                    pg.PlotDataItem(
                        np.ones(self._num_time_bins) * ub, pen='w', width=10
                    )
                )

                self._plot_items['lk']['image'][ii] = pg.ImageItem(border=None)
                self._plot_items['lk']['image'][ii].setZValue(-100)

                self._plots['lk'][ii].addItem(
                    self._plot_items['lk']['image'][ii]
                )

    def _setup_posterior_plots(self, num_plots, bin_edges, arm_coords):
        """Set up the plots visualizing the posterior"""

        for ii in range(num_plots):

            dec_rank = self._config['rank']['decoders'][ii]

            # create plot/plot properties
            self._plots['post'][ii] = self._graphics_widget.addPlot(
                1, ii, 1, 1,
                title=f'Marginalized Posterior (Rank {dec_rank})',
                labels={
                    'left': 'Position bin',
                    'bottom': 'Time (sec)',
                }
            )
            self._plots['post'][ii].setMenuEnabled(False)

            # add plot items
            for lower_bin, upper_bin in arm_coords:
                lb = bin_edges[lower_bin]
                ub = bin_edges[upper_bin + 1]
                # Note: Not storing horizontal line data
                self._plots['post'][ii].addItem(
                    pg.PlotDataItem(
                        np.ones(self._num_time_bins) * lb, pen='w', width=10
                    )
                )
                self._plots['post'][ii].addItem(
                    pg.PlotDataItem(
                        np.ones(self._num_time_bins) * ub, pen='w', width=10
                    )
                )

                self._plot_items['post']['image'][ii] = pg.ImageItem(border=None)
                self._plot_items['post']['image'][ii].setZValue(-100)

                self._plots['post'][ii].addItem(
                    self._plot_items['post']['image'][ii]
                )

    # TODO(DS): Change this to a consensus ripple power 
    def _setup_state_prob_plots(self, num_plots, labels, colors):
        """Set up the plots visualizing state probabilities"""

        for ii in range(num_plots):

            dec_rank = self._config['rank']['decoders'][ii]

            # create plots/plot properties
            self._plots['state'][ii] = self._graphics_widget.addPlot(
                2, ii, 1, 1,
                title=f'State Probability (Rank {dec_rank})',
                labels={
                    'left': 'Probability',
                    'bottom': 'Time (sec)',
                }
            )
            self._plots['state'][ii].addLegend(offset=None)
            self._plots['state'][ii].setRange(yRange=[0, 2])
            self._plots['state'][ii].setMenuEnabled(False)

            # add plot items
            for label, color in zip(labels, colors):
                self._plot_items['state']['data'][ii].append(
                    pg.PlotDataItem(
                        np.zeros(self._num_time_bins), pen=color,
                        width=10, name=label
                    )
                )
                self._plots['state'][ii].addItem(
                    self._plot_items['state']['data'][ii][-1]
                )

    def _update(self):
        """Receive any new messages and potentially update plot data"""

        self._command_interface.receive()
        self._posterior_interface.receive()
        self._arm_events_interface.receive()
        self._dropped_spikes_interface.receive()

        if self._elapsed_timer.elapsed() > self._refresh_msec:
            self._elapsed_timer.start()
            self._update_display_data()

    #NOTE(DS): This updates the display data
    def _update_display_data(self):
        """Update plot data"""

        # set plot data
        for ii in range(len(self._plots['lk'])):
            lk = self._data['lk'][ii]
            lk[np.isnan(lk)] = 0
            self._plot_items['lk']['image'][ii].setImage(
                lk.T * 255, levels=[0, 255]
            )

            post = self._data['post'][ii]
            post[np.isnan(post)] = 0
            self._plot_items['post']['image'][ii].setImage(
                post.T * 255, levels=[0, 255]
            )

            for state_ind in range(self._num_states):
                state_data = self._data['state'][ii][state_ind]
                self._plot_items['state']['data'][ii][state_ind].setData(
                    state_data
                )

    def handle_message(self, msg, mpi_status):
        """Process a (non neural data) received MPI message"""

        if isinstance(msg, messages.SetupComplete):
            if not self._is_setup_complete:
                _show_message(
                    self,
                    "All processes have finished setup. After closing this popup, "
                    "hit record or play, or use the GUI to start decoding.",
                    kind="information"
                )
                self.statusBar().showMessage("Status: READY")
                self._is_setup_complete = True
                self._dialog.enable_general_control()
        elif isinstance(msg, messages.DecoderStarted):
            self.statusBar().showMessage("Status: Decoder Started")
            # gray out dialog startup, gray in dialog shutdown
            self._dialog.update_start_buttons()
        elif isinstance(msg, messages.VerifyStillAlive):
            # raise ValueError()
            self._send_interface.send_alive_message()
        elif isinstance(msg, messages.TerminateSignal):
            self._ok_to_terminate = True
            _show_message(
                self,
                "Processes have terminated, closing GUI",
                kind="information"
            )
            self.close()
        elif mpi_status.tag == messages.MPIMessageTag.POSTERIOR:
            self._update_lk_post_data(msg, mpi_status)
        elif mpi_status.tag == messages.MPIMessageTag.DROPPED_SPIKES:
            self._update_dropped_spikes(msg, mpi_status)
        elif mpi_status.tag == messages.MPIMessageTag.ARM_EVENTS:
            self._update_arm_events(msg, mpi_status)
        else:
            self._class_log.warning(
                f"Received message of type {type(msg)} "
                f"from source: {mpi_status.source}, "
                f" tag: {mpi_status.tag}, ignoring"
            )

    def _update_lk_post_data(self, msg, mpi_status):
        """Update plot data for the likelihood and posterior plots"""

        sender = mpi_status.source
        plot_ind = self._decoder_rank_ind_map[mpi_status.source]
        assert sender == msg[0]['rank']
        
        ind = self._data['ind'][plot_ind]

        # update data
        self._data['lk'][plot_ind][:, ind] = (
            msg[0]['likelihood'] / np.nansum(msg[0]['likelihood'])
        )

        self._data['post'][plot_ind][:, ind] = np.nansum(
            msg[0]['posterior'], axis=0
        )

        self._data['state'][plot_ind][:, ind] = np.nansum(
            msg[0]['posterior'], axis=1
        )

        # update index for next data point to be stored at
        self._data['ind'][plot_ind] = (
            (self._data['ind'][plot_ind] + 1) %
            self._num_time_bins
        )

    def _update_dropped_spikes(self, msg, mpi_status):
        """Update data keeping track of dropped spikes"""

        sender = mpi_status.source
        assert sender == msg[0]['rank']

        ind = self._decoder_rank_ind_map[sender]
        self._sbdata['dropped_spikes'][ind] = msg[0]['pct']

        self._update_status_bar()

    def _update_arm_events(self, msg, mpi_status):
        """Update data keeping track of events detected
        at different arms"""

        for ii, num_events in enumerate(msg):
            self._sbdata['arm_events'][ii] = num_events
        self._sbdata['rewards_delivered'] = np.sum(msg)
        self._update_status_bar()

    def _update_status_bar(self):
        """Update the status bar"""

        sb_string = ""
        for ii, num_events in enumerate(self._sbdata['arm_events']):
            if ii > 0: # skip events belonging to box
                sb_string += f"Arm {ii}: {num_events}, "

        sb_string += "Dropped Spikes: "
        for rank, pct in zip(
            self._decoder_rank_ind_map.keys(),
            self._sbdata['dropped_spikes']
        ):
            sb_string += f"(Rank {rank}: {pct:.3f}%), "

        sb_string += f"Tot. Rewards: {self._sbdata['rewards_delivered']}"

        self.statusBar().showMessage(sb_string)

    def show_all(self):
        """Show this window and the dialog window"""

        self.show()
        self._dialog.show()

    def run(self):
        """Run the processing loop"""

        if self._config['preloaded_model']:
            _show_message(
                self, "Using preloaded encoding model",
                kind='information'
            )
        self._elapsed_timer.start()
        self._timer.start()
        self._dialog.run()

    def closeEvent(self, event):
        """Method called when user attempts to close the window"""

        if not self._ok_to_terminate:
            _show_message(
                self,
                "Processes not finished running. Closing GUI window is disabled",
                kind="critical")
            event.ignore()
        else:
            super().closeEvent(event)

####################################################################################
# Processes
####################################################################################

class GuiProcess(base.RealtimeProcess):
    """Top level object for gui_process"""

    def __init__(self, comm, rank, config):
        super().__init__(comm, rank, config)
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        self._app = app
        self._main_window = DecodingResultsWindow(comm, rank, config)

    def main_loop(self):
        """Main GUI processing loop"""

        self._main_window.show_all()
        self._main_window.run()
        self._app.exec()
        self.class_log.info("GUI process exiting main loop")
