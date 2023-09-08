import time
import numpy as np

from copy import deepcopy

from realtime_decoder import base, utils, messages, binary_record, taskstate


"""Contains objects relevant to detecting if stimulation
should be given"""

class StimDeciderSendInterface(base.MPISendInterface):

    """The interface object a stim decider uses to communicate with
    other processes
    """

    def __init__(self, comm, rank, config):
        super().__init__(comm, rank, config)

    def send_num_rewards(self, num_rewards_arr):
        """Sends the GUI information about the number of rewards
        dispensed for each maze arm"""

        self.comm.Send(
            buf=num_rewards_arr,
            dest=self.config['rank']['gui'][0],
            tag=messages.MPIMessageTag.ARM_EVENTS
        )

    def send_record_register_messages(self):
        """Raises error as an override of the base object's
        'send_record_register_messages()' since this inherited
        object does not send record registration messages to the
        main process"""

        raise NotImplementedError(
            f"This class does not send record registration messages "
            "to the main process"
        )

class TwoArmTrodesStimDecider(base.BinaryRecordBase, base.MessageHandler):
    """Custom stimulation decider for a two-arm maze where Trodes is the
    data acquisition software"""

    def __init__(self, comm, rank, config, trodes_client):

        num_decoders = len(config['rank']['decoders'])
        if num_decoders > 2:
            raise NotImplementedError(
                "This object is not designed to handle more than "
                "two decoding ranks"
            )
        self._num_decoders = num_decoders

        num_regions = len(config['stimulation']['head_direction']['well_loc'])
        if num_regions != 2:
            raise NotImplementedError(
                "This object only handles a two-arm maze"
            )

        burst_labels = [f'burst_{x}' for x in range(num_decoders)]
        spike_count_labels = [f'spike_count_{x}' for x in range(num_decoders)]
        event_spike_count_labels = [f'event_spike_count_{x}' for x in range(num_decoders)]
        avg_spike_rate_labels = [f'avg_spike_rate_{x}' for x in range(num_decoders)]
        credible_int_labels = [f'credible_int_{x}' for x in range(num_decoders)]

        rls = ['region_box', 'region_arm1', 'region_arm2']
        region_labels = [f'{rl}_{x}' for x in range(num_decoders) for rl in rls]

        bls = ['base_box', 'base_arm1', 'base_arm2']
        base_labels = [f'{bl}_{x}' for x in range(num_decoders) for bl in bls]

        armls = ['box', 'arm1', 'arm2']
        arm_labels = [f'{arml}_{x}' for x in range(num_decoders) for arml in armls]

        rtrodes = config['trode_selection']['ripples']
        ripple_ts_labels = [f'ts_{t}_{rtrode}'
            for rtrode in rtrodes
                for t in ('start', 'end')
        ]
        is_active_labels = [f'is_active_{rtrode}' for rtrode in rtrodes]

        super().__init__(
            rank=rank,
            rec_ids=[
                binary_record.RecordIDs.STIM_MESSAGE,
                binary_record.RecordIDs.STIM_HEAD_DIRECTION,
                binary_record.RecordIDs.STIM_RIPPLE_DETECTED,
                binary_record.RecordIDs.STIM_RIPPLE_END,
                binary_record.RecordIDs.STIM_RIPPLE_EVENT
            ],
            rec_labels=[
                ['bin_timestamp_l', 'bin_timestamp_r', 'shortcut_message_sent',
                 'delay', 'velocity', 'mapped_pos', 'task_state',
                 'posterior_max_arm', 'target_arm', 'content_threshold',
                 'max_arm_repeats', 'replay_window_time', 'is_instructive',
                 'unique_trodes', 'center_well_dist', 'max_center_well_dist',
                 'standard_ripple', 'cond_ripple', 'content_ripple',
                 'standard_ripple_consensus', 'cond_ripple_consensus',
                 'content_ripple_consensus'] +
                 burst_labels + spike_count_labels + event_spike_count_labels +
                 avg_spike_rate_labels + credible_int_labels + region_labels +
                 base_labels + arm_labels,
                ['timestamp', 'shortcut_message_sent', 'well', 'raw_x', 'raw_y',
                 'raw_x2', 'raw_y2', 'angle', 'angle_well_1', 'angle_well_2',
                 'well_angle_range', 'within_angle_range', 'center_well_dist',
                 'max_center_well_dist', 'duration', 'rotated_180'],
                ['timestamp', 'ripple_type', 'is_consensus', 'trigger_trode',
                 'num_above_thresh', 'shortcut_message_sent'],
                ['timestamp', 'ripple_type', 'is_consensus', 'trigger_trode',
                 'num_above_thresh'],
                ['timestamp_start', 'timestamp_end', 'trigger_trode_start',
                 'trigger_trode_end', 'ripple_type', 'is_consensus',
                 'num_above_thresh', 'shortcut_message_sent'] +
                 ripple_ts_labels + is_active_labels
            ],
            rec_formats=[
                'qq?dddiiidid?qdd??????' +
                '?'*len(burst_labels) +
                'q'*len(spike_count_labels) +
                'q'*len(event_spike_count_labels) +
                'd'*len(avg_spike_rate_labels) +
                'd'*len(credible_int_labels) +
                'd'*len(region_labels) +
                'd'*len(base_labels) +
                'd'*len(arm_labels),
                'q?idddddddddddd?',
                'q10s?ii?',
                'q10s?ii',
                'qqii10s?i?' + 'q'*len(ripple_ts_labels) + '?'*len(is_active_labels)
            ],
            send_interface=StimDeciderSendInterface(comm, rank, config),
            manager_label='state'
        )

        self._config = config
        self._trodes_client = trodes_client

        self._task_state = 1
        self._task_state_handler = taskstate.TaskStateHandler(self._config)
        self._num_rewards = np.zeros(
            len(self._config['encoder']['position']['arm_coords']),
            dtype='=i4'
        )
        self._instr_rewarded_arms = np.zeros(
            self._config['stimulation']['replay']['instr_max_repeats'],
            dtype='=i4'
        )

        self._decoder_count = [0] * num_decoders

        self._pos_msg_ct = 0
        self._current_pos = 0
        self._current_vel = 0

        self._center_well_dist = 0
        self._is_center_well_proximate = False

        # data variables we record to disk
        self._delay = (
            self._config['decoder']['time_bin']['delay_samples'] /
            self._config['sampling_rate']['spikes']
        )
        self._replay_window_time = (
            self._config['decoder']['time_bin']['samples'] /
            self._config['sampling_rate']['spikes'] *
            self._config['stimulation']['replay']['sliding_window']
        )
        if self._config['stimulation']['instructive']:
            self._max_repeats = self.p_replay['instr_max_repeats']
        else:
            self._max_repeats = -1 # currently no restriction

        ripple_types = ('standard', 'cond', 'content')
        self._init_ripple_misc(rtrodes, ripple_types)
        self._init_head_dir()
        self._init_data_buffers()
        self._seed_mua_stats()
        self._init_stim_params()
        self._init_params()

    def handle_message(self, msg, mpi_status):
        """Process a (non neural data) received MPI message"""

        # feedback, velocity/position, posterior
        if isinstance(msg, messages.GuiMainParameters):
            self._update_gui_params(msg)
        elif mpi_status.tag == messages.MPIMessageTag.RIPPLE_DETECTION:
            self._update_ripples(msg)
        elif mpi_status.tag == messages.MPIMessageTag.VEL_POS:
            self._update_velocity_position(msg)
        elif mpi_status.tag == messages.MPIMessageTag.POSTERIOR:
            self._update_posterior(msg)
        else:
            self._class_log.warning(
                f"Received message of type {type(msg)} "
                f"from source: {mpi_status.source}, "
                f" tag: {mpi_status.tag}, ignoring"
            )

    def _update_gui_params(self, gui_msg):
        """Update parameters that can be changed by the GUI"""

        self.class_log.info("Updating GUI main parameters")

        # manual control of the replay target arm is only allowed
        # for a non-instructive task
        if not self._config['stimulation']['instructive']:
            arm = gui_msg.replay_target_arm
            self.p_replay['target_arm'] = arm
            self.class_log.info(
                "Non instructive task: Updated replay target arm to {arm}"
            )

        self.p_replay['primary_arm_threshold'] = gui_msg.posterior_threshold
        self.p['max_center_well_dist'] = gui_msg.max_center_well_distance
        self.p_ripples['num_above_thresh'] = gui_msg.num_above_threshold
        self.p_head['min_duration'] = gui_msg.min_duration
        self.p_head['well_angle_range'] = gui_msg.well_angle_range
        self.p_head['within_angle_range'] = gui_msg.within_angle_range
        self.p_head['rotate_180'] = gui_msg.rotate_180
        self.p_replay['enabled'] = gui_msg.replay_stim_enabled
        self.p_ripples['enabled'] = gui_msg.ripple_stim_enabled
        self.p_head['enabled'] = gui_msg.head_direction_stim_enabled

    def _update_ripples(self, msg):
        """Update current ripple state"""

        if msg[0]['is_consensus']:
            self._update_cons_ripple_status(msg)
        else:
            self._update_ripple_status(msg)

    def _update_ripple_status(self, msg):
        """Update ripple state for electrode group-based ripple detection"""

        ts = msg[0]['timestamp']
        trode = msg[0]['elec_grp_id']
        rtype = msg[0]['ripple_type']

        if rtype == 'end':

            # an 'end' ripple_type marks the end for standard, content, and
            # conditioning ripples (they are synchronized). so we need to
            # check all keys in the dictionary
            for rt in self._is_in_multichannel_ripple:

                # track end ripple of every trode, regardless of whether
                # this particular timestamp marks the end of a multichannel
                # ripple

                rtrode_ind = self._rtrode_ind_map[trode]
                self._ripple_ts[rt][rtrode_ind, 1] = ts

                if (
                    self._active_trodes[rt] is not None and
                    trode in self._active_trodes[rt]
                ):

                    self._active_trodes[rt].remove(trode)

                # all trodes that had triggered ripple have completed
                if self._active_trodes[rt] == [] and self._is_in_multichannel_ripple[rt]:

                    self.write_record(
                        binary_record.RecordIDs.STIM_RIPPLE_EVENT,
                        self._ripple_event_ts[rt], ts,
                        self._ripple_trigger_trode[rt], trode,
                        bytes(rt, 'utf-8'), False,
                        self.p_ripples['num_above_thresh'],
                        self._ripple_sms[rt],
                        *self._ripple_ts[rt].flatten(),
                        *self._is_rtrode_active[rt]
                    )

                    self._ripple_ts[rt][:] = -1
                    self._is_rtrode_active[rt][:] = False
                    self._is_in_multichannel_ripple[rt] = False

        else: # must be a ripple onset message

            # keeping track of every timestamp. not subject to event lockout
            # constraint
            rtrode_ind = self._rtrode_ind_map[trode]
            self._ripple_ts[rtype][rtrode_ind, 0] = ts

            if (
                not self._is_in_multichannel_ripple[rtype] and
                ts > self._ripple_event_ts[rtype] + self._ripple_event_ls
            ):

                num_above = self.p_ripples['num_above_thresh']
                S = self._stwin_samples

                cand_trodes = self._rtrodes
                cand_timestamps = self._ripple_ts[rtype][:, 0]

                # rearrange trodes according to detection times
                sort_inds = np.argsort(cand_timestamps)
                cand_trodes = cand_trodes[sort_inds]
                cand_timestamps = cand_timestamps[sort_inds]

                # minimum number of trodes are above threshold
                # now determine if they satisfy the criteria
                # that their ripple times fell within a
                # user-specified time window
                if np.sum(cand_timestamps > -1) >= num_above:

                    most_recent_ts = cand_timestamps[-1]

                    if most_recent_ts - cand_timestamps[-num_above] <= S:

                        active_trodes = cand_trodes[-num_above:]
                        for active_trode in active_trodes:
                            active_trode_ind = self._rtrode_ind_map[active_trode]
                            self._is_rtrode_active[rtype][active_trode_ind] = True

                        self._ripple_trigger_trode[rtype] = trode
                        self._active_trodes[rtype] = list(active_trodes) # forces copy
                        self._ripple_event_ts[rtype] = ts
                        self._is_in_multichannel_ripple[rtype] = True

                        send_shortcut_message = self._check_send_shortcut(
                            (
                                self.p_ripples['enabled'] and
                                self.p_ripples['type'] == rtype and
                                self.p_ripples['method'] == 'multichannel'
                            )
                        )
                        if send_shortcut_message:
                            self._trodes_client.send_statescript_shortcut_message(14)

                        self._ripple_sms[rtype] = send_shortcut_message


    def _update_cons_ripple_status(self, msg):
        """Update ripple state for consensus trace-based ripple detection"""

        ts = msg[0]['timestamp']
        trode = msg[0]['elec_grp_id']
        rtype = msg[0]['ripple_type']

        if rtype == 'end':

            for rt in self._is_in_consensus_ripple:

                if (
                    self._is_in_consensus_ripple[rt] and
                    ts > self._cons_ripple_event_ts[rt]
                ):

                    self.write_record(
                        binary_record.RecordIDs.STIM_RIPPLE_EVENT,
                        self._cons_ripple_event_ts[rt], ts,
                        -1, -1,
                        bytes(rt, 'utf-8'), True,
                        -1, self._cons_ripple_sms[rt],
                        *np.full(len(self._rtrodes)*2, -1, dtype=int),
                        *np.zeros(len(self._rtrodes), dtype=bool)
                    )

                    # self.write_record(
                    #     binary_record.RecordIDs.STIM_RIPPLE_END,
                    #     ts, bytes(rt, 'utf-8'), True,
                    #     -1, -1
                    # )
                    self._is_in_consensus_ripple[rt] = False

        else: # must be start

            if (
                not self._is_in_consensus_ripple[rtype] and
                ts > self._cons_ripple_event_ts[rtype] + self._cons_ripple_event_ls
            ):

                self._cons_ripple_event_ts[rtype] = ts
                self._is_in_consensus_ripple[rtype] = True

                send_shortcut_message = self._check_send_shortcut(
                    (
                        self.p_ripples['enabled'] and
                        self.p_ripples['type'] == rtype and
                        self.p_ripples['method'] == 'consensus'
                    )
                )

                if send_shortcut_message:
                    self._trodes_client.send_statescript_shortcut_message(14)

                self._cons_ripple_sms[rtype] = send_shortcut_message

                # self.write_record(
                #     binary_record.RecordIDs.STIM_RIPPLE_DETECTED,
                #     ts, bytes(rtype, 'utf-8'), True,
                #     -1, -1, send_shortcut_message
                # )

    def _update_velocity_position(self, msg):
        """Update data relevant to a new position & velocity data point"""

        self._pos_msg_ct += 1

        self._current_pos = msg[0]['mapped_pos']
        self._current_vel = msg[0]['velocity']

        self._update_head_direction(msg)

        if self._pos_msg_ct % self.p['num_pos_points'] == 0:
            self._task_state = self._task_state_handler.get_task_state(
                msg[0]['timestamp']
            )

        if self._pos_msg_ct % self.p['num_pos_disp'] == 0:
            print(
                'position', self._current_pos,
                'velocity', np.around(self._current_vel, decimals=2),
                'segment', msg[0]['segment'],
                'raw_x', msg[0]['raw_x'], 'raw_y', msg[0]['raw_y'],
                'angle', np.around(self._head_angle, decimals=2),
                'angle_well_1', np.around(self._angle_well_1, decimals=1),
                'angle_well_2', np.around(self._angle_well_2, decimals=1)
            )

    def _update_head_direction(self, msg):
        """Compute head direction angles and stimulate for a head
        direction event if enabled"""

        angle, angle_well_1, angle_well_2 = self._compute_angles(
            msg
        )

        self._head_angle = angle
        self._angle_well_1 = angle_well_1
        self._angle_well_2 = angle_well_2

        N = len(self._angle_buffer)
        self._angle_buffer[self._angle_buffer_ind] = self._head_angle
        self._angle_buffer_ind = (self._angle_buffer_ind + 1) % N

        # not completely populated with real data yet
        if -1000 in self._angle_buffer:
            return

        is_within_angle_range = (
            (
                abs(max(self._angle_buffer) - min(self._angle_buffer)) <=
                self.p_head['within_angle_range']
            )
        )

        x = (msg[0]['raw_x'] + msg[0]['raw_x2'])/2
        y = (msg[0]['raw_y'] + msg[0]['raw_y2'])/2
        self._center_well_dist = np.sqrt(
            (x - self._center_well_loc[0])**2 + (y - self._center_well_loc[1])**2
        ) * self.p['scale_factor']

        # this value is also used for replay event detection and describes
        # whether the animal is CURRENTLY near the center well. to be accurate,
        # it is important to minimize the decoder delay i.e.
        # config['decoder']['time_bin']['delay_samples']
        self._is_center_well_proximate = self._center_well_dist <= self.p['max_center_well_dist']

        ts = msg[0]['timestamp']

        if (
            is_within_angle_range and
            self._is_center_well_proximate and
            ts > self._head_event_ts + self._head_event_ls
        ):

            well = 0 # will this value ever be logged?
            send_shortcut_message = False
            record = False

            # # indeterminate
            # if (
            #     abs(angle - angle_well_1) <= self.p_head['well_angle_range'] and
            #     abs(angle - angle_well_2) <= self.p_head['well_angle_range']
            # ):
            #     self.class_log.warning(
            #         "Could not decide which well to choose! Animal is within "
            #         "the angle range of both wells"
            #     )

            # well 1
            if abs(angle - angle_well_1) <= self.p_head['well_angle_range']:

                print(
                    "Head direction event arm 1", np.around(angle, decimals=2),
                    "at time", np.around(ts/30000, decimals=2),
                    "angle to target", np.around(angle_well_1, decimals=2)
                )
                well = 1
                record = True
                send_shortcut_message = self._check_send_shortcut(
                    self.p_head['enabled']
                )

            # well 2
            if abs(angle - angle_well_2) <= self.p_head['well_angle_range']:

                print(
                    "Head direction event arm 2", np.around(angle, decimals=2),
                    "at time", np.around(ts/30000, decimals=2),
                    "angle to target", np.around(angle, decimals=2)
                )
                well = 2
                record = True
                send_shortcut_message = self._check_send_shortcut(
                    self.p_head['enabled']
                )

            if send_shortcut_message:
                self._trodes_client.send_statescript_shortcut_message(14)

            if record:
                self._head_event_ts = ts
                self.write_record(
                    binary_record.RecordIDs.STIM_HEAD_DIRECTION,
                    ts, send_shortcut_message, well,
                    msg[0]['raw_x'], msg[0]['raw_y'],
                    msg[0]['raw_x2'], msg[0]['raw_y2'],
                    angle, angle_well_1, angle_well_2,
                    self.p_head['well_angle_range'],
                    self.p_head['within_angle_range'],
                    self._center_well_dist,
                    self.p['max_center_well_dist'],
                    self.p_head['min_duration'],
                    self.p_head['rotate_180']
                )


    def _compute_angles(self, msg):
        """Compute head direction angles relative to the reward wells"""

        x1 = msg[0]['raw_x']
        y1 = msg[0]['raw_y']
        x2 = msg[0]['raw_x2']
        y2 = msg[0]['raw_y2']

        x = (x1 + x2)/2
        y = (y2 + y2)/2

        # For all of these calculations, since the origin is
        # at the top left of the image, we negate y2 - y1
        # in the call to np.arctan2() because we're used to
        # an origin at the bottom left of the image

        if self.p_head['rotate_180']:
            head_angle = (
                (180/np.pi) * np.arctan2(-(y1 - y2), x1 - x2)
            )
        else:
            head_angle = (
                (180/np.pi) * np.arctan2(-(y2 - y1), x2 - x1)
            )

        # 120 is arm 2
        angle_well_1 = np.random.choice([130])
        angle_well_1 = 180 / np.pi * (
            np.arctan2(-(self._well_1_y - y), self._well_1_x - x)
        )

        # 60 is arm 1
        angle_well_2 = np.random.choice([70])
        angle_well_2 = 180 / np.pi * (
            np.arctan2(-(self._well_2_y - y), self._well_2_x - x)
        )

        return head_angle, angle_well_1, angle_well_2

    def _update_posterior(self, msg):
        """Handle a message containing posterior data"""

        # set various indices and counts
        self._dec_ind = self._decoder_rank_ind_map[msg[0]['rank']]
        self._dd_ind = self._dd_inds[self._dec_ind]

        # run data processing methods
        self._update_spike_stats(msg)
        self._update_decode_stats(msg)

        if self.p['instructive']:
            self._find_replay_instructive(msg)
        else:
            self._find_replay(msg)

        # advance the relevant decoder data index and counters
        self._dd_inds[self._dec_ind] = (
            (self._dd_inds[self._dec_ind] + 1) %
            self.p_replay['sliding_window']
        )
        self._decoder_count[self._dec_ind] += 1

    def _update_spike_stats(self, msg):
        """Update spiking statistics"""

        ind = self._dec_ind

        # add new spike counts
        self._spike_count[ind] = msg[0]['spike_count']
        self._event_spike_count[ind, self._dd_ind] = msg[0]['spike_count']

        self._update_mua_stats(ind, msg)

        if self._decoder_count[ind] % self.p['num_dec_disp'] == 0:
            print(
                'Decoder', ind, 'firing rate:',
                '(mean:', np.around(self._bin_fr_means[ind], decimals=3),
                'std:', np.around(self._bin_fr_std[ind], decimals=3), ')',
            )

    def _update_mua_stats(self, ind, msg):
        """Update MUA-based spiking rates"""

        spike_rate = msg[0]['spike_count'] / self._dt

        # unlike the MUA, the binned spike firing rate stats
        # cannot be frozen or seeded with initial values, as
        # they are meant to be purely informative (no further
        # computation is done with them)
        (self._bin_fr_means[ind],
        self._bin_fr_M2[ind],
        self._bin_fr_N[ind]
        ) = utils.estimate_new_stats(
            spike_rate,
            self._bin_fr_means[ind],
            self._bin_fr_M2[ind],
            self._bin_fr_N[ind]
        )
        self._bin_fr_std[ind] = np.sqrt(
            self._bin_fr_M2[ind] / self._bin_fr_N[ind]
        )

        self._detect_mua_events(ind, spike_rate)

    def _detect_mua_events(self, ind, spike_rate):
        """Detect MUA burst events"""

        N = self.p['mua_window']
        self._mua_buf[ind, self._decoder_count[ind] % N] = spike_rate
        mua_datapoint = np.mean(self._mua_buf[ind])

        if not self.p['mua_freeze_stats']:
            (self._mua_means[ind],
            self._mua_M2[ind],
            self._mua_N[ind]
            ) = utils.estimate_new_stats(
            mua_datapoint,
            self._mua_means[ind],
            self._mua_M2[ind],
            self._mua_N[ind]
            )
            self._mua_std[ind] = np.sqrt(
                self._mua_M2[ind] / self._mua_N[ind]
            )

        sigma1 = self.p['mua_trigger_thresh']
        sigma2 = self.p['mua_end_thresh']
        if (
            not self._in_burst[ind] and
            mua_datapoint > self._mua_means[ind] + sigma1 * self._mua_std[ind]
        ):
            self._in_burst[ind] = True

        elif (
            self._in_burst[ind] and
            mua_datapoint <= self._mua_means[ind] + sigma2 * self._mua_std[ind]
        ):
            self._in_burst[ind] = False

    def _update_decode_stats(self, msg):
        """Update statistics based on new data containing
        posterior and likelihood estimates"""

        ind = self._dec_ind

        if self.p_replay['method'] == 'posterior':
            marginal_prob = msg[0]['posterior'].sum(axis=0)
            ci_decoder = msg[0]['cred_int_post']
        else: # use likelihood
            marginal_prob = msg[0]['likelihood']
            ci_decoder = msg[0]['cred_int_lk']

        decoder_argmax = np.argmax(marginal_prob)

        # add new credible intervals and argmax of decoder
        # probability
        self._dec_ci_buff[ind, self._dd_ind] = ci_decoder
        self._dec_argmax_buff[ind, self._dd_ind] = decoder_argmax

        # add new credible intervals and argmaxes from encoder
        # data
        self._enc_ci_buff[ind, self._dd_ind] = msg[0]['enc_cred_intervals']
        self._enc_argmax_buff[ind, self._dd_ind] = msg[0]['enc_argmaxes']

        self._update_prob_sums(marginal_prob)

    def _update_prob_sums(self, marginal_prob):
        """Compute probability sums for specific regions in the maze"""

        ind = self._dec_ind

        arm_probs = self._compute_arm_probs(marginal_prob)
        self._arm_ps_buff[ind, self._dd_ind] = arm_probs

        ps_arm1, ps_arm2, ps_arm1_base, ps_arm2_base = self._compute_region_probs(
            marginal_prob
        )

        # add new posterior/likelihood probability sums for desired regions
        # perhaps we want to compute probability sum for box eventually?
        self._region_ps_buff[ind, self._dd_ind, 0] = np.nan
        self._region_ps_buff[ind, self._dd_ind, 1] = ps_arm1
        self._region_ps_buff[ind, self._dd_ind, 2] = ps_arm2

        self._region_ps_base_buff[ind, self._dd_ind, 0] = np.nan
        self._region_ps_base_buff[ind, self._dd_ind, 1] = ps_arm1_base
        self._region_ps_base_buff[ind, self._dd_ind, 2] = ps_arm2_base

    def _compute_arm_probs(self, prob):
        """Compute the probability sum for each maze arm"""

        arm_probs = np.zeros(len(self.p['arm_coords']))
        for ii, (a, b) in enumerate(self.p['arm_coords']):
            arm_probs[ii] = prob[a:b+1].sum()

        return arm_probs

    def _compute_region_probs(self, prob):
        """Compute probability sums for specific regions in the maze"""

        # The particular two-arm maze this object was written for
        # does not change its topology, hence the hard-coding.
        # Nevertheless it might be useful to eventually make
        # this configurable if the position bin size changes,
        # for example
        ps_arm1 = prob[20:25].sum()
        ps_arm2 = prob[36:41].sum()
        ps_arm1_base = prob[13:18].sum()
        ps_arm2_base = prob[29:34].sum()

        return ps_arm1, ps_arm2, ps_arm1_base, ps_arm2_base

    def _find_replay(self, msg):
        """Look for a replay event for a noninstructive task"""

        ts = msg[0]['bin_timestamp_r']
        ind = self._dec_ind

        num_unique = np.count_nonzero(self._enc_ci_buff)

        # don't even bother looking for replay if the basic requirements
        # are not met
        if not (
            ts > self._replay_event_ts + self._replay_event_ls and
            num_unique >= self.p_replay['min_unique_trodes'] and
            self._is_center_well_proximate
        ):
            return

        if self._num_decoders == 2:

            primary_arm_thresh = self.p_replay['primary_arm_threshold']
            secondary_arm_thresh = self.p_replay['secondary_arm_threshold']
            other_arm_thresh = self.p_replay['other_arm_threshold']

            avg_arm_ps_1 = np.mean(self._arm_ps_buff[0], axis=0)
            avg_arm_ps_2 = np.mean(self._arm_ps_buff[1], axis=0)

            # if at least one of the decoders crosses the primary
            # threshold, then we determine which of them has the
            # higher average arm probability sum. the decoder with
            # the lower average arm probability sum has to cross
            # the secondary threshold

            # arm 1 candidate event
            if (
                avg_arm_ps_1[1] > primary_arm_thresh or
                avg_arm_ps_2[1] > primary_arm_thresh
            ):

                if (
                    avg_arm_ps_1[1] > avg_arm_ps_2[1] and
                    avg_arm_ps_2[1] > secondary_arm_thresh and
                    np.all(avg_arm_ps_1[[0, 2]] < other_arm_thresh) and
                    np.all(avg_arm_ps_2[[0, 2]] < other_arm_thresh)
                ):

                    self._handle_replay(1, msg)

                elif (
                    avg_arm_ps_2[1] > avg_arm_ps_1[1] and
                    avg_arm_ps_1[1] > secondary_arm_thresh and
                    np.all(avg_arm_ps_1[[0, 2]] < other_arm_thresh) and
                    np.all(avg_arm_ps_2[[0, 2]] < other_arm_thresh)
                ):

                    self._handle_replay(1, msg)

            # arm 2 candidate event
            elif (
                avg_arm_ps_1[2] > primary_arm_thresh or
                avg_arm_ps_2[2] > primary_arm_thresh
            ):

                if (
                    avg_arm_ps_1[2] > avg_arm_ps_2[2] and
                    avg_arm_ps_2[2] > secondary_arm_thresh and
                    np.all(avg_arm_ps_1[[0, 1]] < other_arm_thresh) and
                    np.all(avg_arm_ps_2[[0, 1]] < other_arm_thresh)
                ):
                    self._handle_replay(2, msg)

                elif (
                    avg_arm_ps_2[2] > avg_arm_ps_1[2] and
                    avg_arm_ps_1[2] > secondary_arm_thresh and
                    np.all(avg_arm_ps_1[[0, 1]] < other_arm_thresh) and
                    np.all(avg_arm_ps_2[[0, 1]] < other_arm_thresh)
                ):
                    self._handle_replay(2, msg)

        else:

            arm_thresh = self.p_replay['primary_arm_threshold']
            other_arm_thresh = self.p_replay['other_arm_threshold']

            avg_arm_ps = np.mean(self._arm_ps_buff[ind], axis=0)

            # arm 1 candidate event
            if (
                avg_arm_ps[1] > arm_thresh and
                np.all(avg_arm_ps[[0, 2]] < other_arm_thresh)
            ):
                self._handle_replay(1, msg)

            # arm 2 candidate event
            elif (
                avg_arm_ps[2] > arm_thresh and
                np.all(avg_arm_ps[[0, 1]] < other_arm_thresh)
            ):
                self._handle_replay(2, msg)

    def _handle_replay(self, arm, msg):
        """Handle a replay event for a non-instructive task"""

        # assumes already satisfied event lockout and minimum unique
        # trodes criteria. all these events should therefore be recorded

        print(f"Replay arm {arm} detected")

        self._replay_event_ts = msg[0]['bin_timestamp_r']

        num_unique = np.count_nonzero(self._enc_ci_buff)
        print(f"Unique trodes: {num_unique}")

        send_shortcut = self._check_send_shortcut(
            self.p_replay['enabled'] and
            arm == self.p_replay['target_arm']
        )

        if send_shortcut:
            self._trodes_client.send_statescript_shortcut_message(14)
            self._num_rewards[arm] += 1
            self.send_interface.send_num_rewards(self._num_rewards)
            print(f"Replay arm {arm} rewarded")

        self.write_record(
            binary_record.RecordIDs.STIM_MESSAGE,
            msg[0]['bin_timestamp_l'], msg[0]['bin_timestamp_r'],
            send_shortcut, self._delay, self._current_vel, self._current_pos,
            self._task_state, arm, self.p_replay['target_arm'],
            self.p_replay['primary_arm_threshold'], self._max_repeats,
            self._replay_window_time, self.p['instructive'],
            num_unique, self._center_well_dist,
            self.p['max_center_well_dist'],
            self._is_in_multichannel_ripple['standard'],
            self._is_in_multichannel_ripple['cond'],
            self._is_in_multichannel_ripple['content'],
            self._is_in_consensus_ripple['standard'],
            self._is_in_consensus_ripple['cond'],
            self._is_in_consensus_ripple['content'],
            *self._in_burst, *self._spike_count,
            *self._event_spike_count.sum(axis=1), *self._bin_fr_means,
            *self._enc_ci_buff.mean(axis=-1).mean(axis=-1),
            *self._region_ps_buff.mean(axis=1).flatten(),
            *self._region_ps_base_buff.mean(axis=1).flatten(),
            *self._arm_ps_buff.mean(axis=1).flatten()
        )

    def _find_replay_instructive(self, msg):
        """Look for a potential replay event for an instructive task"""

        ts = msg[0]['bin_timestamp_r']
        ind = self._dec_ind

        if self._num_decoders == 2:
            raise NotImplementedError(
                "Finding instructive replay events is not implemented "
                "for 2 decoders"
            )
        else:
            num_unique = np.count_nonzero(self._enc_ci_buff)

            # don't even bother looking for replay if the basic requirements are not
            # met
            if not (
                ts > self._replay_event_ts + self._replay_event_ls and
                num_unique >= self.p_replay['min_unique_trodes'] and
                self._is_center_well_proximate
            ):
                return

            arm_thresh = self.p_replay['primary_arm_threshold']
            other_arm_thresh = self.p_replay['other_arm_threshold']

            avg_region_ps = np.mean(self._region_ps_buff[ind], axis=0)
            avg_region_ps_base = np.mean(self._region_ps_base_buff[ind], axis=0)
            avg_arm_ps = np.mean(self._arm_ps_buff[ind], axis=0)

            # arm 1 candidate event
            if (
                avg_region_ps[1] > arm_thresh and
                avg_region_ps_base[1] > arm_thresh and
                np.all(avg_arm_ps[[0, 2]] < other_arm_thresh)
            ):
                self._handle_replay_instructive(1, msg)

            # arm 2 candidate event
            elif (
                avg_region_ps[2] > arm_thresh and
                avg_region_ps_base[2] > arm_thresh and
                np.all(avg_arm_ps[[0, 1]] < other_arm_thresh)
            ):
                self._handle_replay_instructive(2, msg)

    def _handle_replay_instructive(self, arm, msg):
        """Handle a replay event for an instructive task"""

        # assumes already satisfied event lockout and minimum unique
        # trodes criteria. all these events should therefore be recorded

        print(f"INSTRUCTIVE: Replay arm {arm} detected")

        self._replay_event_ts = msg[0]['bin_timestamp_r']

        num_unique = np.count_nonzero(self._enc_ci_buff)
        print(f"INSTRUCTIVE: Unique trodes: {num_unique}")

        outer_arm_visited = utils.get_last_num(self.p['instructive_file'])

        send_shortcut = self._check_send_shortcut(
            self.p_replay['enabled'] and
            arm == self.p_replay['target_arm'] and
            outer_arm_visited
        )

        if send_shortcut:
            self._trodes_client.send_statescript_shortcut_message(14)
            print(f"INSTRUCTIVE: Replay target arm {arm} rewarded")
            utils.write_text_file(self.p['instructive_file'], 0)
            self._instr_rewarded_arms[1:] = self._instr_rewarded_arms[:-1]
            self._instr_rewarded_arms[0] = arm
            self._num_rewards[arm] += 1
            self.send_interface.send_num_rewards(self._num_rewards)
            self._choose_next_instructive_target()

        self.write_record(
            binary_record.RecordIDs.STIM_MESSAGE,
            msg[0]['bin_timestamp_l'], msg[0]['bin_timestamp_r'],
            send_shortcut, self._delay,
            self._current_vel, self._current_pos,
            self._task_state, arm, self.p_replay['target_arm'],
            self.p_replay['primary_arm_threshold'], self._max_repeats,
            self._replay_window_time, self.p['instructive'],
            num_unique, self._center_well_dist,
            self.p['max_center_well_dist'],
            self._is_in_multichannel_ripple['standard'],
            self._is_in_multichannel_ripple['cond'],
            self._is_in_multichannel_ripple['content'],
            self._is_in_consensus_ripple['standard'],
            self._is_in_consensus_ripple['cond'],
            self._is_in_consensus_ripple['content'], *self._spike_count,
            *self._event_spike_count.sum(axis=1), *self._bin_fr_means,
            *self._enc_ci_buff.mean(axis=-1).mean(axis=-1),
            *self._region_ps_buff.mean(axis=1).flatten(),
            *self._region_ps_base_buff.mean(axis=1).flatten(),
            *self._arm_ps_buff.mean(axis=1).flatten()
        )

    def _choose_next_instructive_target(self):
        """Choose next target arm for an instructive task"""

        if np.all(self._instr_rewarded_arms == 1):
            print('INSTRUCTIVE: switch to arm 2')
            self.p_replay['target_arm'] = 2
        elif np.all(self._instr_rewarded_arms == 2):
            print('INSTRUCTIVE: switch to arm 1')
            self.p_replay['target_arm'] = 1
        else:
            self.p_replay['target_arm'] = np.random.choice([1,2],1)[0]
        print(f"INSTRUCTIVE: New target arm: {self.p_replay['target_arm']}")

    def _check_send_shortcut(self, other_condition):
        """Whether or not a shortcut message should be sent"""

        return self._task_state == 2 and other_condition

    def _init_ripple_misc(self, rtrodes, ripple_types):
        """Initialize data used for dealing with ripple data"""

        self._is_in_multichannel_ripple = {
            rtype: False for rtype in ripple_types
        }
        self._is_in_consensus_ripple = deepcopy(
            self._is_in_multichannel_ripple
        )

        self._rtrodes = np.array(rtrodes, dtype=int)

        # not recorded, but used for ripple detection book-keeping
        self._rtrode_ind_map = {}
        for ii, trode in enumerate(rtrodes):
            self._rtrode_ind_map[trode] = ii

        # will be set to variable-length list as needed, see
        # _update_ripple_status()
        self._active_trodes = {
            rtype: None for rtype in ripple_types
        }

        # ripple detection data recorded to disk
        # starts and ends of ripples
        self._ripple_ts = {
            rtype: np.full((len(rtrodes), 2), -1, dtype=int)
            for rtype in ripple_types
        }
        self._is_rtrode_active = {
            rtype: np.zeros(len(rtrodes), dtype=bool)
            for rtype in ripple_types
        }
        self._ripple_trigger_trode = {
            rtype: -1 for rtype in ripple_types
        }
        self._ripple_sms = { # shorthand for "shortcut message sent"
            rtype: False for rtype in ripple_types
        }
        self._cons_ripple_sms = deepcopy(self._ripple_sms)

        # suprathreshold window samples
        self._stwin_samples = int(
            np.around(
                self._config['stimulation']['ripples']['suprathreshold_period'] *
                self._config['sampling_rate']['spikes'],
                decimals=0
            )
        )

    def _init_head_dir(self):
        """Initialize data relevant to head direction data"""

        self._head_angle = 0
        self._center_well_loc = self._config['stimulation']['center_well_loc']

        well_loc = self._config['stimulation']['head_direction']['well_loc']
        self._well_1_x = well_loc[0][0]
        self._well_1_y = well_loc[0][1]
        self._well_2_x = well_loc[1][0]
        self._well_2_y = well_loc[1][1]

    def _init_data_buffers(self):
        """Initialize data objects"""

        # head direction params
        # only an approximation since camera module timestamps do not come in at
        # regularly spaced time intervals (although we assume the acquisition of
        # frames is more or less at a constant frame rate)
        div, rem = divmod(
            (
                self._config['sampling_rate']['position'] *
                self._config['stimulation']['head_direction']['min_duration']
            ),
            1
        )
        if rem:
            div += 1
        self._angle_buffer = np.ones(div) * -1000
        self._angle_buffer_ind = 0

        # generate the lookup mapping. doesn't need to be an OrderedDict
        self._decoder_rank_ind_map = {} # key: rank, value: index
        for ii, rank in enumerate(self._config['rank']['decoders']):
            self._decoder_rank_ind_map[rank] = ii

        N = self._config['stimulation']['replay']['sliding_window']
        num_trodes = self._config['decoder']['cred_int_bufsize']
        num_decoders = self._num_decoders

        self._dec_ind = 0 # decoder index of current posterior message
        self._dd_inds = [0] * num_decoders # shorthand for decode data

        # dim 2 is 3 because 3 regions - box, arm1, arm2
        # _ps_ - shorthand for probability sum
        self._arm_ps_buff = np.zeros((num_decoders, N, 3))
        self._region_ps_buff = np.zeros_like(self._arm_ps_buff)
        self._region_ps_base_buff = np.zeros_like(self._arm_ps_buff)

        self._dec_ci_buff = np.zeros((num_decoders, N))
        self._dec_argmax_buff = np.zeros_like(self._dec_ci_buff)

        self._enc_ci_buff = np.zeros((num_decoders, N, num_trodes))
        self._enc_argmax_buff = np.zeros_like(self._enc_ci_buff)

        # running stats of spiking rate
        self._bin_fr_means = np.zeros(num_decoders)
        self._bin_fr_M2 = np.zeros(num_decoders)
        self._bin_fr_N = np.zeros(num_decoders)
        self._bin_fr_std = np.zeros(num_decoders)
        self._dt = (
            self._config['decoder']['time_bin']['samples'] /
            self._config['sampling_rate']['spikes']
        )

        self._spike_count = np.zeros(num_decoders, dtype=int)
        self._event_spike_count = np.zeros((num_decoders, N), dtype=int)

        # stats of mua
        self._mua_means = np.zeros(num_decoders)
        self._mua_M2 = np.zeros(num_decoders)
        self._mua_N = np.zeros(num_decoders)
        self._mua_std = np.zeros(num_decoders)
        self._mua_buf = np.zeros((
            num_decoders, self._config['mua']['moving_avg_window']
        ))

        # whether mua threshold crossed
        self._in_burst = np.full(num_decoders, False, dtype=bool)

    def _seed_mua_stats(self):
        """Set the initial MUA stats"""

        try:
            for rank, ind in self._decoder_rank_ind_map.items():
                self.class_log.debug(f"Seeded MUA stats for decoder rank {rank}")
                self._mua_means[ind] = self._config['mua']['custom_mean'][rank]
                self._mua_N[ind] = 1
                self._mua_std[ind] = self._config['mua']['custom_std'][rank]
                self._mua_M2[ind] = self._mua_std[ind]**2 * self._mua_N[ind]
        except:
            pass

    def _init_stim_params(self):
        """Initialize parameters governing stimulation"""

        # Convention
        # ts - timestamp
        # ls - lockout samples

        ##################################################################
        # Replay
        self._replay_event_ts = 0
        self._replay_event_ls = int(
            self._config['sampling_rate']['spikes'] *
            self._config['stimulation']['replay']['event_lockout']
        )

        ##################################################################
        # Ripples
        ripple_types = ('standard', 'cond', 'content')
        self._ripple_event_ts = {
            rtype: 0 for rtype in ripple_types
        }
        self._ripple_event_ls = int(
            self._config['sampling_rate']['spikes'] *
            self._config['stimulation']['ripples']['event_lockout']
        )

        # Initial consensus ripple variables same as for multichannel
        # ripple variables
        self._cons_ripple_event_ts = {
            rtype: 0 for rtype in ripple_types
        }
        self._cons_ripple_event_ls = self._ripple_event_ls

        ##################################################################
        # Head direction
        self._head_event_ts = 0
        self._head_event_ls = int(
            self._config['sampling_rate']['spikes'] *
            self._config['stimulation']['head_direction']['event_lockout']
        )

    def _init_params(self):
        """Initialize parameters used by this object"""

        self.p = {}
        self.p['instructive_file'] = self._config[self._config['datasource']]['instructive_file']
        self.p['scale_factor'] = self._config['kinematics']['scale_factor']
        self.p['instructive'] = self._config['stimulation']['instructive']
        # self.p['reward_mode'] = self._config['stimulation']['reward_mode']
        # self.p['shortcut_msg_on'] = self._config['stimulation']['shortcut_msg_on']
        self.p['num_pos_points'] = self._config['stimulation']['num_pos_points']
        self.p['num_pos_disp'] = self._config['display']['stim_decider']['position']
        self.p['num_dec_disp'] = self._config['display']['stim_decider']['decoding_bins']
        self.p['max_center_well_dist'] = self._config['stimulation']['max_center_well_dist']
        self.p['arm_coords'] = self._config['encoder']['position']['arm_coords']
        self.p['mua_trigger_thresh'] = self._config['mua']['threshold']['trigger']
        self.p['mua_end_thresh'] = self._config['mua']['threshold']['end']
        self.p['mua_window'] = self._config['mua']['moving_avg_window']
        self.p['mua_freeze_stats'] = self._config['mua']['freeze_stats']

        # verify some inputs
        replay_method = self._config['stimulation']['replay']['method']
        if replay_method not in ('posterior', 'likelihood'):
            raise ValueError(
                f"Invalid method {replay_method} for replay"
            )

        ripple_method = self._config['stimulation']['ripples']['method']
        if ripple_method not in ('multichannel', 'consensus'):
            raise ValueError(
                f"Invalid method {ripple_method} for ripples"
            )

        # copy replay, ripples, and head direction config information
        # into separate dictionaries
        self.p_replay = {}
        for k, v in self._config['stimulation']['replay'].items():
            self.p_replay[k] = v

        self.p_ripples = {}
        for k, v in self._config['stimulation']['ripples'].items():
            self.p_ripples[k] = v

        self.p_head = {}
        for k, v in self._config['stimulation']['head_direction'].items():
            self.p_head[k] = v