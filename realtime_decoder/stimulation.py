import time
import numpy as np

from copy import deepcopy

from realtime_decoder import base, utils, messages, binary_record

class TwoArmTrodesStimDecider(base.BinaryRecordBase, base.MessageHandler):

    def __init__(self, rank, config, trodes_client):

        super().__init__(
            rank=rank,
            rec_ids=[
                binary_record.RecordIDs.STIM_RIPPLE_DETECTED,
                binary_record.RecordIDs.STIM_RIPPLE_END
            ],
            rec_labels=[
                ['timestamp', 'ripple_type', 'is_consensus', 'trigger_trode', 'num_above_thresh', 'shortcut_message_sent'],
                ['timestamp', 'ripple_type', 'is_consensus', 'trigger_trode', 'num_above_thresh']
            ],
            rec_formats=['q10s?ii?', 'q10s?ii'],
            manager_label='state'
        )

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

        self._config = config
        self._trodes_client = trodes_client

        self._task_state = 1
        self._num_rewards = np.zeros(
            len(self._config['encoder']['position']['arm_coords']),
            dtype='=i4'
        )
        self._instr_rewarded_arms = np.zeros(
            self._config['stimulation']['replay']['instr_max_repeats'],
            dtype='=i4'
        )

        self._pos_msg_ct = 0
        self._current_pos = 0
        self._current_vel = 0

        ripple_types = ('standard', 'cond', 'content')
        self._ripple_trodes = {
            rtype: [] for rtype in ripple_types
        }
        self._ripple_timestamps = deepcopy(self._ripple_trodes)
        self._is_in_multichannel_ripple = {
            rtype: False for rtype in ripple_types
        }
        self._is_in_consensus_ripple = deepcopy(self._is_in_multichannel_ripple)

        self._center_well_loc = self._config['stimulation']['center_well_loc']
        well_loc = self._config['stimulation']['head_direction']['well_loc']
        self._well_1_x = well_loc[0][0]
        self._well_1_y = well_loc[0][1]
        self._well_2_x = well_loc[1][0]
        self._well_2_y = well_loc[1][1]
        self._head_angle = 0
        self._is_center_well_proximate = False

        self._init_stim_params()
        self._init_data_buffers()
        self._init_params()

    def handle_message(self, msg, mpi_status):

        # feedback, velocity/position, posterior
        if isinstance(msg, messages.GuiMainParameters):
            self._update_gui_params(msg)
        elif mpi_status.tag == messages.MPIMessageTag.RIPPLE_DETECTION:
            self._update_ripples(msg)
        elif mpi_status.tag == messages.MPIMessageTag.VEL_POS:
            self._update_velocity_position(msg)
        elif mpi_status.tag == messages.MPIMessageTag.POSTERIOR:
            pass
            ###################################################################################################################
            # Implement when ready
            # self._update_posterior(msg)
            ####################################################################################################################
        else:
            self._class_log.warning(
                f"Received message of type {type(msg)} "
                f"from source: {mpi_status.source}, "
                f" tag: {mpi_status.tag}, ignoring"
            )

    def _update_gui_params(self, gui_msg):
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

        if msg[0]['is_consensus']:
            self._update_cons_ripple_status(msg)
        else:
            self._update_ripple_status(msg)

    def _update_ripple_status(self, msg):

        ts = msg[0]['timestamp']
        trode = msg[0]['elec_grp_id']
        rtype = msg[0]['ripple_type']

        if rtype == 'end':

            # an 'end' ripple_type marks the end for standard, content, and
            # conditioning ripples (they are synchronized). so we need to
            # check all keys in the dictionary
            for rt in self._is_in_multichannel_ripple:

                if (
                    self._is_in_multichannel_ripple[rt] and
                    trode in self._ripple_trodes[rt]
                ):

                    self._ripple_trodes[rt].remove(trode)

                    # all ripple trodes that triggered ripple start have
                    # finished their ripples
                    if self._ripple_trodes[rt] == []:

                        self._ripple_timestamps[rt] = [] # reset vector of timestamps
                        self._is_in_multichannel_ripple[rt] = False
                        #######################################################################################
                        # log end of ripple, trigger trode, rtype
                        #######################################################################################
                        self.write_record(
                            binary_record.RecordIDs.STIM_RIPPLE_END,
                            ts, bytes(rt, 'utf-8'), False, trode, self.p_ripples['num_above_thresh']
                        )

        else: # must be a ripple onset message

            # ok to add ripple trodes
            if (
                not self._is_in_multichannel_ripple[rtype] and
                ts > self._ripple_event_ts[rtype] + self._ripple_event_ls
            ):

                self._ripple_trodes[rtype].append(trode)
                self._ripple_timestamps[rtype].append(ts)

                # now check if number of ripple trodes exceeds minimum
                if len(self._ripple_trodes[rtype]) >= self.p_ripples['num_above_thresh']:

                    assert len(self._ripple_trodes[rtype]) == self.p_ripples['num_above_thresh']

                    self._is_in_multichannel_ripple[rtype] = True

                    #####################################################################################################################
                    # log ripple message - trigger trode, trodes contributing to ripple, timestamps, etc.
                    #####################################################################################################################
                    self._ripple_event_ts[rtype] = ts

                    send_shortcut_message = self._check_send_shortcut(
                        (
                            self.p_ripples['enabled'] and
                            self.p_ripples['type'] == rtype and
                            self.p_ripples['method'] == 'multichannel'
                        )
                    )
                    if send_shortcut_message:
                        pass
                        #################################################################################################################
                        # send shortcut message
                        #################################################################################################################
                    self.write_record(
                        binary_record.RecordIDs.STIM_RIPPLE_DETECTED,
                        ts, bytes(rtype, 'utf-8'), False, trode,
                        self.p_ripples['num_above_thresh'], send_shortcut_message
                    )

    def _update_cons_ripple_status(self, msg):

        ts = msg[0]['timestamp']
        trode = msg[0]['elec_grp_id']
        rtype = msg[0]['ripple_type']

        if rtype == 'end':
            #############################################################################################################################
            for rt in self._is_in_consensus_ripple:

                if self._is_in_consensus_ripple[rt]:
                    self.write_record(
                        binary_record.RecordIDs.STIM_RIPPLE_END,
                        ts, bytes(rt, 'utf-8'), True,
                        -1, -1
                    )
                    self._is_in_consensus_ripple[rt] = False
            # log ripple end
            #############################################################################################################################

        else: # must be start

            if (
                not self._is_in_consensus_ripple[rtype] and
                ts > self._cons_ripple_event_ts[rtype] + self._cons_ripple_event_ls
            ):

                #########################################################################################################################
                # log ripple started
                #########################################################################################################################
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
                    pass
                    ####################################################################################################################
                    # send shortcut message
                    ###################################################################################################################
                self.write_record(
                    binary_record.RecordIDs.STIM_RIPPLE_DETECTED,
                    ts, bytes(rtype, 'utf-8'), True,
                    -1, -1, send_shortcut_message
                )

    def _update_velocity_position(self, msg):

        self._pos_msg_ct += 1

        self._current_pos = msg[0]['mapped_pos']
        self._current_vel = msg[0]['velocity']

        self._update_head_direction(msg)

        if (
            self._pos_msg_ct % self.p['num_pos_points'] == 0
        ):
            self._task_state == utils.get_last_num(
                self.p['taskstate_file']
            )

        if self._pos_msg_ct % self.p['num_pos_disp'] == 0:
            print(
                'position', self._current_pos,
                'velocity', np.around(self._current_vel, decimals=2),
                'segment', msg[0]['segment'],
                'raw_x', msg[0]['raw_x'], 'raw_y', msg[0]['raw_y'],
                'angle', np.around(self._head_angle, decimals=2),
                f'angle_well1 {self._angle_well_1:0.2f} ',
                f'angle_well2 {self._angle_well_2:0.2f}'
            )

    def _update_head_direction(self, msg):

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
        dist = np.sqrt(
            (x - self._center_well_loc[0])**2 + (y - self._center_well_loc[1])**2
        ) * self.p['scale_factor']

        # this value is also used for replay event detection and describes
        # whether the animal is CURRENTLY near the center well. to be accurate,
        # it is important to minimize the decoder delay i.e.
        # config['decoder']['time_bin']['delay_samples']
        self._is_center_well_proximate = dist <= self.p['max_center_well_dist']

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
                    "Head direction event arm 1", angle,
                    np.around(ts/30000, decimals=2),
                    "angle to target", angle_well_1
                )
                well = 1
                record = True
                send_shortcut_message = self._check_send_shortcut(
                    self.p_head['enabled']
                )

            # well 2
            if abs(angle - angle_well_2) <= self.p_head['well_angle_range']:

                print(
                    "Head direction event arm 2", angle,
                    np.around(ts/30000, decimals=2),
                    "angle to target", angle_well_2
                )
                well = 2
                record = True
                send_shortcut_message = self._check_send_shortcut(
                    self.p_head['enabled']
                )

            if record:
                self._head_event_ts = ts
                ############################################################################################################
                # record head stim message
                #############################################################################################################

            if send_shortcut_message:
                pass
                ############################################################################################################
                # send shortcut message
                ############################################################################################################

    def _compute_angles(self, msg):
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

        self._dec_ind = self._decoder_rank_ind_map[msg[0]['rank']]

        self._update_spike_stats(msg)
        self._update_decode_stats(msg)

        if self.p['instructive']:
            self._find_replay_instructive(msg)
        else:
            self._find_replay(msg)

        self._dd_ind = (self._dd_ind + 1) % self.p_replay['sliding_window']

    def _update_spike_stats(self, msg):

        ind = self._dec_ind

        # add new spike counts
        self._spike_count[ind] = msg[0]['spike_count']
        self._event_spike_count[ind, self._dd_ind] = msg[0]['spike_count']

        self._update_bin_firing_rate(ind, msg)

    def _update_bin_firing_rate(self, ind, msg):

        spike_rate = msg[0]['spike_count'] / self._dt
        (self._bin_fr_means[ind],
         self._bin_fr_M2[ind],
         self._bin_fr_N[ind]
        ) = utils.estimate_new_stats(
            spike_rate,
            self._bin_fr_means[ind],
            self._bin_fr_M2[ind],
            self._bin_fr_N[ind]
        )
        self._bin_fr_std[ind] = self._bin_fr_M2[ind] / self._bin_fr_N[ind]

    def _update_decode_stats(self, msg):

        ind = self._dec_ind

        if self.p_replay['method'] == 'map':
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

        ind = self._dec_ind

        arm_probs = self._compute_arm_probs(marginal_prob)
        self._arm_ps_buff[ind, self._dd_ind] = arm_probs

        ps_arm1, ps_arm2, ps_arm1_base, ps_arm2_base = self._compute_region_probs(
            marginal_prob
        )

        # add new posterior/likelihood probability sums for desired regions
        # perhaps we want to compute probability sum for box eventually?
        self._region_ps_buff[ind, self._dd_ind, 0] = 0
        self._region_ps_buff[ind, self._dd_ind, 1] = ps_arm1
        self._region_ps_buff[ind, self._dd_ind, 2] = ps_arm2

        self._region_ps_base_buff[ind, self._dd_ind, 0] = 0
        self._region_ps_base_buff[ind, self._dd_ind, 1] = ps_arm1_base
        self._region_ps_base_buff[ind, self._dd_ind, 2] = ps_arm2_base

    def _compute_arm_probs(self, prob):

        arm_probs = np.zeros(len(self.p['arm_coords']))
        for ii, (a, b) in enumerate(self.p['arm_coords']):
            arm_probs[ii] = prob[a:b+1].sum()

        return arm_probs

    def _compute_region_probs(self, prob):

        ps_arm1 = prob[20:25].sum()
        ps_arm2 = prob[36:41].sum()
        ps_arm1_base = prob[13:18].sum()
        ps_arm2_base = prob[29:34].sum()

        return ps_arm1, ps_arm2, ps_arm1_base, ps_arm2_base

    def _find_replay(self, msg):

        ts = msg[0]['bin_timestamp_r']
        ind = self._dec_ind

        if ts <= self._replay_event_ts + self._replay_event_ls:
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

        # assumes already satisfied event lockout criterion.
        # all these events should therefore be recorded

        print(f"Replay arm {arm} detected")

        self._replay_event_ts = msg[0]['timestamp']

        num_unique = np.count_nonzero(self._enc_ci_buff)
        print(f"Unique trodes: {num_unique}")

        send_shortcut = self._check_send_shortcut(
            self.p_replay['enabled'] and
            num_unique > self.p_replay['min_unique_trodes'] and
            arm == self.p_replay['target_arm'] and
            self._is_center_well_proximate
        )

        if send_shortcut:
            print(f"Replay arm {arm} rewarded")
            self._trodes_client.send_statescript_shortcut_message(14)

        ############################################################################################
        # write record
        ############################################################################################


    def _find_replay_instructive(self, msg):

        ts = msg[0]['bin_timestamp_r']
        ind = self._dec_ind

        if self._num_decoders == 2:
            raise NotImplementedError(
                "Finding instructive replay events is not implemented "
                "for 2 decoders"
            )
        else:
            # not out of lockout
            if ts <= self._replay_event_ts + self._replay_event_ls:
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

        # assumes already satisfied event lockout criterion.
        # all these events should therefore be recorded

        print(f"INSTRUCTIVE: Replay arm {arm} detected")

        self._replay_event_ts = msg[0]['timestamp']

        num_unique = np.count_nonzero(self._enc_ci_buff)
        print(f"INSTRUCTIVE: Unique trodes: {num_unique}")

        outer_arm_visited = utils.get_last_num(self.p['instructive_file'])

        send_shortcut = self._check_send_shortcut(
            self.p_replay['enabled'] and
            outer_arm_visited and
            num_unique > self.p_replay['min_unique_trodes'] and
            arm == self.p_replay['target_arm'] and
            self._is_center_well_proximate
        )

        if send_shortcut:
            print(f"INSTRUCTIVE: Replay target arm {arm} rewarded")
            utils.write_text_file(self.p['instructive_file'], 0)
            self._trodes_client.send_statescript_shortcut_message(14)
            self._instr_rewarded_arms[1:] = self._instr_rewarded_arms[:-1]
            self._instr_rewarded_arms[0] = arm
            self._choose_next_instructive_target()

        ##############################################################################################
        # write record
        ##############################################################################################

    def _choose_next_instructive_target(self):

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

        return self._task_state == 2 and other_condition

    def _init_stim_params(self):

        # Convention
        # ts - timestamp
        # ls - lockout samples

        self._replay_event_ts = 0
        self._replay_event_ls = int(
            self._config['sampling_rate']['spikes'] *
            self._config['stimulation']['replay']['event_lockout']
        )

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

        self._head_event_ts = 0
        self._head_event_ls = int(
            self._config['sampling_rate']['spikes'] *
            self._config['stimulation']['head_direction']['event_lockout']
        )

    def _init_data_buffers(self):

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

        # generate the lookup mapping
        self._decoder_rank_ind_map = {} # key: rank, value: index
        for ii, rank in enumerate(self._config['rank']['decoders']):
            self._decoder_rank_ind_map[rank] = ii

        N = self._config['stimulation']['replay']['sliding_window']
        num_trodes = self._config['decoder']['cred_int_bufsize']
        num_decoders = self._num_decoders

        self._dec_ind = 0 # decoder index of current posterior message
        self._dd_ind = 0 # shorthand for decode data

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

        self._spike_count = np.zeros(num_decoders)
        self._event_spike_count = np.zeros((num_decoders, N))

    def _init_params(self):

        self.p = {}
        self.p['taskstate_file'] = self._config['trodes']['taskstate_file']
        self.p['instructive_file'] = self._config['trodes']['instructive_file']
        self.p['scale_factor'] = self._config['kinematics']['scale_factor']
        self.p['instructive'] = self._config['stimulation']['instructive']
        # self.p['reward_mode'] = self._config['stimulation']['reward_mode']
        # self.p['shortcut_msg_on'] = self._config['stimulation']['shortcut_msg_on']
        self.p['num_pos_points'] = self._config['stimulation']['num_pos_points']
        self.p['num_pos_disp'] = self._config['display']['main']['position']
        self.p['max_center_well_dist'] = self._config['stimulation']['max_center_well_dist']
        self.p['arm_coords'] = self._config['encoder']['position']['arm_coords']

        # verify some inputs
        replay_method = self._config['stimulation']['replay']['method']
        if replay_method not in ('map', 'ml'):
            raise ValueError(
                f"Invalid method {replay_method} for replay"
            )

        ripple_method = self._config['stimulation']['ripples']['method']
        if ripple_method not in ('multichannel', 'consensus'):
            raise ValueError(
                f"Invalid method {ripple_method} for ripples"
            )

        self.p_replay = {}
        for k, v in self._config['stimulation']['replay'].items():
            self.p_replay[k] = v

        self.p_ripples = {}
        for k, v in self._config['stimulation']['ripples'].items():
            self.p_ripples[k] = v

        self.p_head = {}
        for k, v in self._config['stimulation']['head_direction'].items():
            self.p_head[k] = v