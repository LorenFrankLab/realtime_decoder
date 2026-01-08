# Realtime Decoder

## System Requirements

This software was only tested on Linux machines. No guarantees are made for other operating systems.

For best results, you should have as many threads available as MPI processes you intend to run.

## Installation

Clone this repo and navigate to the repo root directory. Install from source with `pip install .`

For a developer installation, use `pip install -e .`

This package will be released to PyPI once it is finalized.

## Running

A sample script is provided in the repo root directory. To run the system, execute:

```
mpiexec -np <num_processes> -bind-to hwthread python -u runscript.py <path/to/config/file>
```

Note: `-bind-to hwthread` is optional but expected to give the best performance if enough threads are available.

# Configuration

Please see the example configuration file in the `example_config` folder. Options are described in more detail below.

## `rank`

Describes which MPI rank should be assigned to each process type.

For `supervisor` and `gui`, only lists of length 1 are supported. For `ripples`, `encoders`, and `decoders`, lists of arbitrary length are supported.

## `rank_settings`

`enable_rec`: A list of ranks that will write their results to disk.

## `trode_selection`

Here, a trode refers to a group of electrodes of arbitrary number. It can be a single channel, a stereotrode, a tetrode, etc.

`ripples`: A list of trodes that contain LFP data. Each trode is identified by its integer ID; string IDs are not supported.

`decoding`: A list of trodes that contain spiking data.

## `decoder_assignment`

Describes which trodes each decoder should handle.

The key is the decoder rank, and the value is a list of decoding tetrodes.

## `algorithm`
Which algorithm the program should run. Currently, `clusterless_decoder` is supported.

## `num_setup_messages`

How many messages the main/supervisor process should send to notify the GUI process that setup is complete. For some reason, these messages may be dropped (especially the first message). A value of 100 seems to work well. If there is never a popup saying all processes have completed setup, and you are sure there are no errors, increase this value until a popup appears.

## `preloaded_model`
If `true`, pre-existing encoding models are used. If `false`, encoding models are built on-the-fly as data are streamed.

## `frozen_model`
Whether encoding models should be allowed to expand. If `false`, new spikes *may* be added to the models. If `true`, new spikes are *guaranteed not* to be added to the models. This option may be toggled on and off by the GUI.

## `files`

`output_dir`: The directory in which all output files should be stored.

`backup_dir`: Optional. If specified, the directory to which critical files should be copied.

`prefix`: The file prefix for output files.

`rec_postfix`: The postfix for binary record files.

`timing_postfix`: The postfix for timing files.

`saved_model_dir`: The directory that contains the encoding model to be used. Only used if `preloaded_model` is `true`.

## `trodes`

Configuration options specific to the [Trodes](https://bitbucket.org/mkarlsso/trodes) data acquisition software. None of these are required if a different data source is used.

`config_file`: The path to the workspace file used for the streamed data.

`taskstate_file`: The path to the file that describes the current task state.

`instructive_file`: The path to the file used for an instructive task. Each line is a 1 or a 0. A 1 indicates an outer arm was visited. A 0 is used to mark separate trials.

`voltage_scaling_factor`: The factor used to convert Trodes raw data values to microvolts.

## `sampling_rate`

`spikes`: The sampling rate of spike data, in Hz.

`lfp`: The sampling rate of LFP data, in Hz.

`position`: The sampling rate of position/camera data, in Hz.

## `ripples`

`max_ripple_samples`: Maximum number of samples that a ripple should be allowed to consist of. For example, a value of 450 with an LFP sampling rate of 1500 Hz is 300 ms, a reasonable duration.

`vel_thresh`: Maximum velocity allowed for ripple detection. Units are cm/s.

`freeze_stats`: Whether to continuously update the mean and standard deviation of the ripple envelope estimate. If `true`, continuous updating is disabled. This option may be toggled on and off by the GUI.

### `filter`

Parameters used for designing the filter that will filter LFP data to the ripple band. Both IIR and FIR filters are supported.

For an IIR filter:

`type`: Must be 'iir'.

`order`: Filter order.

`crit_freqs`: A list of length 2 marking the ripple band cutoff frequencies, in Hz.

`kwargs`: Keyword arguments passed directly to `scipy`'s `iirfilter()` method.

For a FIR filter:

`type`: Must be 'fir'.

`num_taps`: Number of taps used for the filter.

`band_edges`: A list of values marking the different bands, in Hz.

`desired`: The desired magnitude response.

### `smoothing_filter`

Parameters used for designing the filter that will smooth ripple power data. Only FIR filters are supported.

Options are `num_taps`, `band_edges`, and `desired`, which have identical meanings to those described immediately above.

### `threshold`

Units are number of standard deviations above the mean of the ripple envelope estimate. The ripple envelope is estimated as follows:

1. LFP data is filtered to the ripple band.
2. This ripple band data is squared and smoothed by the `smoothing_filter`.
3. The square root of the trace in (2) is taken.

`standard`: Threshold value for a "standard/regular" ripple.

`conditioning`: Threshold value for a conditioning ripple.

`content`: Threshold value for a content ripple.

`end`: Threshold value marking the end of a ripple.

IMPORTANT: Among `standard`, `conditioning`, and `content` ripples, the value of `standard` must be the lowest. This is because `conditioning` and `content` ripples are detected only if a `standard` ripple is active.

## `custom_mean`

Note: This entire section is optional. If specified, the options are used to initialize the mean of the ripple band envelope estimate to a custom value.

The key is the trode numeric ID, and the value is the custom mean.

A special case is the custom mean for the consensus trace. Here the key is `consensus`.

## `custom_std`

This entire section is also optional and similar to the `custom_mean` section above, except the custom values are used to initialize the standard deviation of the ripple band envelope estimate.

## `encoder`

`spk_amp`: For each spike, the global spike amplitude (i.e., over all channels in a trode) must exceed this value to be considered for decoding.

`mark_dim`: Number of dimensions each mark vector has.

`bufsize`: Initial number of elements each encoding model array can hold. During runtime, this array may be expanded so that no data are dropped.

`timings_bufsize`: Similar to `bufsize` above, except this is the initial number of elements the timings array can hold.

`vel_thresh`: Minimum speed the animal must be moving at for a spike to be added to an encoding model. Units are cm/s.

`num_pos_points`: Read the `taskstate_file`, if specified, every `num_pos_points` position data points. For example, if the `position` sampling rate is 30 Hz and this value is 15, then the `taskstate_file` would be read every 0.5 seconds.

### `position`

`lower`: The lower edge value of the position bins. Currently this value must be 0.

`upper`: The upper edge value of the position bins.

`num_bins`: How many position bins there are.

`arm_ids`: Every position data point the encoder process receives contains which line segment the data point came from (i.e., raw positions are linearized onto user-defined line segments). `arm_ids` is an array that maps each line segment to a particular arm of the maze. The size of the array is expected to match the number of line segments.

`arm_coords`: An array of arrays. Each sub-array contains the lower and upper position bins (inclusive) that make up a particular maze arm.

### `mark_kernel`

`mean`: Not currently used.

`std`: The standard deviation of the Gaussian kernel evaluated on the distance between a candidate spike and the other spikes in the encoding model.

`use_filter`: If `true`, a `mark_dim` n-cube is drawn around a candidate mark. If there are enough marks in the n-cube, the mark-position joint probability estimate is computed. Otherwise, the spike is not decoded. If `use_filter` is `false`, every spike is decoded.

`n_std`: Only used if `use_filter` is `true`. For each dimension x of a candidate spike mark, the search area consists of mark_<dimension_x> +/- `n_std` * `std`.

`n_marks_min`: Only used if `use_filter` is `true`. Minimum number of marks that must be in the n-cube surrounding a candidate spike mark to be considered for decoding.

### `dead_channels`

This section is optional.

The key is the trode numeric ID, and the value is an array of channels that should be considered dead.

IMPORTANT: The channels are zero-indexed!

## `decoder`

`bufsize`: Number of spike messages (which are sent by encoder processes) the internal circular buffer can hold.

`timings_bufsize`: Initial number of elements the timings array can hold. During runtime, this array may be expanded so that no data are dropped.

`cred_int_bufsize`: Maximum number of credible interval values (determined from spike/position joint probability estimates) to send.

`num_pos_points`: Read the `taskstate_file`, if specified, every `num_pos_points` position data points. For example, if the `position` sampling rate is 30 Hz and this value is 15, then the `taskstate_file` would be read every 0.5 seconds.

### `time_bin`

For both options below, the reference sampling rate is the `spikes` sampling rate. We also assume LFP timestamps use this reference sampling rate.

`samples`: How many samples make up a time bin. For example, if the `spikes` sampling rate is 30 kHz and `samples` is 180, the time bin size is 6 ms.

`delay_samples`: How many samples behind the current LFP timestamp the right edge of the current time bin should be. For example, if the `spikes` sampling rate is 30 kHz and `delay_samples` is 90, the right edge of the current time bin will be 3 ms behind the current LFP timestamp.

## `clusterless_decoder`

Options used if `algorithm` is "clusterless_decoder".

`state_labels`: An array containing labels for each state.

`transmat_bias`: The bias/offset that should be added to the transition matrix.

## `gui`

`colormap`: The name of the colormap to be used for the likelihood and posterior plots. It must be a valid matplotlib or seaborn colormap/color palette.

`send_interval`: Determines how often parameters are automatically sent to other processes. Set to <= 0 to disable. Units are in seconds.

`refresh_rate`: How often to update the plots. Units are in Hz.

`trace_length`: Length of time to show on the plots. Units are in seconds.

`state_colors`: Array of hex colors to be used for each decoding state. These colors are only relevant to the state probability plot.

`num_xticks`: Number of x-axis ticks to show on the plots.

## `stimulation`

Note: These parameters are specific to whatever custom stimulation decider you are using. Different stim deciders may use different options, and it is up to the user to handle this.

The documentation below is relevant to the TwoArmTrodesStimDecider, a ready-to-use class implemented in stimulation.py.

`instructive`: Whether the data is coming from an instructive task.

`num_pos_points`: Read the `taskstate_file`, if specified, every `num_pos_points` position data points. For example, if the `position` sampling rate is 30 Hz and this value is 15, then the `taskstate_file` would be read every 0.5 seconds.

`center_well_location`: An array consisting of the center well location, in pixels.

`max_center_well_dist`: Used for determining candidate replay and head direction events. For every candidate event, the current distance of the animal from the center well cannot exceed this value. Units are in cm.

### `replay`

Parameters relevant to replay detection.

`enabled`: Whether a statescript message may be sent upon replay detection.

`method`: The probability distribution to use for detecting replay. Must be "posterior" or "likelihood".

`target_arm`: Replay target arm. This value is automatically determined for an instructive task. Otherwise, it may be modified using the GUI.

`event_lockout`: Minimum time imposed between successive candidate replay events. Units are in seconds.

`sliding_window`: Number of time bins to use for replay detection. This value is directly proportional to the length of each time bin.

`primary_arm_threshold`: Minimum average probability sum value required to be considered a replay event for a particular arm. The average is taken over the sliding window.

`secondary_arm_threshold`: Same as above. Only used if there are two decoder processes.

`other_arm_threshold`: Maximum average probability sum of the non-replay arms to be considered a replay event for a particular arm. The average is taken over the sliding window.

`max_arm_repeats`: Not currently used.

`instr_max_repeats`: Maximum number of consecutive replay target arms. This value is only used for an instructive task.

`min_unique_trodes`: A candidate replay event must have at least this number of trodes active. The observation period is the entire sliding window.

## `ripples`

`enabled`: Whether a statescript message may be sent upon global ripple detection.

`type`: Which type of ripple may be used to trigger a statescript message (if enabled). Must be "standard", "cond", or "content".

`method`: Which ripple algorithm type may be used to trigger a statescript message (if enabled). Must be "multichannel" or "consensus".

`event_lockout`: Minimum time imposed between successive candidate ripple events. Units are in seconds.

`num_above_thresh`: Number of trodes that must be in a ripple to trigger event detection.

## `head_direction`

`enabled`: Whether a statescript message may be sent upon head direction detection.

`rotate_180`: Whether to rotate the head direction vector by 180 degrees. This is important because the angle of this vector is used to detect candidate head direction events.

`event_lockout`: Minimum time imposed between successive candidate head direction events. Units are in seconds.

`min_duration`: Determines the observation period for detecting candidate head direction events. It is analogous to the sliding window for replay detection. Units are in seconds.

`well_angle_range`: For every candidate head direction event, the head direction vector must be within the angle to a reward well +/- this value. Units are in degrees.

`within_angle_range`: For every candidate head direction event, the range R of head direction vector angles is taken over the observation period determined by `min_duration`. R must not exceed the value of `within_angle_range`. Units are in degrees.

`wells_locs`: An array of x and y locations for each reward well. Units are in pixels.

## `kinematics`

`smooth_x`: Whether to smooth the x position.

`smooth_y`: Whether to smooth the y position.

`smooth_speed`: Whether to smooth the speed.

`smoothing_filter`: An array containing the smoothing filter coefficients.

`scale_factor`: The factor used to convert from raw data values to cm.

## `cred_interval`

`val`: Probability used to determine the credible interval.

`max_num`: Maximum value of the credible interval to be considered a high-quality spike.

## `display`

### `stim_decider`

`position`: Print out kinematics and head direction angles every `position` number of received position points.

`decoding_bins`: Print out firing rate statistics every `decoding_bins` received decoding bin data points.

### `ripples`

`lfp`: Print out cumulative number of LFP data points received every `lfp` number of LFP data points received.

### `encoder`

`encoding_spikes`: Print number of spikes in the encoding model every `encoding_spikes` number of encoding spikes.

`total_spikes`: Print total number of spikes received every `total_spikes` number of total spikes.

`occupancy`: Print out occupancy information every `occupancy` number of points added to the occupancy array.

`position`: Print out cumulative number of position data points received every `position` number of position data points received.

### `decoder`

`total_spikes`: Print out cumulative number of spikes received from encoding processes every `total_spikes` number of spikes received.

`occupancy`: Print out occupancy information every `occupancy` number of points
