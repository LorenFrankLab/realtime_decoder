# Realtime Decoder

# Configuration

Please see example configuration file in ```example_config``` folder. Options are described in more detail below:

## ```rank```

Describes which MPI rank should be assigned to each process type.

For ```supervisor``` and ```gui```, only lists of length 1 are supported. For ```ripples```, ```encoders```, and ```decoders```, lists of arbitrary length are fine.

## ```rank_settings```

```enable_rec```: A list of ranks that will be writing their results to disk.

## ```trode_selection```

Here a trode refers to a group of electrodes of arbitrary number. It can thus be a single channel, a stereotrode, tetrode, etc.

```ripples```: A list of trodes that contain LFP data. Each trode is identified by its integer ID; string IDs are not supported.

```decoding```: A lit of trodes that contain spiking data.

## ```decoder_assignment```

Describes which trodes each decoder should handle.

The key is the decoder rank, and the item is a list of decoding tetrodes.

## ```algorithm```
Which algorithm the program should run. Currently, ```clusterless_decoder``` is supported.

## ```num_setup_messages```

How many messages the main/supervisor process should send to notify the GUI process that setup is complete. For some reason, these messages may be dropped (especially the first of these messages). A default of 100 seems to work well.

## ```preloaded_model```
If ```true``` pre-existing encoding models are used. If ```false```, encoding models are built on-the-fly as data are streamed.

## ```frozen_model```
Whether encoding models should be allowed to expand. If ```true```, new spikes *may* be added to the models. If ```false```, new spikes are *guaranteed not* to be added to the models.

## ```files```

```output_dir```: The directory in which all the output files should be stored.

```backup_dir```: Optional. If specified, the directory in which critical files should be copied to.

```prefix```: The file prefix for the output files

```rec_postfix```: The postfix for binary record files

```timing_postfix```: The postfix for timing files

```saved_model_dir```: The directory that contains the encoding model to be used. Only used if ```preloaded_model``` is ```true```.

## ```trodes```

Configuration options specific to the [Trodes](https://bitbucket.org/mkarlsso/trodes) data acquisition software. None of these are required if a different data source is used.

```config_file```: The path to the workspace file used for the streamed data.

```taskstate_file```: The path to the file that describes the current task state.

```instructive_file```: The path to the file used for an instructive task. Each line is a 1 or a 0. A 1 indicates an outer arm was visited. A 0 is used to mark separate trials.

```voltage_scaling_factor```: The factor used to convert Trodes raw data values to microvolts.

## ```sampling_rate```

```spikes```: The sampling rate of spike data, in Hz.

```lfp```: The sampling rate of LFP data, in Hz.

```position```: The sampling rate of position/camera data, in Hz.

## ```ripples```

### ```filter```

Parameters used for designing the filter that will filter LFP data to the ripple band. Both IIR and FIR filters are supported.

For an IIR filter:

```type```: Must be 'iir'.

```order```: Filter order.

```crit_freqs```: A list of length 2 marking the ripple band cutoff frequencies, in Hz.

```kwargs```: Keyword arguments passed directly to ```scipy```'s ```iirfilter()``` method

For a FIR filter:

```type```: Must be 'fir'.

```num_taps```: Number of taps used for the filter.

```band_edges```: A list of values marking the different bands, in Hz.

```desired```: The desired magnitude response.

### ```smoothing_filter```

Parameters used for designing the filter that will smooth ripple power data. Only FIR filters are supported.

Options are ```num_taps```, ```band_edges```, ```desired``` and have identical meaning to those described immediately above.

### ```threshold```

Units are number of standard deviations above the mean of the ripple envelope estimate. The ripple envelope is estimated as follows:

1. LFP data is filtered to the ripple band.
2. This ripple band data is squared and smoothed by the ```smoothing_filter```
3. The square root of the trace in (2) is taken.

```standard```: Threshold value for a "standard/regular" ripple.

```conditioning```: Threshold value for a conditioning ripple.

```content```: Threshold value for a content ripple.

```end```: Threshold value marking the end of a ripple.

IMPORTANT: Among ```standard```, ```conditioning```, and ```content``` ripples, the value of ```standard``` must be the lowest. This is because ```conditioning``` and ```content``` ripples are detected only if a ```standard``` ripple is active.

END OF ```threshold``` SECTION.

```max_ripple_samples```: Maximum number of samples that a ripple should be allowed to consist of. A value of 450 (300 ms with a 1500 Hz LFP sampling rate) is reasonable.

```vel_thresh```: Maximum velocity (in cm/s) allowed for a ripple detection.

```freeze_stats```: Whether to continuously update the mean and standard deviation of the ripple envelope estimate. If ```true```,continuous updating is disabled. This option may be toggled on and off by the GUI.



















































































































































































