% PROGRAM NAME: 	Content Discrimination Task
% AUTHOR: Donghoon Shin -- shinapses@gmail.com
% DESCRIPTION: scatescript for content discrimination task with 2 arms & 3 arms (includes all pretraining & actual task protocol)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% PARAMETERS to change
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int TrialforDecision = 100 %(DS) from this trial, start turning on both outer arm LED. for visually guided (Decision = no), make it 100; for content discrimination with decision (Decision = yes), make it -1

int enable_decision_timeout = 1 % to have decision_timeout or not ; RENAMED choice_window -> enable_decision_timeout
int outer_arm_choice_duration = 15000 % (DS) window of outer arm reward / decision time out time interval ; RENAMED reward_window_outer_arm -> outer_arm_choice_duration

int accept_scm = 0 % whether i will use content shortcut message or not (0 if pretraining)


int TS2_durtaion = 3000000		% TS2 timelimit in ms; box time 2400000 = 40 min; 1800000 = 30min, 1200000 = 20min ; RENAMED content_trials_time_limit -> TS2_durtaion
int content_trials_limit = 80              % TS2 trial limit; if center port does not give reward all the time.


int beep_delay = 0                  % delay between detection of RR and sound cue in ms

int lockoutPeriod_timeout= 0 	    % Punishment; length of lockout after not going to the outer arm after a beep
int lockoutPeriod_centerReward = 0  %(DS) if the BEEP sound happens right after the rat started to consume reward, it won't react

%Reward pump duration in ms
int RewDelivery_dur_center_TS1= 300			% how long to deliver the reward at back/center, 100 uL ; RENAMED deliverPeriodBox -> RewDelivery_dur_center_TS1 (shortened from RewardDelivery_duration_center_TS1, >30 char statescript limit)
int RewDelivery_dur_center_TS2 = 300	% milk delivery time for content trials, 100 uL ; RENAMED deliverPeriodBox_content -> RewDelivery_dur_center_TS2 (shortened from RewardDelivery_duration_center_TS2, >30 char statescript limit)
int RewardDelivery_duration_Arms= 400   		% how long to deliver the reward at outer wells, 150 uL -- if 400; reduced ; RENAMED deliverPeriodOuter -> RewardDelivery_duration_Arms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5



% Hardware configuration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PORT (beambreak+LED) (where rats poke):
%   - Center port: 7 (where rats initiate trials)
%   - Back port: 5 (located behind the center port)
%   - Outer arm 1: 9
%   - Outer arm 2: 15
% PUMP ()
%
% Port initialization (turn on/off lights):
portout[7] = 1   % center / on
portout[5] = 0   % back / off
portout[9] = 0   % arm 1 / off
portout[15] = 0  % arm 2 / off
%
% Reward delivery variables initialization
int port_to_reward = 0  % RENAMED rewardWell -> port_to_reward ; will be set to 7 (center) / 5 (back) / 9 (arm1) / 15 (arm2)
int currPort = 0	   	% RENAMED currWell -> currPort
int lastPort = 0		% RENAMED lastWell -> lastPort
int dio = 0
int timer = 3000000				% first timer trial -timers ; RENAMED content_trial_time -> timer; 260


% Initialization of epoch parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% State variables (current task/trial state - changes every trial)
int taskstate = 1					% 1=cued trials (encoding phase), 2=decision trials (decoding phase), 3=termination
int trialPhase = 1              	% 1=wait_for_centerpoke, 2=after_centerpoke; RENAMED trialtype → trialPhase
int target_location = 0				% 1=arm1, 2=arm2, 3=arm3, 4=RemoteRepresentation



% Parameters for Taskstate 1: cued trials (encoding phase)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int correctCued = 1					% whether trial started correctly>?????





% Parameters for Taskstate 2: decision trials (decoding phase)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% number of trial (termination condition)
int content_trials_finish = 0		% content session time limit reached
%
% Counter
int contentReward = 0				% number of outer arm rewards
%
% function 21: reward delivered at center port
int centerRewardReady = 0			% reward ready at center port; RENAMED reward_available_at_center → centerRewardReady
int reward_avail = 0				% reward available window is open



% Trial initialization (state during current trial - resets each trial)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int reward_delivered = 0				% reward already given this trial
int decision_timeout = 0				% decision window timed out
int wrong_outer_visit = 0 				% rat poked wrong outer arm before beep
int reward_available_out_if_poke = 0 	% reward ready if rat pokes outer arm
int content_generation_avail = 0 		% ready to generate next content trial
int checking_decisiontimeout = 1    	% RENAMED ; outer_reward_window_done -> checking_decisiontimeout ; 1 = checking for decision timeout, 0 = not checking
%
% Counters (accumulated across session - increment)
int centerCount = 0					% number of times rewarded at center
int backCount = 0					% number of times rewarded at back
int currTrial_TS2 = 0				% RENAMED; contentTrialCount -> currTrial_TS2 ; number of content trials completed %%rethink about the name
int invalid_outer_arm_visits = 0			%RENAMED; number of outer arm visits during content ContentOuterCOunt --> invalid_outer_arm_visits
int num_arm1_trials = 0				% number of trials targeting arm 1
int num_arm2_trials = 0				% number of trials targeting arm 2
int goalTotal_TS1 = 0					% cumulative outer arm visits
int decision_timeout_number = 0		% number of decision timeouts
int otherCount = 0					% miscellaneous counter
int contentCorrectDecision = 0		% number of correct decisions during content trials
%
% Timing parameters (fixed durations)
int waittime = 0					% delay between center port poke and reward delivery (ms) -- no delay : 0
int target_replay_wait = 10			% delay for target replay
int reward_avail_wait = 3000		% reward window duration
%
% Other
int proximity = 0					% counter for repeated pokes at the same port (specific for center port????) 이거 왜필요함????
int waslock = 0						% flag for lockout state



% FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to deliver reward to box wells
% size of reward determined by taskstate
function 1
	if taskstate == 1 do
		portout[port_to_reward]=1 % reward
		do in RewDelivery_dur_center_TS1
			portout[port_to_reward]=0 % reset reward
		end

	else if taskstate == 2 do
		%disp('content reward at back')
		portout[port_to_reward]=1 % reward
		do in RewDelivery_dur_center_TS2
			portout[port_to_reward]=0 % reset reward
		end
	end
end;

% function to deliver reward to outer wells
function 2
	disp('outer reward')
	portout[port_to_reward]=1 % reward
	do in RewardDelivery_duration_Arms
		portout[port_to_reward]=0 % reset reward
	end
end;

% Function to turn on output
function 3
	portout[dio]=1
end;

% function to turn off output
function 4
	portout[dio]=0
end;

%display status to scatesscript terminal and saved in sc log
function 5
	disp(target_location)
	%disp(centerCount)
	disp(goalTotal_TS1)
	disp(currTrial_TS2)
	disp(contentCorrectDecision)
	disp(decision_timeout_number)
	disp(invalid_outer_arm_visits)
	disp(num_arm1_trials)
	disp(num_arm2_trials)
end;

% function running decision timeout
function 7
	if (enable_decision_timeout == 1) do
		checking_decisiontimeout = 0
		do in outer_arm_choice_duration
			%disp(outer_arm_choice_duration) %NOTE(DS): commented out in 2026-06
			if (reward_avail == 1 && centerRewardReady != 1 )  do
				disp('WHITENOISE')
				reward_avail = 0
				reward_delivered = 0
				centerRewardReady = 0
				portout[9] =0
				portout[15]=0
				portout[7] =1
				disp('DECISION TIMEOUT')
				decision_timeout = 1
				decision_timeout_number = decision_timeout_number + 1
				disp('GET TARGET')
			end
			checking_decisiontimeout = 1
		end
	end
end

%function starting the trial
function 8
	do in beep_delay
		currTrial_TS2 = currTrial_TS2 + 1
		lastPort = 7
		portout[7] = 0

		if (currTrial_TS2 >= TrialforDecision) do
			portout[15] = 1
			portout[9] = 1
		end


		if (target_location == 1) do
			portout[9]=1
			disp('start content trial of TARGET ARM1')
			num_arm1_trials = num_arm1_trials + 1
		else if (target_location == 2) do
			portout[15]=1
			disp('start content trial of TARGET ARM2')
			num_arm2_trials = num_arm2_trials + 1
		end
		disp('BEEP3')


		reward_avail = 1
		reward_delivered = 0
		content_generation_avail = 0
		contentReward = contentReward + 1
		reward_available_out_if_poke = 0
		trigger(7)
	end
end


% turning lights off for every port other than the center port -- so pretrial start
function 10
	if (taskstate == 2) do
		portout[9] = 0
		portout[15] = 0
		portout[4] = 0
		portout[7] = 1
	end
end

function 41
	disp(timer)
end

%% timer for time between trials - to start next trial text NEXT_TRIAL
%% first content trial: 60 sec
%% start timer for content session here too - moved to function 13
function 18
	do in lockoutPeriod_centerReward
		content_generation_avail = 1 % this variable is for content trials
		%disp(content_generation_avail)
	end
	disp('Timer starts')
	trigger(41)
	do in timer
		disp('Timer ends')
		trigger(41)
		if (wrong_outer_visit == 0 && centerRewardReady == 0 && decision_timeout ==0 && reward_available_out_if_poke == 0 && content_generation_avail == 1) do
			disp('2nd Centerpoke ready')
			reward_available_out_if_poke = 1
		end
	end
end;

% this is function for timer trials
% could make a function to run the timer for content wait
% at end of wait turn on back light and make beep
% cutoff was in trials before, now it is time
function 16
	%disp('start timer trial') %NOTE(DS): commented out in 2026-06
	trigger(8)
end;



%% shortcut message from decoder: arm 1 shortcut message
% new: only run if trialPhase = 1
function 14
	disp('short cut message at arm1')
	if (trialPhase == 1 && content_generation_avail == 1 && checking_decisiontimeout == 1 && accept_scm == 1) do
		if (target_location == 1 || target_location == 4) do %based on the target location
			target_location = 1
			trigger(8)
		end
	end
end;

function 6 % ARM 2 shortcut message
	disp('short cut message at arm2')
	if (trialPhase == 1 && content_generation_avail == 1 && checking_decisiontimeout == 1 && accept_scm == 1) do
		if (target_location == 2 || target_location == 4) do %based on the target location
			target_location = 2
			trigger(8)
		end
	end
end;

% first content trial
function 13
	disp('start the TS2 duration timer ')
	do in TS2_durtaion
		content_trials_finish = 1
		disp('TERMINATE EPOCH')
	end

	portout[7] = 1
	reward_delivered = 0
	centerRewardReady = 1
end;

function 21
	%disp('content trial - reward delivered')
	disp('BEEP4')
	centerRewardReady = 0
	reward_avail = 0
	disp('GET TARGET')
	trigger(18)
	if (contentReward == 0) do
		%disp('First trial') %NOTE(DS): commented out in 2026-06
		disp('DECODER_TASK2')
	end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% CALLBACKS -- EVENT-DRIVEN TRIGGERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
callback portin[7] up % center well
	currPort = 7 % well currently active
	disp('UP 7')

	% taskstate == 1
	if (taskstate == 1 && trialPhase == 1 && correctCued == 1) do
		if lastPort != currPort do
			%disp('start 1st cued trial') %NOTE(DS): commented out in 2026-06
			correctCued = 0
			proximity = 1
			do in waittime % currently waittime =0
				if proximity > 0 do
					proximity = 0
					trialPhase = 2
					disp('BEEP1') % after BEEP1, pythonObserver chooseGoal
					disp('BEEP2') % give reward at the center (beep_center)
				end
			end
		else do
			proximity=proximity+1
		end

	% taskstate == 2
	else if (taskstate == 2 && trialPhase == 1) do

		% coming back from outer arm after choice
		if (centerRewardReady == 1) do
			if (content_trials_finish == 1 || contentReward >= content_trials_limit) do
				taskstate = 3
				disp('TERMINATE EPOCH')
			else do
				trigger(21)
			end


		% other
		else if (centerRewardReady == 0) do
			% only make a sound when the rat pokes the center port
			if (reward_available_out_if_poke == 1 && content_generation_avail == 1) do
				disp('NEXT_TRIAL')
				reward_available_out_if_poke = 0

			% coming back to the center port after wrong outer visit
			else if (wrong_outer_visit == 1 ) do
				trigger(18)
				wrong_outer_visit = 0
				decision_timeout = 0 % if the rat barely missed the reward window, then there's no lockout


			% timeout -- if the rat does not do decision
			else if (decision_timeout == 1 ) do
				do in lockoutPeriod_timeout
					trigger(18)
				end
				decision_timeout = 0
				wrong_outer_visit = 0

			end
		end

	%else if (taskstate == 3) do
		%disp('TERMINATE EPOCH') %NOTE(DS): commented out in 2026-06

	end
end;

callback portin[7] down
	lastPort=7 % well left, now last well
	disp('DOWN 7')

end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% outer arm CALLBACKS
% center well: 7
% outer wells: 15,9; 3rd arm: 4
% back well: 5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% back -- this was initially 10 but changed to 5 (20240609-DS)
callback portin[5] up
	currPort = 5
	if currPort != lastPort do
		disp('UP 5')
	end
	% coming back from outer arm after choice
	%if (centerRewardReady == 1) do
	%	trigger(21)
	%end
end;

% back -- this was initially 10 but changed to 5 (20240609-DS)
callback portin[5] down
	lastPort = 5
	disp('DOWN 5')
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% arm 2 -- this was initially 13 but changed to 15 (20240609-DS)
callback portin[15] up
	currPort = 15
	if currPort != lastPort do
		disp('UP 15')

		if (taskstate == 2 && target_location == 2 && reward_avail == 1 && reward_delivered == 0) do
			disp('REWARD ARM2')
			%trigger(21)
		else if (taskstate == 2 && ((target_location == 3 || target_location == 1)) && reward_avail == 1 && reward_delivered == 0) do
			disp('WRONG ARM2')
			%trigger(21)
		else if (taskstate == 2 && (reward_avail == 0 || reward_delivered == 1)) do
			disp('INVALID OUTER ARM VISITS')
		end

	end



end;

callback portin[15] down
	lastPort = 15
	disp('DOWN 15')
	trigger(10)
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% arm 1
callback portin[9] up
	currPort = 9
	if currPort != lastPort do
		disp('UP 9')

		if (taskstate == 2 && target_location == 1 && reward_avail == 1 && reward_delivered == 0) do
			disp('REWARD ARM1')
			%trigger(21)
		else if (taskstate == 2 && (target_location == 2 || target_location == 3) && reward_avail == 1 && reward_delivered == 0) do
			disp('WRONG ARM1')
			%trigger(21)
		else if (taskstate == 2 && (reward_avail == 0 || reward_delivered == 1)) do
			disp('INVALID OUTER ARM VISITS')
		end

	end



end;

callback portin[9] down
	lastPort = 9
	disp('DOWN 9')
	trigger(10)
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% arm 3 -- center arm
callback portin[4] up
	currPort = 4
	if currPort != lastPort do
		disp('UP 4')

		if (taskstate == 2 && target_location == 3 && reward_avail == 1 && reward_delivered == 0) do
			disp('REWARD ARM3')
			%trigger(21)
		else if (taskstate == 2 && (target_location == 2 || target_location == 1) && reward_avail == 1 && reward_delivered == 0) do
			disp('WRONG ARM3')
			%trigger(21)
		else if (taskstate == 2 && (reward_avail == 0 || reward_delivered == 1)) do
			disp('INVALID OUTER ARM VISITS')
		end

	end



end;

callback portin[4] down
	lastPort = 4
	disp('DOWN 4')
	trigger(10)
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
