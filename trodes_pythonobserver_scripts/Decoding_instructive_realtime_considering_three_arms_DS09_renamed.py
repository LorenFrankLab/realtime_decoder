import math
import struct
import re
import time
import random
import numpy as np
import pyaudio
import wave
from itertools import compress
import random

# V8pre_forage
# visits to incorrect wells cause 5s lockout
# exception is repeat visit to prior well (is ok, no lockout)
# can go to any outer well, any number of times
# lockout from getting rip/wait wells wrong is also 5s


# decide what type of up trigger was just received; act accordingly
# only for back and outer, center well defined in statescript
def pokeIn(dio):
	global backWell
	global centerWell
	global outerWells
	global currWell
	global taskState
	global start_session
 
	currWell = int(dio[1])
	# not using doback
	#if currWell == backWell: 
	#	doback()

	# start taskstate1 with first poke - now only do with first poke at center
	if currWell == centerWell and start_session == 0:
		print('write taskstate1')
		with open("/home/lorenlab/realtime_decoder/config/taskstate.txt","a") as reward_arm_file:
			try:
				reward_arm_file.write(str(1)+'\n')
			finally: 
				reward_arm_file.close()
		start_session += 1

	# how do we start???
	# for testing we could start with an outer well visit

	if taskState == 1:
		if currWell == backWell:
			doback()
		else:
			for num in range(len(outerWells)):
				if currWell == outerWells[num]:
					doOuter(num)
	elif taskState == 4:
		pass
	# taskstate 2 or 3	
	else:
		#print(currWell)
		for num in range(len(outerWells)):
			if currWell == outerWells[num]:
				doOuter(num)

# decide what type of down trigger was just recieved; act accordingly
def pokeOut(dio):
	global backWell
	global centerWell
	global outerWells
	global currWell
	global lastWell
	global taskState
 
	currWell = int(dio[1])
	# not using endback
	if currWell == centerWell: 
		endCenter()

	if taskState == 1:
		if currWell == backWell:
			endback()
		else: # if current well is center outer arm (and center arm)
			for num in range(len(outerWells)):
				if currWell == outerWells[num]:
					endOuter()
	else: # if taskState == 2 or 3
		for num in range(len(outerWells)):
			if currWell == outerWells[num]:
				endOuter()
		lastWell = currWell

# NOTE: currently NOT calling this function
# instead use endOuter to start new trial
#back poke: decide trial type and upcoming wait length; turn on lights accordingly
# def doback():
# 	global trialtype  # 0 go to back,1 go to center, 2 go to outer, 3 lockout
# 	global backPump
# 	global lastWell
# 	global currWell
# 	global goalWell
# 	global outerWells

# 	if trialtype == 0:	
# 		trialtype = 1
# 		print("SCQTMESSAGE: trialPhase = "+str(trialtype)+";\n")
# 		# this line set the variable delaytime and then writes the variable to statescript  
# 		delaytime = chooseDelay()
# 		print("SCQTMESSAGE: waittime = "+str(delaytime)+";\n")

# 		# set goal well to 1 arm for this trial - get line from old script
# 		# set goal well based on replay content from spykshrk after enough visits to each arm

# 		# yes this is wrong for content rewared wells, but content information will reset
# 		# goalWell in beep function
# 		# this is where cued trials are set and we want to control disrubiton 
# 		#goalWell = np.random.choice(outerWells,1,replace=False)
# 		print("SCQTMESSAGE: backCount = backCount + 1;\n") # update backcount in SC
# 		print("SCQTMESSAGE: port_to_reward = "+str(backPump)+";\n")
# 		print("SCQTMESSAGE: trigger(1);\n")   # deliver reward
# 	#check for back poke out of sequence, start lockout 1
# 	elif trialtype > 0 and trialtype < 3 and lastWell != currWell:
# 		lockout([0,1])

def chooseDelay():
    global trialtype
    global centercount
    global waitdist
    global startwaitdist

    #print(centercount)

    if centercount<3:  #first 3 trials of of each type should be short
        return startwaitdist[centercount]

    else:
        if centercount<=10:  #trials 4-10 of each type will be avg of startwaitdist and normal waitdist
            return int(round(np.mean([int(np.random.choice(startwaitdist,1)), int(np.random.choice(waitdist,1))])))

        else: # all trial 10 and later
            return int(np.random.choice(waitdist,1))

#NOTE: currently NOT calling this function
# was endback
def endCenter():
	global cuedWells
	global taskState
	global outer_count_content

	# for first content trial, trigger 13
	if taskState == 2 and outer_count_content == 0:
		outer_count_content += 1
		for num in range(len(outerWells)):			# turn off outer lights
			print("SCQTMESSAGE: dio = "+str(outerWells[num])+";\n")
			print("SCQTMESSAGE: trigger(4);\n")
		print('first content trial',outer_count_content)
		print("SCQTMESSAGE: trigger(5);\n")   # display stats
		print("SCQTMESSAGE: trigger(13);\n")

#function: add time to wait dist. 
def addtime(newtime):
	global count
	global waitdist

	count+=1
	# % means mod (reaminder) function
	waitdist[count%8] = int(newtime[1])  #new time 1 will be timediff, not timestamp
	print(waitdist)

# only called during cued outer arm trials
# this function delivers reward at the center
def beep_center():

	global centerPump
	global centerWell
	global trialtype
	global currWell
	global centercount
	global taskState
	global center_reward_counter
	global half_reward_at_center
	global reward_center_during_TS2
	global correct_trial_bit
	global give_only_at_the_correct_bit
	
	centercount+=1

	# for taskstate 2


	# for taskstate 1 and 3
	if taskState != 2:
		# begin outer arm trial section of task
		trialtype = 2                   # ready for outer visit
		print("SCQTMESSAGE: trialPhase = "+str(trialtype)+";\n")
		#deliver reward
	if currWell == centerWell:
		print("SCQTMESSAGE: port_to_reward = "+str(centerPump)+";\n")
	if taskState == 1:
		if half_reward_at_center:
			if center_reward_counter%2 == 0:
				print("SCQTMESSAGE: trigger(1);\n")
			center_reward_counter += 1
		else: 
			print("SCQTMESSAGE: trigger(1);\n")
	
	if reward_center_during_TS2 and give_only_at_the_correct_bit and correct_trial_bit:
		if taskState == 2:
			print("SCQTMESSAGE: trigger(1);\n")
	
	if taskState == 2:	
		if reward_center_during_TS2:
			if (give_only_at_the_correct_bit):
				if correct_trial_bit:
					print("SCQTMESSAGE: trigger(1);\n")
			else:
				print("SCQTMESSAGE: trigger(1);\n")
	
	print("SCQTMESSAGE: centerCount = centerCount + 1;\n") # update centercount in SC

def reward_delivery_at_arm1():

	''' delivering reward at arm 1 during task state 2 -- decision making'''
	print("SCQTMESSAGE: port_to_reward = "+str(outerPumps[1])+";\n")
	print("SCQTMESSAGE: trigger(2);\n")

def reward_delivery_at_arm2():

	''' delivering reward at arm 2 during task state 2 -- decision making'''
	print("SCQTMESSAGE: port_to_reward = "+str(outerPumps[0])+";\n")
	print("SCQTMESSAGE: trigger(2);\n")

def reward_delivery_at_arm3():

	''' delivering reward at arm 3 during task state 2 -- decision making'''
	print("SCQTMESSAGE: port_to_reward = "+str(outerPumps[2])+";\n")
	print("SCQTMESSAGE: trigger(2);\n")

# # only called during content trials
# def beep_back():
# 	global backPump
# 	global backWell
# 	global trialtype
# 	global currWell
# 	global backcount

# 	backcount+=1
# 	## begin outer arm trial section of task
# 	#trialtype = 2                   # ready for outer visit
# 	#print("SCQTMESSAGE: trialPhase = "+str(trialtype)+";\n")
# 	#deliver reward
# 	if currWell == backWell:
# 		print("SCQTMESSAGE: port_to_reward = "+str(backPump)+";\n")
# 	print("SCQTMESSAGE: trigger(1);\n")
# 	print("SCQTMESSAGE: backCount = backCount + 1;\n") # update centercount in SC

# define goalWell - only used during cued arm trials
def chooseGoal():
	global taskState
	global replay_arm
	global outerarm_required_rewards
	global arm1_Goal
	global arm2_Goal
	global arm3_Goal
	global arm4_Goal
	global back_Goal
	global goalWell
	global outerWells
	global backWell
	global centerWell
	global cuedWells
	global cued_trial_counter
	global oldGoal1
	global oldGoal2
	global outer_arm_reward

	global arm1_counter
	global arm2_counter
	global arm1_order8
	global arm2_order8
	global ts1_reward_prob
	global timer_mean
	# taskstate ==1 is cued visits to each outer arm
	if taskState == 1:

		# trial 0: set reward order for each arm
		if cued_trial_counter == 0: # (DS) reward probability during arm visits
			# 2 of 4 arm visits rewarded - ranint(6)
			print("SCQTMESSAGE: timer = "+str(timer_mean)+";\n")
			
			if ts1_reward_prob == 50:
				order_options = np.array([[1,1,0,0],[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1],[0,0,1,1],
							  [0,1,1,0],[0,1,0,1],[1,0,0,1],[0,0,1,1],[1,1,0,0],[1,0,1,0]])# 50% rewarded
			elif ts1_reward_prob == 75:
				order_options = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0],[0,1,1,1],[1,0,1,1],
							   [0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0],[0,1,1,1],[1,0,1,1]])# 75% rewarded
			else:
				# all rewarded
				order_options = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],
							   [1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])

			arm1_order = order_options[np.random.randint(12)]
			arm1_order8 = np.append(arm1_order,order_options[np.random.randint(12)])
			arm1_order8 = np.append(arm1_order8,order_options[np.random.randint(12)])
			arm1_order8 = np.append(arm1_order8,order_options[np.random.randint(12)])
			
			
			arm2_order = order_options[np.random.randint(12)]
			arm2_order8 = np.append(arm2_order,order_options[np.random.randint(12)])
			arm2_order8 = np.append(arm2_order8,order_options[np.random.randint(12)])
			arm2_order8 = np.append(arm2_order8,order_options[np.random.randint(12)])
			
			
			print("SCQTMESSAGE: disp('Cued arm 1 reward order "+str(arm1_order8)+"');\n")
			print("SCQTMESSAGE: disp('Cued arm 2 reward order "+str(arm2_order8)+"');\n")

		# variable for <100% reward
		#outer_arm_reward = np.random.randint(100)<75

		# want to force alternation between arms in the beginning

		# this is boolean way of setting this to 1 or 0
		# valid+_goals is not global just local to this funciton
		#valid_goals = [arm1_Goal<outerarm_required_rewards,
		#			   arm2_Goal<outerarm_required_rewards,
		#			   arm3_Goal<outerarm_required_rewards,
		#			   arm4_Goal<outerarm_required_rewards]
		valid_goals = [arm1_Goal<outerarm_required_rewards,
					   arm2_Goal<outerarm_required_rewards]					   
		print('valid goals',valid_goals)

		# want to force alternation between arms in the beginning (first 2 visits each)
		# okay this seems to work
		#if (cued_trial_counter > 0 and arm1_Goal < outerarm_required_rewards-1 and arm2_Goal < outerarm_required_rewards-1
		#	and arm3_Goal < outerarm_required_rewards-1 and arm4_Goal < outerarm_required_rewards-1):
		#if (cued_trial_counter > 0 and arm1_Goal < outerarm_required_rewards-1
		#	and arm2_Goal < outerarm_required_rewards-1):

		#	print('goal',goalWell)
		#	print('old goal',oldGoal)
		#	print(outerWells.index(oldGoal))
		#	valid_goals[outerWells.index(oldGoal)] = False
		#	print('valid goals no repeat',valid_goals)

		# now want to force alternation only if there have been 2 visits to the same arm
		if (cued_trial_counter > 0 and arm1_Goal < outerarm_required_rewards
			and arm2_Goal < outerarm_required_rewards):

			print('old goal 1',oldGoal1,'old goal 2',oldGoal2)
			if oldGoal1 == oldGoal2:
				#print(outerWells.index(oldGoal))
				print('2 same goals in a row, force alternation')
				valid_goals[outerWells.index(oldGoal1)] = False
				print('valid goals no repeat',valid_goals)		

		# this line doesnt work
		# try this:
		print(list(compress(outerWells,valid_goals)))

		# now only choose from list of outerwells where valid_goals == 1
		goalWell = np.random.choice(list(compress(outerWells,valid_goals)),1,replace=False)
		oldGoal2 = oldGoal1
		oldGoal1 = goalWell
		print('cued goalWell is: ',goalWell)
		print("SCQTMESSAGE: disp('CUED ARM VISITS "+str(outerarm_required_rewards)+"');\n")

		# we want every 6th trial to be to back well
		if cued_trial_counter % 6 == 0 and cued_trial_counter > 1:
			goalWell = [backWell]
			print('back goal')
			#oldGoal = goalWell

		# NOTE: need to substitute the correct arm numbers (not 1-4) for goalWell
		# NOTE: check that arm assignments are correct
		# NOTE: set outer_arm_reward = 1 to have 100% reward
		if goalWell == outerWells[0]:
			outer_arm_reward = arm1_order8[arm1_counter]
			arm1_counter += 1
		elif goalWell == outerWells[1]:
			outer_arm_reward = arm2_order8[arm2_counter]
			arm2_counter += 1	
					
		cued_trial_counter += 1

	# for return to cued visits used 2*outerarm_required_rewards as cutoff
	elif taskState == 3:

		# variable for <100% reward
		outer_arm_reward = np.random.randint(100)<75

		# this is boolean way of setting this to 1 or 0
		# valid+_goals is not global just local to this funciton
		#valid_goals = [arm1_Goal<outerarm_required_rewards+2,
		#			   arm2_Goal<outerarm_required_rewards+2,
		#			   arm3_Goal<outerarm_required_rewards+2,
		#			   arm4_Goal<outerarm_required_rewards+2]
		valid_goals = [arm1_Goal<outerarm_required_rewards+2,
					   arm2_Goal<outerarm_required_rewards+2]					   
		print(valid_goals)

		# now want to force alternation only if there have been 2 visits to the same arm
		if (cued_trial_counter > 0 and arm1_Goal < outerarm_required_rewards+6
			and arm2_Goal < outerarm_required_rewards+6):

			print('old goal 1',oldGoal1,'old goal 2',oldGoal2)
			if oldGoal1 == oldGoal2:
				#print(outerWells.index(oldGoal))
				print('2 same goals in a row, force alternation')
				valid_goals[outerWells.index(oldGoal1)] = False
				print('valid goals no repeat',valid_goals)	

		# this line doesnt work
		#print(outerWells[valid_goals])
		# try this:
		print(list(compress(outerWells,valid_goals)))

		# now only choose from list of outerwells where valid_goals == 1
		goalWell = np.random.choice(list(compress(outerWells,valid_goals)),1,replace=False)
		oldGoal2 = oldGoal1
		oldGoal1 = goalWell		
		print('cued goalWell is: ',goalWell)
		print("SCQTMESSAGE: disp('CUED ARM VISITS "+str(outerarm_required_rewards)+"');\n")

		#goalWell = np.random.choice(outerWells[valid_goals],1,replace=False)


	# if taskState == 2, content trials
	# define goalWell as outerwell matching replay arm
	# replay_arm - 1 to make 0 based becuase it is used an an index into outerWells
	# if replay arm not updated will index to -1 and error out
	# NEW: no goalwell if content trials
	else: 
		#goalWell = [outerWells[replay_arm-1]]
		#print('content goalWell is: ',goalWell)
		#print('replay arm is: ',replay_arm)
		print('content trial - no outer arm goals')
		outer_arm_reward = 1

# only called during cued outer arm trials
def endWait():
	global trialtype
	global goalWell
	global currWell
	global outerWells
	global arm1_Goal
	global arm2_Goal
	global arm3_Goal
	global arm4_Goal

	if trialtype == 2:   # wait complete

		print("SCQTMESSAGE: dio = "+str(currWell)+";\n")     # turn off rip light
		print("SCQTMESSAGE: trigger(4);\n")	

		print("SCQTMESSAGE: trigger(5);\n")   # display stats
		print("SCQTMESSAGE: disp('CURRENTGOAL IS "+str(goalWell[0])+" TASK_STATE IS "+str(taskState)+"');\n") 
		
		# use taskState to determine whether this is a cued trial or content trial
		# only for which outer lights to turn on
		if taskState == 1:
			# turn on light for goalWell only
			# brackets required to get integer only for statescript
			print("SCQTMESSAGE: dio = "+str(goalWell[0])+";\n")
			print("SCQTMESSAGE: trigger(3);\n")

		elif taskState == 3:
			# turn on light for goalWell only
			# brackets required to get integer only for statescript
			print("SCQTMESSAGE: dio = "+str(goalWell[0])+";\n")
			print("SCQTMESSAGE: trigger(3);\n")


		# taskState = 2
		# this may or may not happen...
		#else:
			# NEW: no outer arms - so re-start wait here
			# we want this wait to have no light, turn on light after wait ends
			# function 16 turns on light and makes beep after wait
			
			#trialtype = 1
			#print('first content trial, in endWait')
			#print("SCQTMESSAGE: trigger(16);\n")

			# the statescript function for each arm now produces the beep after recieving the REPLAY_ARM message
			# so goalWell should be updated based on the message from spykshrk not the random number
			## turn on all outer lights
			#for num in range(len(outerWells)):
			#	print("SCQTMESSAGE: dio = "+str(outerWells[num])+";\n")
			#	print("SCQTMESSAGE: trigger(3);\n")
			#	print('current goal is ',str(goalWell[0]))

			#print("SCQTMESSAGE: dio = "+str(centerWell)+";\n")   # turn center light on
			#print("SCQTMESSAGE: trigger(3);\n")
			#print("SCQTMESSAGE: trigger(5);\n")   # display stats


		# save out current goal well to the text file
		#with open("/home/lorenlab/spykshrk_realtime/config/rewarded_arm_trodes.txt","a") as reward_arm_file:
		#		try:
	#				reward_arm_file.write(str(currWell-9)+' '+str(taskState)+' '+str(goalWell[0]-9)+'\n')
		#		finally: 
		#			reward_arm_file.close()


def doback():
	global backPump
	global trialtype
	global allGoal
	global goalWell 
	global currWell
	global lastWell
	global backWell
	global taskState

	if trialtype == 2:
		if currWell in goalWell:
			trialtype = 1      # outer satisfied, old: head back next (0). new: head to center (1)

		print("SCQTMESSAGE: trialPhase = "+str(trialtype)+";\n")
		print('current well',currWell)
		print('goal well',goalWell)
		if currWell in goalWell and taskState == 1:  # repeated; reward
			print("SCQTMESSAGE: port_to_reward = "+str(backPump)+";\n")
			print("SCQTMESSAGE: trigger(2);\n")
			("SCQTMESSAGE: back reward delivered;\n")
			print("SCQTMESSAGE: correctCued = "+str(1)+";\n")


# called by any visit to outer arm
def doOuter(val):
	global outerPumps
	global cuedPumps
	global trialtype
	global allGoal
	global goalWell 
	global currWell
	global lastWell
	global backWell
	global waslock
	global arm1_Goal
	global arm2_Goal
	global arm3_Goal
	global arm4_Goal
	global back_Goal
	global taskState
	global outerarm_required_rewards
	global outer_arm_reward

	if taskState == 2:
		print('outer visit during content trials')

	if trialtype == 2:
		if currWell in goalWell:
			trialtype = 1      # outer satisfied, old: head back next (0). new: head to center (1)

		print("SCQTMESSAGE: trialPhase = "+str(trialtype)+";\n")
		print('current well',currWell)
		print('goal well',goalWell)
		print('value',val)
		if currWell in goalWell :  # repeated; reward

			if taskState in [1, 3]:
			    print("SCQTMESSAGE: port_to_reward = " + str(outerPumps[val]) + ";\n")

			# only deliver reward if this is one of the rewarded visits
			if outer_arm_reward:
				print("SCQTMESSAGE: trigger(2);\n")   # deliver reward
				print("outer reward delivered;")
			else:
				print("no outer reward;")
			

			print("SCQTMESSAGE: correctCued = "+str(1)+";\n")
			allGoal+=1
			# create and add to counter for each of the 4 outer arms
			if currWell == outerWells[0]:
				arm1_Goal+=1
				#print('arm1',arm1_Goal)
			elif currWell == outerWells[1]:
				arm2_Goal+=1
				#print('arm2',arm2_Goal)

			elif currWell == outerWells[2]:
				arm3_Goal+=1
				#print('arm3',arm3_Goal)
			elif currWell == outerWells[3]:
				arm4_Goal+=1
				#print('arm4',arm4_Goal)	
			print('arm1',arm1_Goal,'arm2',arm2_Goal,'arm3',arm3_Goal,'arm4',arm4_Goal)

			# write to a file that will record last rewarded arm so that spykshrk can read it
			# bug: at first content trial it writes all 4 wells, not sure why, after that it works
			# write (currWell - 9) to get back to arms 1-4

			print('current well',currWell)
			with open("/home/lorenlab/spykshrk_realtime/config/rewarded_arm_trodes.txt","a") as reward_arm_file:
				try:
					reward_arm_file.write(str(currWell-9)+' '+str(taskState)+' '+str(goalWell[0])+'\n')
				finally: 
					reward_arm_file.close()

			# use this info to update taskState
			# if required rewards met for all arms, switch to content trials (taskState=2)
			# could replace this with a vector with the visits for each arm
			# 4 arms
			#if arm1_Goal>=outerarm_required_rewards and arm2_Goal>=outerarm_required_rewards and arm3_Goal>=outerarm_required_rewards and arm4_Goal>=outerarm_required_rewards:

			# end of cued trials, two arm V track:
			# activate taskstate 2, tell statescript and decoder
			#if (taskState == 1 and arm1_Goal>=outerarm_required_rewards and arm2_Goal>=outerarm_required_rewards
			#	 and arm3_Goal>=outerarm_required_rewards and arm4_Goal>=outerarm_required_rewards):
			if (taskState == 1 and arm1_Goal>=outerarm_required_rewards
				 and arm2_Goal>=outerarm_required_rewards):			
				taskState = 2
				print("SCQTMESSAGE: taskstate = "+str(taskState)+";\n") # update taskstate in SC
				print('switched to content trials')
				# move decoder start to new function below
				#with open("/home/lorenlab/spykshrk_realtime/config/taskstate.txt","a") as reward_arm_file:
				#	try:
				#		reward_arm_file.write(str(2)+'\n')
				#	finally: 
				#		reward_arm_file.close()				

			# TO DO: turn off all lights at end of taskState 3 and set taskState back to 1 in text file
			elif taskState == 3 and arm1_Goal>=2+outerarm_required_rewards and arm2_Goal>=2+outerarm_required_rewards:
				#with open("/home/lorenlab/spykshrk_realtime/config/taskstate.txt","a") as reward_arm_file:
				#	try:
				#		reward_arm_file.write(str(1)+'\n')
				#	finally: 
				#		reward_arm_file.close()	
				taskState = 4
				print('task finished!')
				print("SCQTMESSAGE: dio = "+str(backWell)+";\n") # turn off back well
				print("SCQTMESSAGE: trigger(4);\n")
				print("SCQTMESSAGE: dio = "+str(centerWell)+";\n")  #turn off all center and outer well lights
				print("SCQTMESSAGE: trigger(4);\n")
				for num in range(len(outerWells)):
					print("SCQTMESSAGE: dio = "+str(outerWells[num])+";\n")
					print("SCQTMESSAGE: trigger(4);\n")
				
			print("SCQTMESSAGE: goalTotal_TS1 = "+str(allGoal)+";\n") # update goaltotal_TS1 in SC

		else:   # wrong well; add to forage record if newly visited
			print("SCQTMESSAGE: otherCount = otherCount + 1;\n") # update othercount in SC

	elif trialtype < 2 and waslock<1:
		pass
		#lockout([0,1])

def endback():
	global trialtype
	global allGoal
	global goalWell 
	global currWell
	global lastWell
	global backWell
	global taskState
	global centerWell

	if trialtype == 1 and lastWell != currWell and taskState == 1 and currWell in goalWell:
		print("SCQTMESSAGE: dio = "+str(backWell)+";\n")
		print("SCQTMESSAGE: trigger(4);\n")			

		print("SCQTMESSAGE: dio = "+str(centerWell)+";\n")   # turn center light on
		print("SCQTMESSAGE: trigger(3);\n")
		print("SCQTMESSAGE: trigger(5);\n")   # display stats

		# this line set the variable delaytime and then writes the variable to statescript  
		delaytime = chooseDelay()
		print("SCQTMESSAGE: waittime = "+str(delaytime)+";\n")	

# called by any visit to outer arm
def endOuter():
	global trialtype
	global outerWells
	global backWell
	global lastWell
	global currWell
	global goalWell
	global centerWell
	global taskState
	global outer_count_content

	# note: this should run at the end of the last cued trial, right after switch to taskstate 2
	#print('lastwell',lastWell)
	#print('curr well',currWell)
	# we only want this to happen if he visits correct well
	#if trialtype == 1 and lastWell != currWell and taskState == 1:  # outer satisfied. old: 0, new: 1
	if trialtype == 1 and lastWell != currWell and taskState == 1 and currWell in goalWell:
		#for num in range(len(outerWells)):			# turn off outer lights
		#	print("SCQTMESSAGE: dio = "+str(outerWells[num])+";\n")
		#	print("SCQTMESSAGE: trigger(4);\n")
		for num in range(len(outerWells)):			# turn off outer lights
			print("SCQTMESSAGE: dio = "+str(outerWells[num])+";\n")
			print("SCQTMESSAGE: trigger(4);\n")			
		# now we need to run endback - to start wait at center
		# original
		#print("SCQTMESSAGE: dio = "+str(backWell)+";\n")   # turn backwell on
		#print("SCQTMESSAGE: trigger(3);\n")
		#print("SCQTMESSAGE: trigger(5);\n")   # display stats
		# no back
		print("SCQTMESSAGE: dio = "+str(centerWell)+";\n")   # turn center light on
		print("SCQTMESSAGE: trigger(3);\n")
		print("SCQTMESSAGE: trigger(5);\n")   # display stats

		# this line set the variable delaytime and then writes the variable to statescript  
		delaytime = chooseDelay()
		print("SCQTMESSAGE: waittime = "+str(delaytime)+";\n")

		# we may want to define a lockout type here

	# for first content trial, trigger 13
	elif taskState == 2 and outer_count_content == 0:
		outer_count_content += 1
		for num in range(len(outerWells)):			# turn off outer lights
			print("SCQTMESSAGE: dio = "+str(outerWells[num])+";\n")
			print("SCQTMESSAGE: trigger(4);\n")
		print('first content trial',outer_count_content)
		print("SCQTMESSAGE: trigger(5);\n")   # display stats
		print("SCQTMESSAGE: trigger(13);\n")

	# return to cued trials
	elif trialtype == 1 and lastWell != currWell and taskState == 3:  # outer satisfied. old: 0, new: 1
		for num in range(len(outerWells)):			# turn off outer lights
			print("SCQTMESSAGE: dio = "+str(outerWells[num])+";\n")
			print("SCQTMESSAGE: trigger(4);\n")
		# now we need to run endback - to start wait at center
		print("SCQTMESSAGE: dio = "+str(centerWell)+";\n")   # turn center light on
		print("SCQTMESSAGE: trigger(3);\n")
		print("SCQTMESSAGE: trigger(5);\n")   # display stats

		# this line set the variable delaytime and then writes the variable to statescript  
		delaytime = chooseDelay()
		print("SCQTMESSAGE: waittime = "+str(delaytime)+";\n")	

# called when "LOCKOUT" printed by statescript or doOuter
def lockout(val):   # turn off all lights for certain amount of time
	global centerWell
	global outerWells
	global lastWell
	global trialtype
	global waslock

	print("lockout val "+str(val)+"\n")
	locktype = int(val[1])
	trialtype = 3
	print("SCQTMESSAGE: trialPhase = "+str(trialtype)+";\n")
	#print("SCQTMESSAGE: trigger(6);\n")  # start lockout timer in SCQTMESSAGE
	#turn off all lights
	print("SCQTMESSAGE: dio = "+str(backWell)+";\n") # turn off back well
	print("SCQTMESSAGE: trigger(4);\n")
	print("SCQTMESSAGE: dio = "+str(centerWell)+";\n")  #turn off all center and outer well lights
	print("SCQTMESSAGE: trigger(4);\n")
	for num in range(len(outerWells)):
		print("SCQTMESSAGE: dio = "+str(outerWells[num])+";\n")
		print("SCQTMESSAGE: trigger(4);\n")
	waslock=1
	print("SCQTMESSAGE: waslock = "+str(waslock)+";\n") # turn off back well
	if locktype == 1:
		print("SCQTMESSAGE: locktype1 = locktype1 + 1;\n") # type 1 = wrong well order
	if locktype == 2:
		print("SCQTMESSAGE: locktype2 = locktype2 + 1;\n") # type 2 = impatience at center well

# read from statescript, used to re-start cued arm trials
# NOTE: changed global backWell to centerWell
def lockend():
	global trialtype
	global centerWell

	# no back: now reset trial type to 1 not 0, and tell statescript
	trialtype = 1
	print("SCQTMESSAGE: trialPhase = "+str(trialtype)+";\n")
	# back well
	#print("SCQTMESSAGE: trialPhase = "+str(trialtype)+";\n")
	#print("SCQTMESSAGE: dio = "+str(backWell)+";\n") # turn on back well
	#rint("SCQTMESSAGE: trigger(3);\n")

	# no back
	print("SCQTMESSAGE: dio = "+str(centerWell)+";\n")   # turn center light on
	print("SCQTMESSAGE: trigger(3);\n")
	print("SCQTMESSAGE: trigger(5);\n")   # display stats

	# choose wait time, tell statescript
	delaytime = chooseDelay()
	print("SCQTMESSAGE: waittime = "+str(delaytime)+";\n")

	# set last well to 10 so that loop with center well timer will run
	print("SCQTMESSAGE: lastPort = "+str(5)+";\n")


# called from statescript, when "NEXT_TRIAL" displayed
def startContentTrial():
	global trialtype
	global timer_mean
	global target_location_vec
	global trial_instructive
	global currWell
	global centerWell
	global n_generated_future_BEEP 
	global n_th_BEEP
	global timer_max 
	global timer_min 
	#global content_trial_dist

	# no back: now reset trial type to 1 not 0, and tell statescript
	# with back, keep as trialtype 1
	trialtype = 1
	print("SCQTMESSAGE: trialPhase = "+str(trialtype)+";\n")
	print("SCQTMESSAGE: trigger(5);\n")   # display stats

	# choose time between trials
	#content_trial_time = int(np.random.choice(content_trial_dist,1))
	# sample from exponential distribution, mean 30 sec
	# note: 3 sec is added in statescript

	# original line for trial length
	# 30 here is for 30 seconds
	content_trial_time = int((1 + np.random.exponential(scale= timer_mean, size=None)))
	if content_trial_time > timer_max: 
		content_trial_time = timer_max
	if content_trial_time < timer_min:
		content_trial_time =timer_min + content_trial_time
	# testing - reward every 3 sec
	#content_trial_time = 3000
	
	if (currWell == centerWell or currWell == backWell) and n_generated_future_BEEP == n_th_BEEP:

		print("SCQTMESSAGE: timer = "+str(content_trial_time)+";\n")
		print('start next content trial')
		
		print("SCQTMESSAGE: trigger(16);\n")
		
		print('n_generated_future_BEEP:' + str(n_generated_future_BEEP))
		print('n_th_BEEP:' + str(n_th_BEEP))
		
	else:
		print("SCQTMESSAGE: disp('Wrong recent arm "+str(currWell)+"');\n")
		print('n_th_BEEP:' + str(n_th_BEEP))
		print('n_generated_future_BEEP:' + str(n_generated_future_BEEP))

def send_target_location():
	global target_location_vec
	global trial_instructive
	global number_max_trial
	
	print('sending target location to statescript')
	print("SCQTMESSAGE: target_location = "+str(target_location_vec[trial_instructive%number_max_trial])+";\n")
	print("SCQTMESSAGE: disp(target_location);\n")
	trial_instructive = trial_instructive +1 

# called from statescript, when "TASKSTATE3" displayed
def returnToCued():
	global taskState
	global target_location_vec
	global animal_decision_vec
	print('return to cued trials')
	taskState = 3
	# tell decoder it is taskState 3 now
	with open("/home/lorenlab/realtime_decoder/config/taskstate.txt","a") as reward_arm_file:
		try:
			reward_arm_file.write(str(3)+'\n')
		finally: 
			reward_arm_file.close()	
	# use lockend to start new cued trial
	#lockend()
	# use normal BEEP1 and BEEP2 to start first trial
	chooseGoal()
	beep_center()
	endWait()
	print("SCQTMESSAGE: disp('correct target order " + str(target_location_vec) + "')\n")
	print("SCQTMESSAGE: disp('animal decision order " + str(animal_decision_vec) + "')\n")

# called from statescript, when "waslock" displayed
def updateWaslock(val):
	global waslock

	waslock = int(val[1])
	print("SCQTMESSAGE: waslock = "+str(waslock)+";\n")

# Function: generate cowbell sound
def generate_beep():
	global stream
	stream.write(data)

def makewhitenoise():  #play white noise for duration of lockout
	global locksoundlength
	global whitenoiseloudness_scale

	soundlength = int(44100*locksoundlength/1000)
	p = pyaudio.PyAudio()
	stream = p.open(format = 8, channels = 1, rate = 44100, output = True)
	whitenoise = whitenoiseloudness_scale * np.random.randint(700,size = soundlength)
	data = struct.pack("%dh"%(len(whitenoise)), *list(whitenoise))    
	stream.write(data)
	stream.close()
	p.terminate()

# Function: generate cowbell sound
def generate_epoch_end_song():
	File='Jeopardy.wav'
	volume_scale = 0.1
	spf = wave.open(File, 'rb')
	signal = spf.readframes(-1)
	signal = np.frombuffer(signal, dtype='Int16')
	signal = np.int16(signal * volume_scale)
	
	p = pyaudio.PyAudio()
	stream = p.open(format =p.get_format_from_width(spf.getsampwidth()),
				channels = spf.getnchannels(),
				rate = spf.getframerate(),
				output = True)
	#play 
	data = struct.pack("%dh"%(len(signal)), *list(signal))    
	stream.write(data)
	stream.close()
	p.terminate()
		

def decoder_task2():
	print('write taskstate2 for decoder')
	with open("/home/lorenlab/realtime_decoder/config/taskstate.txt","a") as reward_arm_file:
		try:
			reward_arm_file.write(str(2)+'\n')
		finally: 
			reward_arm_file.close()	
	print(f"target order = {target_location_vec}")
	print("SCQTMESSAGE: timer = " + str(timer_min) + ";\n")
def print_decision_status():

	global target_location_vec
	global target_arm_vec
	global animal_decision_vec 
	
	print("SCQTMESSAGE: disp('correct target order " + str(target_location_vec) + "')\n")
	print("SCQTMESSAGE: disp('target that were used " + str(target_arm_vec) + "')\n")		
	print("SCQTMESSAGE: disp('animal decision order " + str(animal_decision_vec) + "')\n")





def generate_list():

    global number_max_trial
    global max_consecutive
    # Initialize the first element randomly to start the list
    result = [random.choice([1, 2])]
    
    while len(result) < number_max_trial:
        current_choice = random.choice([1, 2])
        # Count how many times the last number has occurred consecutively
        if len(result) >= max_consecutive:
            # Check the last few entries if they are the same and match the current choice
            if all(x == result[-1] for x in result[-max_consecutive:]):
                # Force a different choice
                current_choice = 1 if result[-1] == 2 else 2
        result.append(current_choice)
    
    return result

# This is the custom callback function. When events occur, addScQtEvent will
# call this function. This function MUST BE NAMED 'callback'!!!!
def callback(line):

	global waslock
	global goalWell
	global replay_arm 
	global n_generated_future_BEEP
	global n_th_BEEP
	global target_arm_vec
	global animal_decision_vec
	global target_location_vec
	global target_location_vec1
	global target_location_vec2
	global stream # for beep sound 
	global p  # for beep sound
	global timer_mean
	global timer_max
	global timer_min
	global correct_trial_bit
	global reward_center_during_TS2
	
	if line.find("UP") >= 0: #input triggered
		pokeIn(re.findall(r'\d+',line))
	if line.find("DOWN") >= 0: #input triggered
		pokeOut(re.findall(r'\d+',line))
	# add ripwait to holding vector
	if line.find("riptime") >=0:
		addtime(re.findall(r'\d+',line))
	if line.find("BEEP1") >= 0:
		chooseGoal()
		print(target_location_vec)
		#beep()
	# this is only called by cued trials, remove sound cue
	if line.find("BEEP2") >= 0:
		beep_center()
		#generate_beep()
		#mec added to turn on outer lights at same time as beep
		endWait()
	# for content trials. beep sound
	if line.find("BEEP3") >= 0:
		generate_beep()
	# for content trials. reward only no beep sound
	if line.find("BEEP4") >= 0:
		beep_center()
	if line.find("LOCKOUT") >= 0: # lockout procedure
		lockout(re.findall(r'\d+',line))
	if line.find("LOCKEND") >= 0: # reset trialtype to 0
		lockend()
	if line.find("WHITENOISE") >= 0: # make noise during lockout
		makewhitenoise()
	if line.find("waslock") >= 0:  #update waslock value
		updateWaslock(re.findall(r'\d+',line))
	# function for reading specific arm output from spykshrk
	# note: had to reprint statescript variable replay_arm after it comes in, in order for python to see it
	if line.find("replay_arm") >= 0:
		replay_arm = re.findall(r'\d+',line)
		replay_arm = int(replay_arm[1])
		print('replay arm from callback', replay_arm)
	# to start next content trial based on function 18 in statescript
	if line.find("Timer ends") >=0:
		n_th_BEEP = n_th_BEEP + 1
	if line.find("NEXT_TRIAL") >= 0:
		startContentTrial()
		
		
	
	if line.find("TASKSTATE3") >= 0: #(DS) This should go away soon as TERMINATE EPOCH does the same function
		returnToCued()	
	if line.find("TERMINATE EPOCH") >= 0:
		returnToCued()	
		print("SCQTMESSAGE: taskstate = 3; \n")
		print("SCQTMESSAGE: portout[" + str(outerWells[0]) + "] = 0; \n")
		print("SCQTMESSAGE: portout[" + str(outerWells[1]) + "] = 0; \n")
		print("SCQTMESSAGE: portout[" + str(centerWell) + "] = 0; \n")
		
		stream.close()
		p.terminate()
		
		print_decision_status()
		generate_epoch_end_song()
		
	# to write taskstate2 to text file for decoder
	if line.find("DECODER_TASK2") >= 0:
		decoder_task2()
		
		
	# correct choice of the arm	
	if line.find("REWARD ARM1") >= 0:
		reward_delivery_at_arm1()
		print("SCQTMESSAGE: lockoutPeriod_centerReward = " + str(lockoutPeriod_centerReward) + ";\n")
		animal_decision_vec.append(1)
		print("SCQTMESSAGE: reward_delivered = 1; \n")
		print("SCQTMESSAGE: centerRewardReady = 1; \n")
		print("SCQTMESSAGE: contentCorrectDecision = contentCorrectDecision + 1; \n")		
		correct_trial_bit = True
		
	if line.find("REWARD ARM2") >= 0:
		reward_delivery_at_arm2()		
		print("SCQTMESSAGE: lockoutPeriod_centerReward = " + str(lockoutPeriod_centerReward) + ";\n")
		animal_decision_vec.append(2)
		print("SCQTMESSAGE: reward_delivered = 1; \n")
		print("SCQTMESSAGE: centerRewardReady = 1; \n")
		print("SCQTMESSAGE: contentCorrectDecision = contentCorrectDecision + 1; \n")
		correct_trial_bit = True

	if line.find("REWARD ARM3") >= 0:
		reward_delivery_at_arm3()		
		print("SCQTMESSAGE: lockoutPeriod_centerReward = " + str(lockoutPeriod_centerReward) + ";\n")
		animal_decision_vec.append(3)
		print("SCQTMESSAGE: reward_delivered = 1; \n")
		print("SCQTMESSAGE: centerRewardReady = 1; \n")
		print("SCQTMESSAGE: contentCorrectDecision = contentCorrectDecision + 1; \n")
		correct_trial_bit = True

	# wrong choice of the arm	
	if line.find("WRONG ARM1") >= 0:
		animal_decision_vec.append(1)
		print("SCQTMESSAGE: lockoutPeriod_centerReward = " + str(lockoutPeriod_centerNoReward) + ";\n")
		makewhitenoise()
		print("SCQTMESSAGE: reward_avail = 0; \n") 
		print("SCQTMESSAGE: centerRewardReady = 1; \n") 
		correct_trial_bit = False
		
	if line.find("WRONG ARM2") >= 0:
		animal_decision_vec.append(2)	
		print("SCQTMESSAGE: lockoutPeriod_centerReward = " + str(lockoutPeriod_centerNoReward) + ";\n")
		makewhitenoise()
		print("SCQTMESSAGE: reward_avail = 0; \n") 
		print("SCQTMESSAGE: centerRewardReady = 1; \n") 
		correct_trial_bit = False

	if line.find("WRONG ARM3") >= 0:
		animal_decision_vec.append(3)	
		print("SCQTMESSAGE: lockoutPeriod_centerReward = " + str(lockoutPeriod_centerNoReward) + ";\n")
		makewhitenoise()
		print("SCQTMESSAGE: reward_avail = 0; \n") 
		print("SCQTMESSAGE: centerRewardReady = 1; \n") 
		correct_trial_bit = False
		
	# outer wrong arm visit at the wrong time
	if line.find("INVALID OUTER ARM VISITS") >=0:
		makewhitenoise()
		print("SCQTMESSAGE: invalid_outer_arm_visits = invalid_outer_arm_visits + 1; \n")
		print("SCQTMESSAGE: reward_available_out_if_poke = 0; \n") 
		print("SCQTMESSAGE: wrong_outer_visit = 1; \n") 
		
		
	if line.find("OUTER ARMS DONE") >= 0:
		#print("SCQTMESSAGE: portout[9] = 0 \n")
		#print("SCQTMESSAGE: portout[15] = 0 \n")
		#print("SCQTMESSAGE: portout[7] = 1 \n")
		print("SCQTMESSAGE: portout[" + str(outerWells[0]) + "] = 0; \n")
		print("SCQTMESSAGE: portout[" + str(outerWells[1]) + "] = 0; \n")
		print("SCQTMESSAGE: portout[" + str(centerWell) + "] = 1; \n")	
	
	
	
	if line.find("Timer starts") >=0:
		n_generated_future_BEEP = n_generated_future_BEEP + 1
		print("SCQTMESSAGE: trigger(5);\n")
	if line.find("DECISION TIMEOUT") >=0:
		animal_decision_vec.append(0)	
	if line.find("GET TARGET") >=0:
		send_target_location()
	if line.find("TARGET ARM1") >= 0:
		target_arm_vec.append(1)
	if line.find("TARGET ARM2") >= 0:
		target_arm_vec.append(2)
	if line.find("DISPLAY") >= 0:
		print_decision_status()
	if line.find("FREE TRIAL") >= 0:
		target_location_vec = target_location_vec2
	if line.find("TARGET TRIAL") >= 0:
		target_location_vec = target_location_vec1
	if line.find("TIMERMEAN") >= 0: #input triggered
		value = int(re.findall(r'\d+',line)[1])
		print(type(value))
		print(value)
		timer_mean = int(value) #ms         
		timer_max = value + 8000             
		timer_min = value - 8000
		print(timer_mean)
		print(timer_max)
		print(timer_min)
	if line.find('reward_center_during_TS2') >=0:
		if reward_center_during_TS2 == 0:
			reward_center_during_TS2 = 1
		else:
			reward_center_during_TS2 = 0
	

# all global variables are initialized
# all variables can be used anywhere in this script
# define wells, old outer: 10,11,12,13
# 1st version: wells 8 (8) and 4 (12)
# 2nd version: arms 6 (10) and 2 (14)
# 3rd version: arms 7 (9) and 3 (13)
# 4th version: arms 5(11) and 1 (15)
# tree track: arm 2 (14) and 1 (15) 
# 4 arm: arm5 (12), arm6 (13), arm7 (14), arm8 (15) 


backWell = 5
centerWell = 7
outerWells = [15,9,4] #CHANGED 13 TO 15 (20240609- DS)
#outerWells = [12,13,14,15]
#cuedWells=[1,8,12]
#cuedWells = [1,10,14]
#cuedWells = [1,9,13]
#cuedWells = [1,11,15]

# define pumps +9 for arm
# old outer: 19,20,21,22
# pumps: 8 (17) and 12 (21)
# 2nd version: 10 (19) and 14 (23)
# 3rd version: 9 (18) and 13 (22)
# 4th version: 11 (20) and 15 (24)
# tree track: 14 (23) and 15 (24)
# 4 arm: 21,22,23,24

# problem with breakout board for pump at arm 1, moved from 24 to 22
backPump = 8
centerPump = 14
outerPumps = [11,12,10]

#outerPumps = [21,22,23,24]
#cuedPumps = [25,20,24]

#global variables
# new startup: lastwell = 10, before lastwell = -1
lastWell = -1
currWell = -1

# no back: initiaze with trialtype 2 (was 0 before) and turn on one outer well light
# new startup: trialtype = 1, before trialtype = 2
trialtype = 1

allGoal = 0

# counters for each indivudal arm
arm1_Goal = 0
arm2_Goal = 0
arm3_Goal = 0
arm4_Goal = 0
back_Goal = 0

outer_count_content = 0
backcount = 0
outer_arm_reward = 0
arm1_counter = 0
arm2_counter = 0
arm1_order8 = []
arm2_order8 = []

# task state variable
# 1 = cued reward well
# 2 = content contigent rewrad well
taskState = 1
center_reward_counter = 0
# should start at -1 so that it doesnt return a real arm
# this may require a check that replay-arm does not equal -1
replay_arm = 0

# choose one well at random of the 4
# no back - try to initalize with well 10 and a list (was 0 before)
goalWell = [5]
cued_trial_counter = 0
oldGoal1 = 0 
oldGoal2 = 0
start_session = 0

startwaitdist = [100, 100, 100]
waitdist = [100] 
#content_trial_dist = [5000]
count = 0

waslock=0
centercount = 0
n_generated_future_BEEP = 0 # this two variable has to match in order for rat to get BEEP (to prevent rats to get reward if it pokes outer arm)
n_th_BEEP = 0
locksoundlength = 1000
whitenoiseloudness_scale = 20

number_max_trial = 75
max_consecutive = 5
target_location_vec1 = generate_list()
target_location_vec2 = [4] * number_max_trial



animal_decision_vec = []
target_arm_vec = []
trial_instructive = 0
correct_trial_bit = True  #in TS2, giving reward at the center if only correct

#(DS) preprocess the beep sound
##############################################################################################################################
File='Beep.wav'
spf = wave.open(File, 'rb')
signal = spf.readframes(-1)
signal = 0.3*np.fromstring(signal, 'Int16')
signal = np.clip(signal, -32768, 32767)
signal = signal.astype(np.int16)

p = pyaudio.PyAudio()	
stream = p.open(format =p.get_format_from_width(spf.getsampwidth()),
			channels = 1,
			rate = spf.getframerate(),
			output = True)
#play 
data = struct.pack("%dh"%(len(signal)), *list(signal))   
##############################################################################################################################




# (DS) variables that I can change 
##############################################################################################################################
# TS2 timer variable
timer_mean = 6000 #ms         # timer average; interval between beeps -- 20 is for final pretraining, reduce it to 5 for testing;
timer_max = 12000 #ms             # timer max;  maximum timer between trial is 40s
timer_min = 3000 #ms              # timer min;

# TS2: This panal is about reward delivery at center in TS2;
reward_center_during_TS2 = False 
give_only_at_the_correct_bit = True # osnly correct at center in TS2; 

#TS1
half_reward_at_center = True # TS1: if giving half rewards at center during TS1; False = 100%, True = 50%

# TS1 outer arm variable
outerarm_required_rewards = 12  # number of visits required to change task state from 1 to 2
ts1_reward_prob = 50 #for outer arm

# lockout variable (no need to worry about this for now)
lockoutPeriod_centerReward = 0;# NOTE(DS): when no reward at the center no need for a lockout15000;
lockoutPeriod_centerNoReward = 0;

# determining the target location (1 if target determined by the system, 2 if target determined by the rat)
target_location_vec = target_location_vec1 # vec1 if target_trial, vec2 if free_trial
##############################################################################################################################

