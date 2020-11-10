#!/bin/bash

PYTHON="/usr/local/Cellar/python@3.8/3.8.5/bin/python3.8"
############################
#						   #
# default setting for LTR  #
#						   #
############################

LTR_DATA_DIR=out/ranklib_data
LTR_SRC_DIR=src/ranklib
LTR_RANKER=ListNet
LTR_RANKER_ID=7
LTR_OPT_METRIC=NDCG
LTR_OPT_K=500
LTR_LEARNING_RATE=0.000001
LTR_EPOCHS=10000


#########################
#						#
# default setting for R	#
#						#
#########################
R_DATA_DIR=out
R_SRC_DIR=src/rscripts


#################################
#						        #
# default setting for $PYTHON	#
#						        #
#################################
PY_SRC_DIR=src/pyscripts
export PY_SRC_DIR="src/pyscripts"
export PY_DATA_SRC_DIR=data


####################################
#						           #
# default setting for experiment   #
#						           #
####################################
DATA_N=2000  # only validation purpose
COUNT_FILE_NAME="_count"
SPLIT_FLAG=None
EVAL_K=50,100,200
OUT_DATA_DIR="synthetic_data"
SRC_DATA=None

LTR_TRIAL_N=10
DATA_TRIAL_N=20
LTR_SETTINGS=("Full")



###########################################################
#						                                  #
# INPUT SETTING for experiment, change based on dataset   #
#						                                  #
###########################################################
DATA_FLAG="$1"
MODEL_FLAG="$2"
MODEL_DESP="$3"
MEDIATOR_ATT="$4"

#################################################
#						                        #
# Functions for complete routine of experiments	#
#						                        #
#################################################

#################################################
#						        #
# Functions for generate data	#
#						        #
#################################################

echo "$PYTHON \"$PY_SRC_DIR/gen_orig_data.py\" --data_dir $OUT_DATA_DIR --data_flag $DATA_FLAG --para_file $MODEL_DESP --run $DATA_TRIAL_N"
$PYTHON "$PY_SRC_DIR/gen_orig_data.py" --data_dir $OUT_DATA_DIR --data_flag $DATA_FLAG --para_file $MODEL_DESP --run $DATA_TRIAL_N
# unverstood it samples probabilities from a normal distribution.

#################################################
#						        #
# Functions for adding noise	#
#						        #
#################################################
echo "$PYTHON \"../gen_noise-cifrank.py\""
$PYTHON "../gen_noise-cifrank.py"
#########################################################
#						                                #
# Functions for estimate causal model on the data,      #
# causal model is specified in 'rscripts'               #
#						                                #
#########################################################
Rscript --vanilla "$R_SRC_DIR/${DATA_FLAG}_${MODEL_FLAG}.R" $R_DATA_DIR $DATA_TRIAL_N
############################################################################
#
# Functions to get counterfactual data from estimated causal model         #
#
############################################################################
if [ $MEDIATOR_ATT == "G" ]
then
	## first experiment
	COUNTER_G="F" # counder factual group
	OTHER_G="M"
	HIDDEN_G="B"
	echo "MEDIATION ON SINGLE SENSITIVE ATTRIBUTE GENDER"
	echo "$PYTHON \"$PY_SRC_DIR/gen_counter_data.py\" --data_dir $OUT_DATA_DIR --data_flag $DATA_FLAG --model_flag $MODEL_FLAG --counter_g $COUNTER_G --other_g $OTHER_G --hidden_g $HIDDEN_G --med_s $MEDIATOR_ATT --val_n $DATA_N --counter_run $DATA_TRIAL_N --src_data $SRC_DATA"
	$PYTHON "$PY_SRC_DIR/gen_counter_data.py" --data_dir $OUT_DATA_DIR --data_flag $DATA_FLAG --model_flag $MODEL_FLAG --counter_g $COUNTER_G --other_g $OTHER_G --hidden_g $HIDDEN_G --med_s $MEDIATOR_ATT --val_n $DATA_N --counter_run $DATA_TRIAL_N --src_data $SRC_DATA

else
	if [ $MEDIATOR_ATT == "R" ]
	then
		COUNTER_G="B" # counder factual group
		OTHER_G="W"
		HIDDEN_G="F"
		echo "MEDIATION ON SINGLE SENSITIVE ATTRIBUTE RACE"
		$PYTHON "$PY_SRC_DIR/gen_counter_data.py" --data_dir $OUT_DATA_DIR --data_flag $DATA_FLAG --model_flag $MODEL_FLAG --counter_g $COUNTER_G --other_g $OTHER_G --hidden_g $HIDDEN_G --med_s $MEDIATOR_ATT --val_n $DATA_N --counter_run $DATA_TRIAL_N --src_data $SRC_DATA

	else
		COUNTER_G="FB" # counder factual group
		echo "MEDIATION ON MULTIPLE SENSITIVE ATTRIBUTES"
		$PYTHON "$PY_SRC_DIR/gen_counter_data.py" --data_dir $OUT_DATA_DIR --data_flag $DATA_FLAG --model_flag $MODEL_FLAG --counter_g $COUNTER_G --val_n $DATA_N --counter_run $DATA_TRIAL_N --src_data $SRC_DATA
	fi
fi

# exit 0

#################################################
#						        #
# Functions for denoising the counterfactual data for evaluation	#
#						        #
#################################################
echo "$PYTHON \"../denoise_count.py\""
$PYTHON "../denoise_count.py"

##############################################
#						                     #
# Functions to prepare ranklib inputs        #
# ONLY SUPPORT MODEL m1 and m2 NOW	         #
#						                     #
##############################################
if [ $MODEL_FLAG == "m2" ]
then
	EVAL_COUNTER_RANKINGS="Y,Y_count,Y_count_resolve"
	EVAL_LTR_RANKINGS="Y__Y__full,Y_count__Y__full,Y_count__Y_count__full,Y_count_resolve__Y__full,Y_count_resolve__Y_count_resolve__full"
	echo "EVALUATION ON BOTH RESOLVING AND NON-RESOLVING"
else
	EVAL_COUNTER_RANKINGS="Y,Y_count"
	EVAL_LTR_RANKINGS="Y__Y__full,Y_count__Y__full,Y_count__Y_count__full"
	echo "EVALUATION ON NON-RESOLVING"
fi

# # evaluation for selection rate
$PYTHON "$PY_SRC_DIR/eval_rankings.py" --data_flag $DATA_FLAG --model_flag $MODEL_FLAG --eval_ks $EVAL_K --rankings "$EVAL_COUNTER_RANKINGS,Y_quotas_R,Y_quotas_G,Y_quotas_GR" --measure select_rate --file_n $COUNT_FILE_NAME
# evaluation for rKL
$PYTHON "$PY_SRC_DIR/eval_rankings.py" --data_flag $DATA_FLAG --model_flag $MODEL_FLAG --eval_ks $EVAL_K --rankings $EVAL_COUNTER_RANKINGS --measure rKL --file_n $COUNT_FILE_NAME
# evaluation for ratio
$PYTHON "$PY_SRC_DIR/eval_rankings.py" --data_flag $DATA_FLAG --model_flag $MODEL_FLAG --eval_ks $EVAL_K --rankings $EVAL_COUNTER_RANKINGS --measure igf --file_n $COUNT_FILE_NAME
# evaluation for score utility
$PYTHON "$PY_SRC_DIR/eval_rankings.py" --data_flag $DATA_FLAG --model_flag $MODEL_FLAG --eval_ks $EVAL_K --rankings "$EVAL_COUNTER_RANKINGS,Y_quotas_R,Y_quotas_G,Y_quotas_GR" --measure score_utility --file_n $COUNT_FILE_NAME

$PYTHON "$PY_SRC_DIR/eval_rankings.py" --data_flag $DATA_FLAG --model_flag $MODEL_FLAG --eval_ks $EVAL_K --rankings "$EVAL_COUNTER_RANKINGS,Y_quotas_R,Y_quotas_G,Y_quotas_GR" --measure selection_lift --file_n $COUNT_FILE_NAME

$PYTHON ../custom_plot.py

#################################
#						        #
# Functions to generate plots 	#
#						        #
#################################

$PYTHON "$PY_SRC_DIR/gen_plots.py" --data_flag $DATA_FLAG --model_flag $MODEL_FLAG --rankings "$EVAL_COUNTER_RANKINGS,Y_quotas_R,Y_quotas_G,Y_quotas_GR" --plot_ks $EVAL_K --y_col select_rate --y_max 2.2 --file_n $COUNT_FILE_NAME
$PYTHON "$PY_SRC_DIR/gen_plots.py" --data_flag $DATA_FLAG --model_flag $MODEL_FLAG --rankings $EVAL_COUNTER_RANKINGS --plot_ks $EVAL_K --y_col rKL --y_max 2.1
