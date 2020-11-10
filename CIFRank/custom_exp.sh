PYTHON="/usr/local/Cellar/python@3.8/3.8.5/bin/python3.8"

DATA_FLAG="mv"
MODEL_FLAG="m2"
MODEL_DESP="data/mv_m2.json"
MEDIATOR_ATT="G"

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
# EVAL_K=50,100,100
EVAL_K=50,100
OUT_DATA_DIR="synthetic_data"
SRC_DATA=None

LTR_TRIAL_N=10
DATA_TRIAL_N=$1
LTR_SETTINGS=("Full")

# Rscript --vanilla "$R_SRC_DIR/${DATA_FLAG}_${MODEL_FLAG}.R" $R_DATA_DIR $DATA_TRIAL_N

# MEDIATION ON SINGLE SENSITIVE ATTRIBUTE GENDER
COUNTER_G="F" # counder factual group
OTHER_G="M"
HIDDEN_G="B"
$PYTHON "$PY_SRC_DIR/gen_counter_data.py" --data_dir $OUT_DATA_DIR --data_flag $DATA_FLAG --model_flag $MODEL_FLAG --counter_g $COUNTER_G --other_g $OTHER_G --hidden_g $HIDDEN_G --med_s $MEDIATOR_ATT --val_n $DATA_N --counter_run $DATA_TRIAL_N --src_data $SRC_DATA

$PYTHON "denoise_count.py" $DATA_TRIAL_N

# EVALUATION ON BOTH RESOLVING AND NON-RESOLVING
EVAL_COUNTER_RANKINGS="Y,Y_count,Y_count_resolve"

# $PYTHON "$PY_SRC_DIR/eval_rankings.py" --data_flag $DATA_FLAG --model_flag $MODEL_FLAG --eval_ks $EVAL_K --rankings "$EVAL_COUNTER_RANKINGS,Y_quotas_R,Y_quotas_G,Y_quotas_GR" --measure select_rate --file_n $COUNT_FILE_NAME
# # evaluation for rKL
# $PYTHON "$PY_SRC_DIR/eval_rankings.py" --data_flag $DATA_FLAG --model_flag $MODEL_FLAG --eval_ks $EVAL_K --rankings $EVAL_COUNTER_RANKINGS --measure rKL --file_n $COUNT_FILE_NAME
# # evaluation for ratio
# $PYTHON "$PY_SRC_DIR/eval_rankings.py" --data_flag $DATA_FLAG --model_flag $MODEL_FLAG --eval_ks $EVAL_K --rankings $EVAL_COUNTER_RANKINGS --measure igf --file_n $COUNT_FILE_NAME
# # evaluation for score utility
# $PYTHON "$PY_SRC_DIR/eval_rankings.py" --data_flag $DATA_FLAG --model_flag $MODEL_FLAG --eval_ks $EVAL_K --rankings "$EVAL_COUNTER_RANKINGS,Y_quotas_R,Y_quotas_G,Y_quotas_GR" --measure score_utility --file_n $COUNT_FILE_NAME
##### CUSTOM
# ,Y_quotas_R,Y_quotas_G,Y_quotas_GR
$PYTHON "$PY_SRC_DIR/eval_rankings.py" --data_flag $DATA_FLAG --model_flag $MODEL_FLAG --eval_ks $EVAL_K --rankings "$EVAL_COUNTER_RANKINGS" --measure raw_counts --file_n $COUNT_FILE_NAME
$PYTHON "$PY_SRC_DIR/eval_rankings.py" --data_flag $DATA_FLAG --model_flag $MODEL_FLAG --eval_ks $EVAL_K --rankings "$EVAL_COUNTER_RANKINGS" --measure get_util --file_n $COUNT_FILE_NAME


touch done_causal.mark
