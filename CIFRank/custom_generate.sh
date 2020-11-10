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
EVAL_K=50,100
OUT_DATA_DIR="synthetic_data"
SRC_DATA=None

LTR_TRIAL_N=10
DATA_TRIAL_N=$1
LTR_SETTINGS=("Full")

$PYTHON "$PY_SRC_DIR/gen_orig_data.py" --data_dir $OUT_DATA_DIR --data_flag $DATA_FLAG --para_file $MODEL_DESP --run $DATA_TRIAL_N

touch done_generation.mark
