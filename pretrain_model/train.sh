PROBLEM=gec_pretrain_transformer
MODEL=transformer
HPARAMS=transformer_base_gec

USR_DIR=./
DATA_DIR=./data_dir/origin_dataset
TMP_DIR=./tmp_dir
OUTPUT_DIR=./output_dir/retraining

#mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR
#
##Generate data
#t2t-datagen \
#   --data_dir=$DATA_DIR \
#   --tmp_dir=$TMP_DIR \
#   --problem=$PROBLEM \
#   --t2t_usr_dir=$USR_DIR
#
##Train
#t2t-trainer \
#   --data_dir=$DATA_DIR \
#   --problem=$PROBLEM \
#   --model=$MODEL \
#   --hparams_set=$HPARAMS \
#   --output_dir=$OUTPUT_DIR \
#   --t2t_usr_dir=$USR_DIR \
#   #--train_steps=77000  #合成数据100万句，subword数量31309609，batch_size=2048, 所以一个epoch 31309609/2048步, pre-trainig 5 个epoch， 只在合成数据上设置该参数
#   #--train_steps=77000/GPU_nums #GPU_params
#
#
#exit

#Decode
DECODE_FILE_PATH=$TMP_DIR/wi_locness/wi_locness.NABC.dev.pieces.src
DECODE_RESULT=./wi_locness.NABC.dev.pieces.tgt.finetuning

t2t-decoder \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --output_dir=$OUTPUT_DIR \
    --decode_hparams="beam_size=12,alpha=0.6" \
    --decode_from_file=$DECODE_FILE_PATH \
    --decode_to_file=$DECODE_RESULT \
    --t2t_usr_dir=$USR_DIR