#!/bin/sh


src=en
tgt=de

SAVEDIR=saves/TED/$src-$tgt
MODELDIR=$SAVEDIR/models/contextGateTW
LOGDIR=$SAVEDIR/logs/


mkdir -p $LOGDIR
mkdir -p $MODELDIR

type=lstm
method=adam
size=512


th bin/train.lua -data $SAVEDIR/tensor-ss-train.t7 -save_model $MODELDIR/model.$type.$method.$size \
			 -layers 2 \
			 -rnn_size $size \
			 -brnn \
			 -brnn_merge concat \
			 -word_vec_size $size \
			 -input_feed 1 \
			 -dropout 0.5 \
			 -optim $method \
			 -learning_rate 0.002 \
			 -max_batch_size 128 \
			 -start_epoch 1 \
			 -end_epoch 25 \
			 -start_decay_at 12 \
			 -learning_rate_decay 0.7 \
			 -seed 1234 \
			 -attention cgate \
			 -tie_embedding \
			 -gpuid 1 2>&1 | tee $LOGDIR/$type.$method.$size.contextGateTW.log
			 
			 
			 
#~ th bin/train.lua -data $SAVEDIR/tensor-train.t7 -save_model $MODELDIR/model.$type.$method.$size \
			 #~ -layers 2 \
			 #~ -rnn_size $size \
			 #~ -brnn \
			 #~ -brnn_merge concat \
			 #~ -word_vec_size $size \
			 #~ -input_feed 1 \
			 #~ -dropout 0.5 \
			 #~ -optim $method \
			 #~ -learning_rate 0.001 \
			 #~ -max_batch_size 128 \
			 #~ -start_epoch 10 \
			 #~ -end_epoch 25 \
			 #~ -start_decay_at 12 \
			 #~ -learning_rate_decay 0.7 \
			 #~ -seed 1234 \
			 #~ -attention cgate \
			 #~ -tie_embedding \
			 #~ -train_from $MODELDIR/model.lstm.adam.512_epoch9_9.79.t7 \
			 #~ -gpuid 1 2>&1 | tee $LOGDIR/$type.$method.$size.contextGateTW.log
			 #~ 
#~ th trainLenPredictor.lua -data $SAVEDIR/tensor-train.t7 -save_model $MODELDIR/lenPrediction/model.$type.$method.$size \
			 #~ -layers 2 \
			 #~ -rnn_size $size \
			 #~ -brnn \
			 #~ -brnn_merge concat \
			 #~ -word_vec_size $size \
			 #~ -input_feed 1 \
			 #~ -dropout 0.5 \
			 #~ -optim $method \
			 #~ -learning_rate 0.001 \
			 #~ -max_batch_size 128 \
			 #~ -start_epoch 1 \
			 #~ -end_epoch 10 \
			 #~ -start_decay_at 8 \
			 #~ -learning_rate_decay 0.7 \
			 #~ -seed 1234 \
			 #~ -attention cgate \
			 #~ -train_from $MODELDIR/model.$type.$method.${size}_epoch15* \
			 #~ -gpuid 1 2>&1 | tee $LOGDIR/$type.$method.$size.contextGate.log
			 #~ 
# Resume training from epoch 5 
#~ th train.lua -data $SAVEDIR/WMT-train.t7 -save_model $MODELDIR/model.$type.$method.$size \
			 #~ -layers 2 \
			 #~ -rnn_size $size \
			 #~ -brnn \
			 #~ -brnn_merge concat \
			 #~ -word_vec_size $size \
			 #~ -input_feed 1 \
			 #~ -dropout 0.2 \
			 #~ -optim $method \
			 #~ -learning_rate 0.001 \
			 #~ -max_batch_size 128 \
			 #~ -start_epoch 6 \
			 #~ -end_epoch 15 \
			 #~ -start_decay_at 9 \
			 #~ -learning_rate_decay 0.7 \
			 #~ -train_from $MODELDIR/model.lstm.adam.1024_epoch5_3.37.t7 \
			 #~ -continue  \
			 #~ -gpuid 1 2>&1 | tee $LOGDIR/$type.$method.$size.log
			

