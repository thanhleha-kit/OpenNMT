#!/bin/sh


src=en
tgt=de
DATADIR=$HOME/workspace/Torch7/OpenNMT/data/TED.EN-DE
SAVEDIR=saves/TED/$src-$tgt

mkdir -p $SAVEDIR

th preprocess.lua -train_src $DATADIR/corpus.bpe.filtered.$src      -train_tgt $DATADIR/corpus.bpe.filtered.$tgt          \
	              -valid_src $DATADIR/dev.bpe.filtered.$src  -valid_tgt $DATADIR/dev.bpe.filtered.$tgt     \
	              -src_seq_length 80 -tgt_seq_length 80 \
	              -save_data $SAVEDIR/tensor




