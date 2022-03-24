include ../../common.mk
NAME=gmm
ifeq ($(GPU), A100)
	FUTHARK_TUNING=--tuning=.tuning.a100
	FUTHARK_TILE_SIZE=16
	FUTHARK_REG_TILE=4
        PRECISION=f64
else  
ifeq ($(GPU), 2080TI)
	FUTHARK_TUNING=--tuning=.tuning.2080ti
	FUTHARK_TILE_SIZE=16
	FUTHARK_REG_TILE=4
else
        FUTHARK_BENCH_OPTIONS=
endif
endif


$(RESULTS): data lstm.json lstm_pytorch.json lstm_jax.json
	make gen_results

data:
	chmod +x mkdata.sh
	./mkdata.sh $(ADBench) $(PRECISION)
