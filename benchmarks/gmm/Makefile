include ../common.mk
NAME=gmm
CONVERT=true
ifeq ($(GPU), A100)
	FUTHARK_TUNING=--tuning=.tuning.a100
	FUTHARK_TILE_SIZE=16
	FUTHARK_REG_TILE=3
        PRECISION=f64
else ifeq ($(GPU), MI100)
	FUTHARK_TUNING=--tuning=.tuning.mi100
	FUTHARK_TILE_SIZE=16
	FUTHARK_REG_TILE=3
        PRECISION=f64
else
        FUTHARK_BENCH_OPTIONS=
endif

.PHONY: results

results: gmm.json gmm_pytorch.json
	make gen_results
