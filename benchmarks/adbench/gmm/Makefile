include ../../common.mk
NAME=gmm

ifeq ($(shell if lshw -C display 2>/dev/null | grep -q 'A100'; then echo yes; fi),yes)
	FUTHARK_TUNING=--tuning=.tuning.a100
	FUTHARK_TILE_SIZE=16
	FUTHARK_REG_TILE=3
        PRECISION=f64
else
	FUTHARK_TUNING=--tuning=.tuning.2080ti
	FUTHARK_TILE_SIZE=8
	FUTHARK_REG_TILE=4
        PRECISION=f32
endif

.PHONY: results
results: data gmm.json gmm_pytorch.json
	make gen_results

data:
	chmod +x mkdata.sh
	./mkdata.sh $(ADBench) $(PRECISION)
