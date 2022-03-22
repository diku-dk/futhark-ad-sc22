include ../../common.mk
NAME=kmeans
JAC_SPEEDUP=False
OBJ_SPEEDUP=True

ifeq ($(shell if lshw -C display 2>/dev/null | grep -q 'A100'; then echo yes; fi),yes)
	FUTHARK_TUNING=--tuning=.tuning
	FUTHARK_TILE_SIZE=4
	FUTHARK_REG_TILE=4
        PRECISION=f32
else
	FUTHARK_TUNING=--tuning=.tuning
	FUTHARK_TILE_SIZE=4
	FUTHARK_REG_TILE=4
        PRECISION=f32
endif

.PHONY: results
results: kmeans_sparse.json kmeans_sparse_pytorch.json kmeans_sparse_manual.json kmeans_sparse_jax.json
	make gen_results
