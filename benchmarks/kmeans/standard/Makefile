include ../../common.mk
NAME=kmeans
JAC_SPEEDUP=False
OBJ_SPEEDUP=True
FUTHARK_BENCH_OPTIONS=
ifeq ($(GPU), A100)
        JAX=TRUE
endif


.PHONY: results data

results: data kmeans.json kmeans_pytorch.json kmeans_manual.json $(if $(JAX),kmeans_jax.json)
	make gen_results

data:
	chmod +x mkdata.sh
	./mkdata.sh

