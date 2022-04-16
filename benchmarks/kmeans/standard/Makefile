include ../../common.mk
NAME=kmeans
JAC_SPEEDUP=False
OBJ_SPEEDUP=True
FUTHARK_BENCH_OPTIONS=

.PHONY: results data

results: data kmeans.json kmeans_pytorch.json kmeans_manual.json $(ifneq ($(GPU), MI100), kmean_jax.json)
	make gen_results

data:
	chmod +x mkdata.sh
	./mkdata.sh

