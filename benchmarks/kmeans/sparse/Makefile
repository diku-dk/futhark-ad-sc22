include ../../common.mk
NAME=kmeans_sparse
JAC_SPEEDUP=False
OBJ_SPEEDUP=True
FUTHARK_BENCH_OPTIONS=

.PHONY: results

results: kmeans_sparse.json kmeans_sparse_pytorch.json kmeans_sparse_manual.json $(ifneq ($(GPU), MI100), kmean_sparse_jax.json)
	make gen_results

