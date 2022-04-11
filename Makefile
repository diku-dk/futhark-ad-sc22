figure_6:
	nix-shell --pure --run ./setup-adbench.sh
	nix-shell --pure --run ./run-adbench.sh
	python3 scripts/figure_6.py

figure_7: tmp/xsbench-original.txt tmp/rsbench-original.txt tmp/lbm-original.txt tmp/xsbench-futhark.json tmp/rsbench-futhark.json tmp/lbm-futhark.json
	python3 scripts/figure_7.py

figure_8:
	cd benchmarks/kmeans/standard && make results && make table

figure_9:
	cd benchmarks/kmeans/sparse && make results && make table

figure_11:
	cd benchmarks/gmm && make results && make table

figure_12:
	cd benchmarks/lstm && make results && make table

tmp/xsbench-original.txt:
	mkdir -p tmp
	make -C originals/xsbench
	(cd originals/xsbench && \
	  for i in $$(seq 10); do ./XSBench -s small -m event | grep Runtime | tail -n 1; done) \
          | awk '{print $$2}' | tee $@ || rm -f $@

tmp/rsbench-original.txt:
	mkdir -p tmp
	make -C originals/rsbench
	(cd originals/rsbench && \
	  for i in $$(seq 10); do ./rsbench -s small -m event | grep Runtime | tail -n 1; done) \
          | awk '{print $$2}' | tee $@ || rm -f $@

tmp/lbm-original.txt:
	mkdir -p tmp
	(for i in $$(seq 10); do make -C originals/lbm run; done) \
          | awk '/Kernel/ {print $$3}' | tee $@ || rm -f $@

tmp/xsbench-futhark.json:
	bin/futhark bench --backend cuda benchmarks/xsbench/xsbench.fut --json $@

tmp/rsbench-futhark.json:
	bin/futhark bench --backend cuda benchmarks/rsbench/rsbench.fut --json $@

tmp/lbm-futhark.json:
	bin/futhark bench --backend cuda benchmarks/lbm/lbm.fut --json $@

bin/futhark:
	cd futhark && nix-build --argstr suffix ad
	mkdir -p bin
	tar --extract -C bin/ --strip-components 2 -f futhark/result/futhark-ad.tar.xz futhark-ad/bin/futhark
