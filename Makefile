figure_9:
	nix-shell --pure --run ./setup-adbench.sh
	nix-shell --pure --run ./run-adbench.sh
	python3 scripts/figure_9.py

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

tmp/xsbench-futhark.json: bin/futhark
	bin/futhark bench --backend opencl benchmarks/xsbench/xsbench.fut --json $@

tmp/rsbench-futhark.json: bin/futhark
	bin/futhark bench --backend opencl benchmarks/rsbench/rsbench.fut --json $@

figure_10: tmp/xsbench-original.txt tmp/rsbench-original.txt tmp/xsbench-futhark.json tmp/rsbench-futhark.json
	false

bin/futhark:
	cd futhark && nix-build --argstr suffix ad
	mkdir -p bin
	tar --extract -C bin/ --strip-components 2 -f futhark/result/futhark-ad.tar.xz futhark-ad/bin/futhark
