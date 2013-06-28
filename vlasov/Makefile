

all:
	python setup.py build_ext --inplace
	
	$(MAKE) -C core
	$(MAKE) -C explicit
	$(MAKE) -C predictor
	$(MAKE) -C solvers
	

clean:
	rm -f *.c
	rm -f *.so
	rm -fR build
	
	$(MAKE) clean -C core
	$(MAKE) clean -C explicit
	$(MAKE) clean -C predictor
	$(MAKE) clean -C solvers
	