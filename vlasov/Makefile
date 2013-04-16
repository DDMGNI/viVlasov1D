

all:
	python setup.py build_ext --inplace
	
	$(MAKE) -C predictor
	$(MAKE) -C vi
	

clean:
	rm -f *.c
	rm -f *.so
	rm -fR build
	
	$(MAKE) clean -C predictor
	$(MAKE) clean -C vi
	