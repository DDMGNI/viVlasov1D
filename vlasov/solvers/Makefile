

all:
	$(MAKE) -C full
	$(MAKE) -C poisson
	$(MAKE) -C vlasov
	

clean:
	rm -f *.c
	rm -f *.so
	rm -fR build
	
	$(MAKE) clean -C full
	$(MAKE) clean -C poisson
	$(MAKE) clean -C vlasov
	