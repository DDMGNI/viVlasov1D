

all:
	$(MAKE) -C components
	$(MAKE) -C explicit
	$(MAKE) -C full
	$(MAKE) -C poisson
	$(MAKE) -C vlasov
	$(MAKE) -C preconditioner
	

clean:
	rm -f *.c
	rm -f *.so
	rm -fR build
	
	$(MAKE) clean -C components
	$(MAKE) clean -C explicit
	$(MAKE) clean -C full
	$(MAKE) clean -C poisson
	$(MAKE) clean -C vlasov
	$(MAKE) clean -C preconditioner
	