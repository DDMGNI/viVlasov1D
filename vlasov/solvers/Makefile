

all:
	$(MAKE) -C components
	$(MAKE) -C explicit
	$(MAKE) -C full
	$(MAKE) -C poisson
	$(MAKE) -C preconditioner
	$(MAKE) -C vlasov
	

clean:
	rm -f *.c
	rm -f *.so
	rm -fR build
	
	$(MAKE) clean -C components
	$(MAKE) clean -C explicit
	$(MAKE) clean -C full
	$(MAKE) clean -C poisson
	$(MAKE) clean -C preconditioner
	$(MAKE) clean -C vlasov
	