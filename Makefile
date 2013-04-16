
PYTHONPATH := $(CURDIR):${PYTHONPATH}
export PYTHONPATH

all:
	$(MAKE) -C vlasov
	

clean:
	$(MAKE) clean -C vlasov
