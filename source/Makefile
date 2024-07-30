target: x

x:
	number=10 ; while [[ $$number -le 19 ]] ; do \
        echo $$number ; \
		python main.py -i $$number; \
		((number = number + 1)) ; \
    done
	 