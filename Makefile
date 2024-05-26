target: x

x:
	number=0 ; while [[ $$number -le 19 ]] ; do \
        echo $$number ; \
		python source/main.py -i $$number; \
		((number = number + 1)) ; \
    done
	