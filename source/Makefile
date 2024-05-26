target: x

x:
	number=0 ; while [[ $$number -le 9 ]] ; do \
        echo $$number ; \
		python main.py -i $$number; \
		((number = number + 1)) ; \
    done
	