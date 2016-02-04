.PHONY : clean

SOURCES = avg_filter.cu

avg_filter	:	$(SOURCES) Makefile
				nvcc --compiler-options -Wall $(SOURCES) -o $@ -lopencv_core -lopencv_highgui
clean		:
			rm -f avg_filter
