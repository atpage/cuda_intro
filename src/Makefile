.PHONY : clean

SOURCES = avg_filter.cu
HEADERS = helper_cuda.h helper_string.h

avg_filter	:	$(SOURCES) $(HEADERS) Makefile
				nvcc --compiler-options -Wall $(SOURCES) -o $@ -lopencv_core -lopencv_highgui
clean		:
			rm -f avg_filter
