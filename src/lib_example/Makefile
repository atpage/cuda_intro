.PHONY : lib, clean, install, uninstall, FORCE

LIBNAME = libhypot.so
INSTALLDIR = /usr/local/lib/

lib			:	$(LIBNAME)

$(LIBNAME)		:	hypot.cu
					nvcc -Xcompiler -fPIC -shared -o $@ $^

install			:	
					make lib  # TODO: as user not root
					cp $(LIBNAME) $(INSTALLDIR)
					ldconfig

uninstall		:	
					rm $(INSTALLDIR)$(LIBNAME)

clean			:	
					rm -f *.a *.o *.so

FORCE			:
