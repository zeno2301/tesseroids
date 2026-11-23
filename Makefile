VERSION := $(shell python print_version.py 2>/dev/null || python3 print_version.py 2>/dev/null || git describe --tags --always 2>/dev/null || echo unknown)
DIST := dist
PKGNAME := tesseroids-$(VERSION)
PKG := $(DIST)/$(PKGNAME)

# Prism field components (program suffixes)
PRISM_FIELDS := pot gx gy gz gxx gxy gxz gyy gyz gzz

# C sources shared by the CPU and CUDA builds
LIBSRC := src/lib/logger.c src/lib/version.c src/lib/constants.c \
          src/lib/geometry.c src/lib/parsers.c

# CUDA build settings (override on the command line if needed, e.g.
#   make build-cuda NVCCARCH="-arch=native" CUDA_PATH=/opt/cuda)
NVCC ?= nvcc
CUDA_PATH ?= /usr/local/cuda
NVCCFLAGS ?= -O3 -I$(CUDA_PATH)/include
# PTX for compute_60 JITs on any GPU sm_60+, plus native SASS for sm_75
NVCCARCH ?= -gencode arch=compute_60,code=compute_60 -gencode arch=compute_75,code=sm_75
# nvcc не принимает -Wl,... напрямую: опции линкёра передаём через -Xlinker
CUDALIBS := -L$(CUDA_PATH)/lib64 -lcudart -Xlinker -rpath -Xlinker $(CUDA_PATH)/lib64
CUDA_CUSRC := src/lib/grav_prism_cuda.cu src/lib/prismg_main_cuda.cu
CUDA_HDRS := src/lib/grav_prism_cuda.h src/lib/grav_prism_kernels.cuh \
             src/lib/prismg_main_cuda.h

.PHONY: help build check test build-cpu build-cuda test-cpu package pkg-bin32 pkg-bin64 pkg-win32 pkg-win64 pkg-src clean

help:
	@echo "Commands:"
	@echo "  build        Compile the source (scons)"
	@echo "  build-cpu    Compile the CPU prism* programs with gcc into bin/"
	@echo "  build-cuda   Compile the CUDA prism* programs with nvcc into bin-cuda/"
	@echo "  test         Run the tests (scons build)"
	@echo "  test-cpu     Build and run the unit tests with gcc"
	@echo "  check        Compile in debug mode and check for ANSI compatibility"
	@echo "  clean        Clean the directory"
	@echo "  package      Make binary packages for all supported platforms"
	@echo "  pkg-bin32    Make a 32bit linux binary distribution"
	@echo "  pkg-bin64    Make a 64bit linux binary distribution"
	@echo "  pkg-win32    Cross-compile 32bit Windows binary distribution"
	@echo "  pkg-win64    Cross-compile 64bit Windows binary distribution"
	@echo "  pkg-src      Make a source distribution"

build:
	scons

check:
	scons mode=check

test: build
	./tesstest

# Generate src/lib/version.c from the template
src/lib/version.c: src/lib/version.template
	sed -e 's/$$VERSION/$(VERSION)/' -e 's/$$YEAR/$(shell date +%Y)/' $< > $@

# CPU versions of the prism* programs (plain gcc, no scons needed)
build-cpu: src/lib/version.c $(PRISM_FIELDS:%=bin/prism%)

bin/prism%: src/prism%.c src/lib/prismg_main.c src/lib/grav_prism.c $(LIBSRC)
	mkdir -p bin
	gcc -O3 -Isrc/lib -o $@ $< src/lib/prismg_main.c src/lib/grav_prism.c $(LIBSRC) -lm

# CUDA versions of the prism* programs (need nvcc and an NVIDIA GPU to run)
build-cuda: src/lib/version.c $(PRISM_FIELDS:%=bin-cuda/prism%)

bin-cuda/prism%: src/cuda/prism%.c $(CUDA_CUSRC) $(CUDA_HDRS) $(LIBSRC)
	mkdir -p bin-cuda
	$(NVCC) $(NVCCFLAGS) $(NVCCARCH) -Isrc/lib -o $@ $< $(CUDA_CUSRC) $(LIBSRC) $(CUDALIBS)

# Unit tests with plain gcc (includes the CUDA device-code validation tests)
test-cpu: src/lib/version.c tesstest
	./tesstest

tesstest: test/test_all.c $(LIBSRC) src/lib/grav_prism.c src/lib/grav_prism_sph.c \
          src/lib/grav_tess.c src/lib/glq.c src/lib/grav_sphere.c
	gcc -O2 -Isrc/lib -o $@ test/test_all.c src/lib/*.c -lm

package: pkg-bin32 pkg-bin64 pkg-win32 pkg-win64

pkg-src:
	@echo "\nCreating source package\n"
	@echo "#######################\n"
	bash clean.sh
	mkdir -p $(PKG)
	cp -r src $(PKG)
	cp -r test $(PKG)
	cp -r cookbook $(PKG)
	cp print_version.py $(PKG)
	cp versioneer.py $(PKG)
	cp README.md $(PKG)/README.md
	cp LICENSE.txt $(PKG)
	cp DEPENDENCIES.txt $(PKG)
	cp CITATION.txt $(PKG)
	cp SConstruct $(PKG)
	cd $(DIST); zip -r $(PKGNAME)-src.zip $(PKGNAME); cd ..
	cd $(DIST); tar -zcvf $(PKGNAME)-src.tar.gz $(PKGNAME); cd ..
	rm -rf $(PKG)

pkg-bin32:
	@echo "\nCreating 32bit Linux package"
	@echo "############################\n"
	bash clean.sh
	scons mode=bin32
	./tesstest
	mkdir -p $(PKG)
	cp -r cookbook $(PKG)
	cp -r bin $(PKG)
	cp README.md $(PKG)/README.md
	cp LICENSE.txt $(PKG)
	cp DEPENDENCIES.txt $(PKG)
	cp CITATION.txt $(PKG)
	cd $(DIST); tar -zcvf $(PKGNAME)-bin32.tar.gz $(PKGNAME); cd ..
	rm -r $(PKG)

pkg-bin64:
	@echo "\nCreating 64bit Linux package"
	@echo "############################\n"
	bash clean.sh
	scons
	./tesstest
	mkdir -p $(PKG)
	cp -r cookbook $(PKG)
	cp -r bin $(PKG)
	cp README.md $(PKG)/README.md
	cp LICENSE.txt $(PKG)
	cp DEPENDENCIES.txt $(PKG)
	cp CITATION.txt $(PKG)
	cd $(DIST); tar -zcvf $(PKGNAME)-bin64.tar.gz $(PKGNAME); cd ..
	rm -r $(PKG)

pkg-win32:
	@echo "\nCreating 32bit Windows package"
	@echo "##############################\n"
	bash clean.sh
	scons mode=win32
	# Need to clear the wine install or it will complain when running both
	# 32bit and 64bit
	rm -r ~/.wine
	wine tesstest.exe
	mkdir $(PKG)
	cp -r cookbook $(PKG)
	cp -r bin $(PKG)
	cp README.md $(PKG)/README.md
	cp LICENSE.txt $(PKG)
	cp DEPENDENCIES.txt $(PKG)
	cp CITATION.txt $(PKG)
	cd $(DIST); zip -r $(PKGNAME)-win32.zip $(PKGNAME); cd ..
	rm -r $(PKG)

pkg-win64:
	@echo "\nCreating 64bit Windows package"
	@echo "##############################\n"
	bash clean.sh
	scons mode=win64
	# Need to clear the wine install or it will complain when running both
	# 32bit and 64bit
	rm -r ~/.wine
	wine64 tesstest.exe
	mkdir $(PKG)
	cp -r cookbook $(PKG)
	cp -r bin $(PKG)
	cp README.md $(PKG)/README.md
	cp LICENSE.txt $(PKG)
	cp DEPENDENCIES.txt $(PKG)
	cp CITATION.txt $(PKG)
	cd $(DIST); zip -r $(PKGNAME)-win64.zip $(PKGNAME); cd ..
	rm -r $(PKG)

clean:
	bash clean.sh
