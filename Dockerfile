FROM ghcr.io/fenics/test-env@sha256:270f275e5232dc73373cde73015de9f978a072b6583284c1a9e80b2701dc6464
RUN apt update
RUN fetch_repo() { \
    repo="$1"; repo_url="$2"; sha="$3"; \
    mkdir $repo && \
    cd $repo && \
    git init && \
    git remote add origin $repo_url && \
    git fetch --depth=1 origin $sha && \
    git checkout FETCH_HEAD && \
    cd ..; \
    }; \
    fetch_repo basix https://github.com/FEniCS/basix eff3bb56cf80ba9507bc0495e371d5efb80924d1 && \
    fetch_repo ufl https://github.com/FEniCS/ufl 6e94c2a2cbb4c0c915ed15bca309bdb91cc1b318 && \
    fetch_repo ffcx https://github.com/FEniCS/ffcx 3a7171798e91c014a9973ddfa6b05b169f479288 && \
    fetch_repo dolfinx https://github.com/ordinary-slim/dolfinx 825937b11f76a7f5094e2214d56cf5631194b20a && \
    fetch_repo multiphenicsx https://github.com/ordinary-slim/multiphenicsx 8fc2e36429d8e070ade5ae2562c606ca26c12bf3 && \
    fetch_repo mhs_fenicsx https://github.com/ordinary-slim/mhs_fenicsx ac2a99efc87e3255536ddb090f7ef7aaed2534e0

# env vars building
ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
ENV CC="gcc"
ENV CXX="c++"
ENV DOLFINX_MODE=real
ENV DOLFINX_PRECISION=64
ENV PETSC_INT_SIZE=32
ENV PETSC_ARCH=linux-gnu-${DOLFINX_MODE}${DOLFINX_PRECISION}-${PETSC_INT_SIZE}
ENV LD_LIBRARY_PATH="/usr/local/petsc/${PETSC_ARCH}/lib/:${DOLFINX_DIR}/lib/:$LD_LIBRARY_PATH"
ENV PYVISTA_OFF_SCREEN=True
ENV DOLFINX_CMAKE_CXX_FLAGS="-march=native"
ENV DOLFINX_CMAKE_BUILD_TYPE="RELEASE"
ENV PATH=$PATH:/root/.local/bin
ENV CMAKE_NUM_PROCESSES=4
ENV CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_NUM_PROCESSES}
ENV NINJAFLAGS="-j${CMAKE_NUM_PROCESSES}"

# Build basix
RUN cd basix && \
    build_dir="build-dir-${DOLFINX_CMAKE_BUILD_TYPE}" && \
    CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_NUM_PROCESSES} cmake -G Ninja -B ${build_dir} -DCMAKE_BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} -S ./cpp/ && \
    CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_NUM_PROCESSES} cmake --build ${build_dir} --parallel 3 && \
    CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_NUM_PROCESSES} cmake --install ${build_dir} && \
    python3 -m pip -v install nanobind "scikit-build-core==0.10.0" pyproject_metadata pathspec && \
    python3 -m pip -v install --check-build-dependencies --config-settings=install.strip=false--config-settings=build-dir=${build_dir} --config-settings=cmake.build-type="${DOLFINX_CMAKE_BUILD_TYPE}" --config-settings=cmake.define.CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_NUM_PROCESSES}  --no-build-isolation ./python && \
    cd ..

# Build ufl
RUN cd ufl && \
    python3 -m pip install -e . -v --upgrade

# Build ffcx
RUN cd ffcx && \
    python3 -m pip install -e . -v --upgrade

# Build dolfinx cpp
RUN cd dolfinx && \
    BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} && \
    CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_NUM_PROCESSES} && \
    cmake -G Ninja -B build-dir-${DOLFINX_MODE}${DOLFINX_PRECISION} -DCMAKE_INSTALL_PREFIX=${DOLFINX_DIR} -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_CXX_FLAGS="${DOLFINX_CMAKE_CXX_FLAGS}" -S cpp CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_NUM_PROCESSES} && \
    cmake --build build-dir-${DOLFINX_MODE}${DOLFINX_PRECISION} && \
    cmake --install build-dir-${DOLFINX_MODE}${DOLFINX_PRECISION} && \
    cd ..

# Build dolfinx python
# The build command is repeated twice because of a weird race condition bug
# First compile at some point recruits all the processors and crashes at a file
# Second compile completes
RUN cd dolfinx && \
    build_command="python3 -m pip -v install --check-build-dependencies --config-settings=build-dir="build-${DOLFINX_CMAKE_BUILD_TYPE}-${DOLFINX_MODE}-${DOLFINX_PRECISION}" --config-settings=install.strip=false --config-settings=cmake.build-type="${DOLFINX_CMAKE_BUILD_TYPE}" --no-build-isolation -e ./python/" && \
    eval ${build_command} || \
    eval ${build_command}

# Build multiphenicsx
RUN cd multiphenicsx && \
    build_dir="build-dir-${DOLFINX_CMAKE_BUILD_TYPE}" && \
    python3 -m pip install --check-build-dependencies --no-build-isolation --config-settings=cmake.build-type="${DOLFINX_CMAKE_BUILD_TYPE}" --config-settings=build-dir=${build_dir} --verbose -e '.' && \
    cmake --install ${build_dir} --prefix=/usr/local && \
    cd ..

# Build mhs_fenicsx
RUN cd mhs_fenicsx && \
    build_dir="build-dir-${DOLFINX_CMAKE_BUILD_TYPE}" && \
    cmake -B ${build_dir} -DCMAKE_BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} -S ./cpp/ && \
    cmake --build ${build_dir} && \
    cmake --install ${build_dir} && \
    python3 -m pip -v install -e python --config-settings=build-dir=${build_dir} --no-build-isolation && \
    cd ..

# Install dependencies
RUN python3 -m pip install numba line_profiler pytest pyyaml
