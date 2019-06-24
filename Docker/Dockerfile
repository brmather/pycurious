#################################################
#  Short docker file to distribute some notebooks
#################################################

ARG FROMIMG_ARG=brmather/pycurious-base:0.9.1
FROM ${FROMIMG_ARG}

##################################################
# Non standard as the files come from the packages

USER root
WORKDIR /home/jovyan/

RUN apk add \
    libxml2-dev \
    libxslt-dev 

RUN python3 -m pip install --no-cache-dir --upgrade \
    jupyter \
    pytest

### PyCurious - Notebooks

ENV MODULE_DIR="pycurious-src"
ADD --chown=jovyan:jovyan . / pycurious-src/
RUN cd $MODULE_DIR && python3 -m pip install --no-cache-dir --upgrade --no-deps .

RUN ipython3 -c 'import pycurious; pycurious.documentation.install_documentation(path="Notebooks")'


# change ownership of everything
ENV NB_USER jovyan
RUN chown -R jovyan:jovyan /home/jovyan
USER jovyan


# Run the tests with PyTest
RUN cd $MODULE_DIR && pytest tests/


## These are supplied by the build script
## build-dockerfile.sh

ARG IMAGENAME_ARG
ARG PROJ_NAME_ARG=pycurious
ARG NB_PORT_ARG=8888
ARG NB_PASSWD_ARG=""
ARG NB_DIR_ARG="Notebooks"
ARG START_NB_ARG="0-StartHere.ipynb"

# The args need to go into the environment so they
# can be picked up by commands/templates (defined previously)
# when the container runs

ENV IMAGENAME=$IMAGENAME_ARG
ENV PROJ_NAME=$PROJ_NAME_ARG
ENV NB_PORT=$NB_PORT_ARG
ENV NB_PASSWD=$NB_PASSWD_ARG
ENV NB_DIR=$NB_DIR_ARG
ENV START_NB=$START_NB_ARG


# Trust all notebooks
RUN find -name \*.ipynb  -print0 | xargs -0 jupyter trust

# expose notebook port server port
EXPOSE $NB_PORT

VOLUME /home/jovyan/$NB_DIR/user_data


ENTRYPOINT ["/sbin/tini", "--"]

# launch notebook
ADD --chown=jovyan:jovyan Docker/scripts/run-jupyter.sh scripts/run-jupyter.sh
CMD scripts/run-jupyter.sh