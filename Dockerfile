FROM brmather/pycurious-base:latest

# add pycurious
WORKDIR /work
ENV MODULE_DIR="pycurious-src"
RUN addgroup -S jovyan && \
    adduser -D -S jovyan -G jovyan

ADD --chown=jovyan:jovyan . / pycurious-src/
RUN cd pycurious-src && \
    python3 -m pip install --no-deps --upgrade . && \
    mv pycurious/Examples/data /home/jovyan/ && \
    mv pycurious/Examples/Notebooks /home/jovyan/ && \
    mv pycurious/Examples/Scripts /home/jovyan/ && \
    cd /work && \
    rm -rf /work/pycurious-src


# change ownership of everything
ENV NB_USER jovyan
RUN chown -R jovyan:jovyan /home/jovyan
USER jovyan
WORKDIR /home/jovyan/
RUN find -name \*.ipynb  -print0 | xargs -0 jupyter trust


# launch notebook
ENTRYPOINT ["jupyter", "notebook", "--ip='0.0.0.0'", "--no-browser"]