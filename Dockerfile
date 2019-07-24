#################################################
#  Short docker file to distribute some notebooks
#################################################

ARG FROMIMG_ARG=brmather/pycurious:1.0
FROM ${FROMIMG_ARG}

USER jovyan

WORKDIR /home/jovyan

# Trust all notebooks
RUN find -name \*.ipynb  -print0 | xargs -0 jupyter trust

# expose notebook port server port
EXPOSE 8888

VOLUME /home/jovyan/$NB_DIR/user_data


ENTRYPOINT ["/sbin/tini", "--"]

# launch notebook
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--no-browser", "--NotebookApp.token='' ", "--NotebookApp.default_url=/tree/0-StartHere.ipynb"]
