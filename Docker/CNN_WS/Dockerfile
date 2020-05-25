FROM registry.access.redhat.com/ubi8/ubi:latest 

LABEL io.openshift.tag=machinelearning

#COPY requirements.txt /tmp/requirements.txt

RUN yum install python3 -y && \
pip3 install matplotlib numpy

RUN python3 -m pip --no-cache-dir install --upgrade \
pip \
setuptools

RUN pip3 install tensorflow
RUN pip3 install keras

#COPY train.py /tmp/train.py

CMD ["python3","/tmp/train.py"]


EXPOSE 80 
EXPOSE 443
