FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN echo "==> Upgrading apk and installing system utilities ...." \
 && apt -y update \
 && apt-get install -y wget \
 && apt-get -y install sudo

RUN echo "==> Installing Python3 and pip ...." \  
 && apt-get install python3 -y \
 && apt install python3-pip -y \
 && apt install openssh-client -y

RUN echo "==> Install dos2unix..." \
  && sudo apt-get install dos2unix -y 

RUN echo "==> Install requirements.." \
  && pip install --break-system-packages -U --quiet langchain_community langchain-openai \
  && pip install --break-system-packages streamlit --upgrade

RUN echo "==> Install openai.." \
  && pip install --break-system-packages openai

COPY /SpaceAce /SpaceAce/
COPY /scripts /scripts/

RUN echo "==> Convert script..." \
  && dos2unix /scripts/startup.sh

CMD ["/bin/bash", "/scripts/startup.sh"]