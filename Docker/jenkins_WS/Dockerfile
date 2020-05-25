FROM registry.access.redhat.com/ubi8/ubi:latest

RUN echo -e "[docker]\nname=docker\nbaseurl=https://download.docker.com/linux/centos/7/x86_64/stable/\ngpgcheck=0\n[baseos]\nname=baseos\nbaseurl=http://mirror.centos.org/centos-8/8/BaseOS/x86_64/os/\ngpgcheck=0\[appstream]\nname=appstream\nbaseurl=http://mirror.centos.org/centos-8/8/AppStream/x86_64/os/\ngpgcheck=0">>/etc/yum.repos.d/docker.repo && \ 
yum repolist && \ 
yum install -y docker-ce --nobest -y 

RUN yum install -y wget && \ 
wget -O /etc/yum.repos.d/jenkins.repo https://pkg.jenkins.io/redhat-stable/jenkins.repo && \
rpm --import https://pkg.jenkins.io/redhat-stable/jenkins.io.key && \
yum install -y jenkins
RUN yum install -y java-11-openjdk.x86_64 && \
yum clean all && \
echo "jenkins ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN yum install git -y && \
yum install bc -y

EXPOSE 8080 
EXPOSE 80

CMD java -jar /usr/lib/jenkins/jenkins.war
