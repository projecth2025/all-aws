# IMPORTANT: Amazon Linux 2 based Lambda image
FROM public.ecr.aws/lambda/python:3.10-x86_64

# ---------------- System dependencies required by LibreOffice ----------------
RUN yum install -y \
    wget \
    tar \
    gzip \
    fontconfig \
    freetype \
    libSM \
    libICE \
    libXinerama \
    libXrandr \
    libXcursor \
    libXdamage \
    libXcomposite \
    libXfixes \
    libXrender \
    libXt \
    libXext \
    cups-libs \
    dbus-libs \
    nss \
    atk \
    at-spi2-atk \
    gtk3 \
    mesa-libEGL \
    mesa-libGL \
    liberation-fonts \
    dejavu-sans-fonts \
    dejavu-serif-fonts \
    dejavu-sans-mono-fonts \
    && yum clean all \
    && rm -rf /var/cache/yum

# ---------------- Install LibreOffice 7.6.5.2 from ARCHIVE ----------------
ENV LO_VERSION=7.6.5.2

RUN cd /tmp && \
    wget https://downloadarchive.documentfoundation.org/libreoffice/old/${LO_VERSION}/rpm/x86_64/LibreOffice_${LO_VERSION}_Linux_x86-64_rpm.tar.gz && \
    tar -xzf LibreOffice_${LO_VERSION}_Linux_x86-64_rpm.tar.gz && \
    yum install -y LibreOffice_${LO_VERSION}_Linux_x86-64_rpm/RPMS/*.rpm && \
    rm -rf /tmp/*

# ---------------- Stable soffice path (CRITICAL) ----------------
RUN ln -s /opt/libreoffice*/program/soffice /usr/bin/soffice

# ---------------- Python dependencies ----------------
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ---------------- Lambda function code ----------------
COPY lambda_function.py .

# ---------------- Lambda entrypoint ----------------
CMD ["lambda_function.lambda_handler"]