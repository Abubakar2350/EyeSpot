# Dockerfile for the EyeSpot Hugging Face Space.
#
# Hugging Face removed the native Streamlit SDK, so Streamlit apps now ship as
# Docker Spaces. The container installs the dependencies and serves the app on
# port 7860, which is the port Spaces expects.

FROM python:3.11-slim

# ffmpeg powers the H.264 transcode that makes result videos playable in the
# browser. OpenCV is the headless build, so no display libraries are needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000, so build for that user from the start
RUN useradd -m -u 1000 user
WORKDIR /app
RUN chown -R user:user /app
USER user

ENV PATH="/home/user/.local/bin:$PATH"
ENV HOME=/home/user
ENV PYTHONUNBUFFERED=1

# Ultralytics and matplotlib both need a writable config directory
ENV YOLO_CONFIG_DIR=/home/user/.config/Ultralytics
ENV MPLCONFIGDIR=/home/user/.cache/matplotlib

# Tells the app it is hosted, so the live camera tab explains itself instead of
# failing to reach a camera it could never see from here
ENV EYESPOT_CLOUD=1
ENV STREAMLIT_SERVER_HEADLESS=true

# Dependencies first, so code changes do not invalidate the install layer
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

EXPOSE 7860

CMD ["streamlit", "run", "ui/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0"]
