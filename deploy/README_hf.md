---
title: EyeSpot
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
short_description: Real-time object detection and tracking with YOLOv8
---

# 🎯 EyeSpot — Real-Time Object Detection & Tracking

Real-time object detection and tracking built on **YOLOv8 (Ultralytics)** and
**OpenCV**. Upload an image or a video and EyeSpot will find and label objects
across all 80 COCO classes — person, car, dog, cat, bird, laptop, phone, and more.

## Try it

- **🖼️ Image tab** — upload a photo and see detections instantly
- **🎬 Video tab** — upload a clip; you get an annotated video, a summary of
  everything found, and downloads for both the video and a CSV detection log

Use the sidebar to change the model, adjust the confidence threshold, filter to
specific classes, or turn on **object tracking**.

## Object tracking

With tracking enabled, every object receives a **persistent ID** that stays the
same from frame to frame. That is what makes it possible to answer questions a
plain detector cannot:

- how many **unique** people or vehicles appeared, rather than a per-frame count
- how long each one stayed in view
- what route it took through the scene

## What it detects

All 80 COCO classes, with no training required. Group shortcuts make filtering
easy: `people`, `animals`, `pets`, `vehicles`, `road`.

It reports the **class** of an object — "person", "bird", "car". It does not do
face recognition, bird species, or licence plates; those need custom training.

## Note on the live camera tab

The live camera tab cannot work in a hosted demo. This server has no webcam, and
it cannot reach a camera on your home network — `192.168.x.x` addresses are
private and do not exist on the public internet.

Running EyeSpot on your own machine unlocks the rest of it: a laptop webcam, a
phone used as an IP camera over HTTP, or a CCTV/RTSP stream, plus a target-lock
mode that follows one chosen object with a trail and automatic re-acquisition
when it disappears and comes back.

## Performance

`yolov8n.pt` at 640 px reaches roughly **55 FPS** inference on a modern CPU
(18 ms per frame). This Space runs on shared CPU hardware, so expect it to be
slower here than on your own machine.

## Credits

Detection models: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
