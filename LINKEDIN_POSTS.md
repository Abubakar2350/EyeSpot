# LinkedIn Posts — EyeSpot

Three posts, each pairing with one demo video you record. Copy the text between
the rules, attach your clip, publish.

> **Before posting, read section 4.** Both links below are currently private, so
> anyone clicking them today sees a login wall or a 404. That takes about
> 30 seconds to fix.

---

## Post 1 — Image detection

**Pair with:** a short clip of dragging a photo into the app and the boxes
appearing. A busy street photo shows it off better than a simple one.

---

I built a real-time object detection system. Here it is finding people and vehicles in a single photo — in 18 milliseconds.

EyeSpot runs YOLOv8 on any image you give it and labels every object it recognises, with a confidence score on each box.

A few things I learned building it:

→ 80 object classes come free. The pretrained COCO model already knows person, car, dog, cat, bird, laptop, phone, bottle, chair, and 71 more. No training required to get started.

→ "Animal" is not a class. COCO has bird, cat, dog, horse, sheep, cow, elephant, bear, zebra and giraffe — but no single "animal" label. I added group shortcuts so `--classes animals` expands to all ten. Small detail, big difference in usability.

→ Confidence thresholds matter more than model size. Dropping from 0.5 to 0.3 finds small objects the default misses, at the cost of more false positives. Making that adjustable at runtime turned out to be the most-used feature I built.

Speed on a plain CPU, no GPU: 55 FPS at 640px inference. The target was 15.

Next post: what happens when you point it at video, and why detection alone isn't enough.

🔗 Try it: [YOUR APP LINK]

#ComputerVision #YOLO #Python #OpenCV #MachineLearning #AI #DeepLearning #ObjectDetection #Streamlit #BuildInPublic

---

## Post 2 — Video and object tracking

**Pair with:** a clip of a video processing with `#1`, `#2`, `#3` IDs visible on
the boxes, then the summary table and download buttons at the end.

---

Detection counts boxes. Tracking counts things. That difference took me a while to properly understand.

Run object detection on a 60-second video and you get an answer like "793 frames contained a car." Useless. Was that one car for 793 frames, or 793 cars?

Object tracking answers it. Every object gets a persistent ID that survives across frames — so the same car stays #4 whether it's moving, partly hidden, or briefly out of shot.

On the clip in this video: 1440 frames, processed at 52.8 FPS on CPU, and the real answer was 8 unique vehicles.

That unlocks the questions you actually care about:
→ How many people entered, not how many person-shaped boxes appeared
→ How long each one stayed (their frame count ÷ FPS)
→ What route they took through the scene

One honest detail: the nano model occasionally labelled a car as "train" or "airplane" on dramatic close-up angles. Not a bug — a small model doing its best with an unusual viewpoint. A larger model and a higher confidence threshold fix it. I'm leaving that in the demo because pretending models are perfect helps nobody.

Every detection is logged to CSV with timestamps, track IDs and coordinates, so you can analyse footage after the fact in pandas.

🔗 Try it: [YOUR APP LINK]

#ComputerVision #ObjectTracking #YOLO #ByteTrack #Python #DataEngineering #MachineLearning #AI #OpenCV #BuildInPublic

---

## Post 3 — CCTV, phone cameras, and a networking lesson

**Pair with:** two clips joined — your phone acting as an IP camera being
detected on the laptop, then opening the hosted app on your phone and using the
browser camera live.

---

Your old Android phone is an IP camera. Mine is now a security camera, and it took two minutes to set up.

EyeSpot doesn't care what the camera is. It needs a video stream, and it accepts four kinds:

📱 A phone over HTTP — install IP Webcam, tap Start server, use the URL
📹 CCTV or IP cameras over RTSP — Hikvision, Dahua, Tapo, Reolink, CP Plus
💻 A laptop or USB webcam
🎬 Video files

Since typing `rtsp://admin:password@192.168.1.10:554/stream2` every time is miserable — and puts your CCTV password in your shell history — cameras get saved under a name. After the first time, it's just `--camera gate`.

But the thing I actually want to share is what I got wrong.

I assumed that once the app was deployed to the cloud, I could point it at my home CCTV from anywhere. I couldn't. Not a bug I could fix — a fact about how the internet works.

Home cameras live on private addresses: 192.168.x.x, 10.x.x.x. Those ranges are reserved for local networks and aren't routed on the public internet. Millions of homes use 192.168.1.10 at the same moment. A server in a datacentre has no path to yours, and never will.

So the hosted demo does the honest thing: it tells you that, instead of failing with a confusing timeout. And it offers what actually does work in a browser — your own device's camera, with permission. Open it on your phone and it analyses what your phone sees, because the browser captures the frames and hands them over. The server never needs to reach the camera at all.

For real CCTV, you run it on a machine at home. That's not a workaround — it's the correct architecture. Your footage and credentials never leave your network.

And whatever you do: don't port-forward a camera to the internet to get around this. Exposed CCTV is scanned constantly and has built botnets.

🔗 Try it: [YOUR APP LINK]

#ComputerVision #IoT #CCTV #Networking #RTSP #Python #YOLO #HomeSecurity #SoftwareEngineering #BuildInPublic

---

## 4. Before you post — fix the links

Both links are private right now. Checked while writing this:

| Link | Status | What a visitor sees |
|---|---|---|
| `eyespot.streamlit.app` | HTTP 303 → login | A Streamlit sign-in page, not your app |
| `github.com/Abubakar2350/EyeSpot` | Private | A 404 |

### Make the app public (30 seconds)

Open the app → **Share** in the top-right → **Make this app public**.
Or Settings → **Sharing** → *Who can view this app* → **"This app is public and searchable"**.

Then replace `[YOUR APP LINK]` in all three posts with:

```
https://eyespot.streamlit.app
```

### The repository is a separate decision

You do not have to make it public. Plenty of good LinkedIn posts link only to a
live demo.

| Choose | Why |
|---|---|
| **Demo only** | Recruiters click demos far more than repos. Your code stays yours. |
| **Demo + repo** | Some engineering roles want to read code. It is the strongest signal you can send — but only if you are happy for it to be read. |

If you do make the repo public, add a second line to each post:

```
💻 Code: https://github.com/Abubakar2350/EyeSpot
```

Nothing in the repository leaks secrets — `cameras.json` has always been
gitignored, and no password was ever committed. That was checked before the
first push.

### If you would rather not make either public

Drop the 🔗 line and end with:

```
Happy to walk through the code or give a live demo — drop me a message.
```

Weaker than a working link, but better than a broken one.

---

## 5. Posting tips

**Timing.** Space them out — one every two or three days. Three posts in one day
competes with yourself.

**The first line is everything.** LinkedIn truncates at roughly 200 characters
behind a "see more". Each post above front-loads a hook for that reason.

**Video beats text.** Native uploads outperform YouTube links. Keep each clip
under 60 seconds and make sure the boxes and labels are readable after
compression — record at 1080p, and prefer a busy scene over a static one.

**Reply to comments.** Early replies drive reach more than any hashtag choice.

**On the honest bits.** Post 2 admits the model misclassifies, post 3 admits I
got the architecture wrong. Keep those. Engineers trust posts that concede
limitations, and they attract better comments than polished claims do.
