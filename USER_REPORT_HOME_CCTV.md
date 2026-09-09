# User Report — Home CCTV, Continuous Monitoring, and Multi-User Access

| | |
|---|---|
| **Date** | 9 September 2026 |
| **Project** | EyeSpot — Real-Time Object Detection & Tracking |
| **Scope** | Connecting a home CCTV camera, whether that setup persists, and what a Google-sign-in multi-user version would require |
| **Related** | [FINAL_REPORT.md](FINAL_REPORT.md) · [README.md](README.md) |

---

## 1. Summary

Three questions were asked. Short answers first, detail below.

| Question | Answer |
|---|---|
| How do I connect my home CCTV? | Four steps, about ten minutes — section 3. It works **on your own machine**. |
| Once set up, does it stay saved? | **Locally yes**, permanently, in `cameras.json`. **On the hosted app, no** — and it cannot be fixed by configuration; section 4 explains why. |
| Can it watch all the time? | Not today. The app watches while you have it open. Always-on monitoring is a genuinely different system — section 5. |
| Can we add Google sign-in so every user has their own cameras? | Technically yes, and Streamlit supports it natively. But sign-in is the *easy* part; the hard part is that a cloud server cannot reach anyone's home camera. Section 7 sets out what a real version needs. |

**The single most important finding in this report:** the hosted app at
`eyespot.streamlit.app` can never talk to your home CCTV, no matter what code is
written. This is a property of how the internet addresses home networks, not a
limitation of EyeSpot. Section 6 explains it, and gives four ways around it with
honest trade-offs.

---

## 2. Your network, as measured

Taken from this machine, so the examples below use real values rather than
placeholders:

| Item | Value |
|---|---|
| This PC | `192.168.100.175` |
| Router / gateway | `192.168.100.1` |
| Home subnet | `192.168.100.0/24` — addresses `192.168.100.1` to `192.168.100.254` |
| Tools available | `vlc`, `ffprobe`, `nc` (no `nmap`) |

Your CCTV camera, once connected to the same Wi-Fi or LAN, will receive an
address inside `192.168.100.x`.

### Scan result — no camera is reachable yet

The whole subnet was scanned while writing this report:

| Check | Result |
|---|---|
| Devices seen on the network | 10 |
| Answering on RTSP port 554 | **none** |
| Answering on web ports 80 / 8080 / 8000 | only the router (`.1`) and this PC (`.175`) |

So the camera is not currently reachable from this machine. Three likely
reasons, in order of how often they turn out to be the cause:

1. **RTSP is switched off in the camera.** Most consumer cameras ship with it
   disabled, because their own phone app uses a cloud connection instead. Look
   for "RTSP", "ONVIF", or "Local access" in the camera app's advanced settings
   and enable it. TP-Link Tapo, for instance, also requires you to create a
   separate "camera account" before RTSP will accept a login.
2. **The camera is on a different network.** Many routers keep 2.4 GHz and 5 GHz
   on separate subnets, and guest networks are isolated by design. The camera
   and the PC must be on the same one.
3. **The camera is cloud-only.** A minority of inexpensive models expose no local
   stream at all and work solely through the vendor's app. These cannot be used
   with EyeSpot — or with any other local software — and no configuration will
   change that. RTSP or ONVIF support is the thing to check before buying.

Re-run the scan in step 1 below after enabling RTSP; when the camera appears,
the rest of section 3 applies unchanged.

---

## 3. Connecting your home CCTV

### Step 1 — Find the camera's IP address

Two reliable methods:

**Router admin page (easiest).** Open `http://192.168.100.1` in a browser, log
in, and look for "Connected Devices", "DHCP Clients" or "Device List". The camera
usually appears under its brand name — Hikvision, Dahua, Tapo, CP Plus, Reolink.

**Scan for the RTSP port.** CCTV cameras listen on port 554. This sweeps your
subnet in a few seconds and prints any device answering on that port:

```bash
for i in $(seq 1 254); do
  (nc -z -w1 192.168.100.$i 554 2>/dev/null && echo "RTSP found: 192.168.100.$i") &
done; wait
```

### Step 2 — Build the RTSP URL

Every brand has its own path. Use the **substream** — it is a lower resolution,
which is faster for detection and entirely sufficient:

| Brand | Substream URL |
|---|---|
| Hikvision | `rtsp://user:pass@IP:554/Streaming/Channels/102` |
| Dahua / CP Plus | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=1` |
| TP-Link Tapo | `rtsp://user:pass@IP:554/stream2` |
| Reolink | `rtsp://user:pass@IP:554/h264Preview_01_sub` |
| ONVIF (generic) | `rtsp://user:pass@IP:554/onvif2` |

If you do not know the brand's format, the Streamlit UI has a **URL builder**:
choose *CCTV / IP camera*, pick the brand, enter the IP, and it assembles the URL
and URL-encodes the password for you.

The username and password are the ones set in the camera's own app. Many cameras
require you to enable RTSP first — look for "RTSP" or "ONVIF" in the camera app's
advanced settings.

### Step 3 — Test in VLC before touching EyeSpot

This saves a great deal of time. VLC → **Media → Open Network Stream** → paste
the URL. If VLC plays it, EyeSpot will play it. If VLC does not, the problem is
the camera, the credentials, or the network — not this software.

You can also test from the terminal:

```bash
ffprobe -v error -rtsp_transport tcp -i "rtsp://user:pass@192.168.100.50:554/stream2" \
        -show_entries stream=codec_name,width,height -of csv=p=0
```

### Step 4 — Save it once

```bash
python main.py --source "rtsp://admin:yourpass@192.168.100.50:554/stream2" \
               --save-camera home --camera-note "living room"
```

From then on:

```bash
python main.py --camera home                       # watch it
python main.py --camera home --classes person      # only people
python main.py --camera home --track --log         # with IDs, writing a CSV
```

---

## 4. Does the setup persist?

### On your own machine — yes, permanently

`--save-camera` writes to `cameras.json` in the project folder. It survives
reboots, updates and reinstalls of the app. You set the camera up once and never
type the URL again. `python main.py --list-cameras` shows what is saved, with
passwords masked.

That file is deliberately **excluded from git**, because it contains your CCTV
password. Only the template `cameras.example.json` is committed.

### On the hosted app — no, and this cannot be configured away

Two independent reasons:

1. **Streamlit Community Cloud storage is ephemeral.** The server re-clones the
   repository on every restart, and restarts happen routinely — after a code
   push, and after the app sleeps following 12 hours without traffic. Anything
   written at runtime is lost.
2. **`cameras.json` is not in the repository**, precisely because it holds
   passwords. So even a fresh clone starts with no cameras.

| Environment | Cameras persist? | Why |
|---|---|---|
| Your PC | Yes, permanently | `cameras.json` on real disk |
| `eyespot.streamlit.app` | No | Ephemeral storage, and the file is gitignored |

Making this persist on a hosted app is not a matter of removing the gitignore
rule — committing CCTV passwords to a repository would be a serious mistake.
Persistence for a hosted, multi-user version requires a database with encrypted
credential storage, which is section 7.

---

## 5. "Show it to me all the time" — what that actually requires

### What the app does today

EyeSpot is **session-based**. It shows live detection while you have it open, and
stops when you close it. That is the right design for inspection and testing, and
it is what the PRD asked for.

Additionally, on Streamlit Community Cloud the app **sleeps after 12 hours with
no traffic**. It is not deleted — anyone opening the link wakes it in a few
seconds — but it is not running during that time.

### What continuous monitoring means

"Watch my camera all the time and tell me what happened" is a different product
with different parts:

| Component | Status today |
|---|---|
| Read the camera stream | Built |
| Detect objects and track them | Built |
| Run unattended, 24/7, restarting on failure | **Missing** — needs a service/daemon |
| Record events to durable storage | Partly — CSV exists, but no database |
| Notify you when something happens | **Missing** |
| Browse history: "what happened last night?" | **Missing** |
| Store clips or snapshots of events | Partly — screenshots exist, not event-linked |

### The nearest achievable version, today

A genuinely useful 24/7 setup on your own machine, with no new code:

```bash
python main.py --camera home --classes person --track --log --save --no-display
```

Run it under a service manager so it restarts automatically. On Linux, a
`systemd` unit; on Windows, Task Scheduler. It will run headless, record every
detection to CSV with timestamps and track IDs, and save annotated video.

You could then review a night's activity with a few lines of pandas:

```python
import pandas as pd
df = pd.read_csv("outputs/detections_home.csv")
print(df.groupby("label")["track_id"].nunique())          # unique people, vehicles…
print(df[df.label == "person"].timestamp.min())           # first person of the night
```

What this still does not do is **alert** you. Adding that is a modest,
well-defined piece of work: watch for a class appearing, debounce it so a person
standing still does not fire fifty times, save a snapshot, and send a
notification. That is the single highest-value next feature, and it is
independent of everything else in this report.

---

## 6. The core obstacle: a cloud server cannot reach your home camera

This is the finding that shapes every option below, so it is worth stating
precisely.

Your camera's address, `192.168.100.x`, is a **private address**. Private ranges
(`192.168.x.x`, `10.x.x.x`, `172.16–31.x.x`) are reserved by internet standard
for use inside local networks. They are not routed on the public internet.
Millions of homes use `192.168.100.50` simultaneously; the address is meaningless
outside your own house.

The hosted app runs in a datacentre in the United States. When it tries to open
`rtsp://192.168.100.50:554/...`, it looks for that address on *its own* network,
finds nothing, and times out. No amount of code changes this. The app now detects
that it is hosted and says so plainly, rather than failing with a confusing
connection error.

### Four ways around it

| # | Approach | Effort | Security | Verdict |
|---|---|---|---|---|
| A | Run EyeSpot at home | Low | Excellent — nothing leaves your network | **Best for a single user** |
| B | Encrypted tunnel (Tailscale / Cloudflare Tunnel) | Medium | Good — no open ports, encrypted | Good for remote access to your own cameras |
| C | Port forwarding on the router | Low | **Poor — do not do this** | See warning below |
| D | Local agent that connects outward to a cloud service | High | Good, done properly | The only design that supports many users |

**Option A — run it at home.** A spare PC, or a Raspberry Pi 5, sitting on the
same network as the camera. Direct LAN access, no internet round-trip, full
frame rate, and your camera credentials never leave the house. For one household
this is simply the correct answer.

**Option B — an encrypted tunnel.** Tailscale creates a private network between
your devices; Cloudflare Tunnel exposes a specific service through an outbound
connection. Both avoid opening any inbound port on your router. This lets you
reach your own cameras from anywhere, and is the natural extension of Option A
when you want remote access.

> ### Option C — port forwarding: a warning
>
> Forwarding port 554 on your router to the camera makes it reachable from the
> entire internet. Internet-exposed CCTV cameras are scanned continuously and
> automatically. Many ship with default or weak credentials and firmware that is
> rarely updated; large botnets have historically been assembled from exactly
> these devices. Sites exist whose only purpose is to index exposed camera
> streams.
>
> Please do not expose a home camera this way. Option B achieves the same
> outcome without an open port.

**Option D — a local agent.** A small program runs at home, connects *outward*
to a cloud service, and pushes frames or events up. Because the connection is
outbound, no router configuration and no open ports are needed. This is how
commercial systems — Ring, Nest, Eufy — work, and it is the only design that
scales to many users, each with cameras on their own private network.

---

## 7. Multi-user with Google sign-in

### The good news: authentication is genuinely easy

Streamlit 1.63 — the version this project already uses — ships native OpenID
Connect support. Verified on this machine:

```
streamlit : 1.63.0
st.login  : True
st.logout : True
st.user   : True
```

Adding Google sign-in requires no third-party library. The work is:

1. Create an OAuth client in Google Cloud Console
2. Register the redirect URI: `https://eyespot.streamlit.app/oauth2callback`
3. Add five keys to Streamlit secrets under `[auth]`:

```toml
[auth]
redirect_uri = "https://eyespot.streamlit.app/oauth2callback"
cookie_secret = "a long random string"
client_id = "…from Google Cloud Console"
client_secret = "…from Google Cloud Console"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

4. Gate the app on `st.user.is_logged_in`, and use `st.user.email` as the
   identity key

Realistically a few hours of work, most of it in the Google Cloud Console.

One thing to check before relying on it: Community Cloud's free tier allows only
**one private app at a time**, and there are reports that an
authentication-configured app counts against that allowance. Worth confirming
against your own account before building on it.

### The bad news: sign-in solves the smallest part of the problem

Logging users in tells you *who* they are. It does nothing about the fact that
their cameras sit on private networks a cloud server cannot reach. A
signed-in user would face exactly the same wall described in section 6.

A working multi-user product needs all of this:

| Component | Why it is needed | Rough effort |
|---|---|---|
| Google sign-in | Identify the user | Small |
| Database | Store each user's cameras, events, settings | Medium |
| **Encrypted credential storage** | You would be holding other people's CCTV passwords | Medium, and unforgiving |
| **Local agent** | The only way to reach a camera on a private network | **Large** |
| Agent ↔ cloud protocol | Pairing, authentication, reconnection, versioning | Large |
| Always-on workers | Someone must process streams while nobody is watching | Medium |
| Event store + notifications | The actual product value: "movement at 2am, here is the clip" | Medium |
| Media storage | Snapshots and clips, with a retention policy | Medium |
| Multi-tenant isolation | User A must never see user B's camera | Medium, and unforgiving |

The two rows marked *unforgiving* deserve emphasis. Holding another person's home
CCTV credentials, and their footage, is a serious responsibility. A bug that
leaks one user's stream to another is not an inconvenience — it is a privacy
breach involving video of someone's home. If this is ever built for real users,
those two areas need disproportionate care and independent review.

### Cost reality

This cannot run on free hosting. Continuous stream processing means a machine
that is always awake, which is the opposite of what free tiers provide.

| Piece | Realistic minimum |
|---|---|
| Always-on app + worker | A small paid VPS or container host |
| Database | A managed Postgres instance, small tier |
| Media storage | Object storage, priced by volume and retention |
| Per-user compute | Each concurrent stream needs CPU; this is what scales the bill |

The last row is the one that decides the economics. YOLO inference is not free —
this project measured about 18 ms per frame on a 16-core CPU. Ten users watching
ten cameras continuously is a real compute bill, not a rounding error.

### A sensible order of work

| Phase | Deliverable | Depends on |
|---|---|---|
| **2a** | Alerts and event history on a local install: detect a class, debounce, save a snapshot, notify | Nothing — buildable now |
| **2b** | Remote access to your own cameras via Tailscale or Cloudflare Tunnel | 2a |
| **2c** | Google sign-in on the hosted app, with per-user saved settings in a database | A database |
| **2d** | Local agent that pairs to an account and streams events outward | 2c |
| **2e** | Multi-tenant hosting, media storage, retention, billing | Everything above |

Phase 2a is worth building regardless of whether the rest ever happens: it is
what turns EyeSpot from something you watch into something that watches for you.
Phases 2d and 2e are where this stops being a portfolio project and becomes a
product with running costs and privacy obligations.

---

## 8. Recommendation

**For your own home, now.** Run EyeSpot on a machine at home — Option A. Save the
camera once with `--save-camera`; it persists permanently. Run it headless with
`--log --save` under a service manager for unattended recording. Add Tailscale
(Option B) when you want to check it from outside the house. This is free,
private, and works today.

**For the hosted demo.** Leave it as the showcase it is: image and video upload,
plus the viewer's own phone camera through the browser. It demonstrates the
detection and tracking work convincingly without pretending to reach anyone's
home network.

**Before building the multi-user version.** Build phase 2a first. Alerts and
event history are what make continuous monitoring useful at all, they are
achievable in a single focused effort, and if the multi-user product is ever
built, that work is needed anyway.

**And a caution worth repeating.** Do not port-forward the camera to get around
the networking problem. It is the one shortcut here with a genuinely bad outcome.

---

## Sources

- [Streamlit — User authentication and information](https://docs.streamlit.io/develop/concepts/connections/authentication)
- [Streamlit — Use the Google Auth Platform to authenticate users](https://docs.streamlit.io/develop/tutorials/authentication/google)
- [Streamlit — st.login](https://docs.streamlit.io/develop/api-reference/user/st.login)
- [Streamlit — App resources and limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
- [Streamlit — Status and limitations of Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/status)
