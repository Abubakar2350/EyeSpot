# EyeSpot — Next Phases

| | |
|---|---|
| **Date** | 10 September 2026 |
| **Status of Phase 1** | Complete and deployed — see [FINAL_REPORT.md](FINAL_REPORT.md) |
| **Purpose** | What could be built next, sized honestly and put in a sensible order |

---

## 1. Where the project stands

Phase 1 delivered a working detection and tracking system with two interfaces,
four kinds of camera input, and a hosted demo. Everything the PRD asked for is
built, plus five features it deferred.

What Phase 1 deliberately did not do is make EyeSpot *autonomous*. Today it
shows you what it sees **while you watch**. It does not watch on your behalf,
does not tell you when something happens, and does not remember what happened
last night in any form richer than a CSV file.

That gap — between a viewer and a monitor — is the theme of almost everything
worth building next.

The gaps below were verified against the codebase rather than assumed:

| Capability | Present today |
|---|---|
| Alerts / notifications | No code of any kind |
| Zone or line-crossing counting | None |
| Multiple cameras at once | No — single source, no threading |
| Target lock in the web UI | No — CLI only |
| Automated tests in CI | No — the suite is hand-rolled and run manually |
| Installable package | No `pyproject.toml`; the app runs from a checkout |

---

## 2. How items are sized

Effort is relative, not a promise of hours.

| Size | Meaning |
|---|---|
| **S** | A focused session. One file, few moving parts. |
| **M** | A few sessions. Touches several modules, needs its own tests. |
| **L** | A significant piece of work with design decisions to settle first. |
| **XL** | A project in its own right, with running costs and ongoing obligations. |

---

## 3. Phase 2 — Make it watch for you

The theme: turn a viewer into a monitor. Everything here is buildable on the
existing codebase with no new infrastructure.

### 2.1 Alert system · **M** · highest value

The single most valuable thing missing. Detection already works; what is absent
is anyone being told.

- Watch for a class appearing (`person` at night, say)
- **Debounce it** — this is the real work. Without it, one person standing still
  fires an alert on every frame, fifty times a second. Needs a cooldown per
  track ID, and a minimum persistence before firing, so a single bad frame does
  not trigger anything.
- Save a snapshot and a short clip around the event
- Send a notification: desktop, Telegram bot, email, or webhook

The tracking work from Phase 1 makes this far easier than it would otherwise be:
because objects carry persistent IDs, "one person arrived" is distinguishable
from "one person is still here", which is exactly what a debounce needs.

*Depends on: nothing. Build this first.*

### 2.2 Event log and history · **M**

The CSV is a raw record, not a history. This turns it into one.

- SQLite database of events rather than one row per detection per frame
- Each event: what, when, how long, which camera, which snapshot
- A `--report` command producing "last night: 3 people between 01:12 and 04:40,
  2 vehicles" instead of 40,000 CSV rows
- A history tab in the web UI

*Depends on: 2.1 to define what an "event" is.*

### 2.3 Zone and line-crossing counting · **M**

The most-requested feature in any CCTV system, and absent here.

- Draw a polygon (a region) or a line on the frame
- Count track IDs entering, leaving, or crossing
- Directional counting: "in" versus "out"

Persistent track IDs make this tractable — you watch for an ID whose centre
moves from one side of a line to the other. Without tracking it would be
impossible, which is why it belongs after Phase 1, not during it.

*Depends on: tracking (done). Interacts well with 2.1.*

### 2.4 Run as a service · **S**

Unattended operation, today, needs a shell and a terminal left open.

- A `systemd` unit file for Linux and instructions for Windows Task Scheduler
- Restart on failure, start on boot
- Log rotation so `outputs/` does not grow without bound

*Depends on: nothing. Pairs naturally with 2.1.*

### 2.5 Event-only recording · **S**

Currently `--save` writes everything, which fills a disk quickly on a 24/7
camera. Record only a window around each event — a few seconds before and after.

*Depends on: 2.1.*

---

## 4. Phase 3 — Accuracy and reach

### 3.1 Small-object detection · **M**

Distant birds, faraway people and number plates are all the same problem: the
object is a handful of pixels and the model misses it. The standard fix is
slicing — cut the frame into overlapping tiles, run inference on each, and merge
the results (the SAHI approach). Considerably slower, and worth it only for the
cases that need it.

This directly addresses the bird-watching case already documented in the README,
where the advice today is "use a camera with optical zoom".

### 3.2 Custom training · **L**

The only path past the 80 COCO classes. A separate discipline: collecting
images, labelling them, training, evaluating, and iterating. Worth it only when
there is a specific object COCO does not know — a particular tool, a specific
product, a uniform.

Phase 1 already loads any `.pt` file through `--model`, so a custom model drops
in with no code changes. The work is entirely in producing the model.

### 3.3 Privacy mode — face and plate blurring · **M**

Genuinely important if footage is ever stored or shared. Blur faces and number
plates in saved video and snapshots. For a home security system this is
courtesy; if the camera can see a public street or a neighbour's property, it
may be a legal requirement depending on jurisdiction.

Worth noting: this needs a face detector in addition to YOLO's person class,
since "person" is a whole body, not a face.

### 3.4 Model benchmarking command · **S**

A `--benchmark` flag that times `yolov8n`, `s`, and `m` at several image sizes on
the user's own machine and prints a table. Removes the guesswork from a decision
that currently depends on hardware the documentation cannot know about.

---

## 5. Phase 4 — Engineering quality

None of this changes what the product does. All of it makes the project more
credible to another engineer reading it — which matters for a portfolio piece.

### 4.1 pytest and continuous integration · **M**

`test_smoke.py` is a hand-rolled runner: 18 checks, all passing, but run
manually. Converting to pytest and adding a GitHub Actions workflow means every
push is verified automatically, and a green badge on the README is a real
signal.

The main cost is CI runtime — the model download and torch install make for a
slow job, so the workflow needs caching to be practical.

### 4.2 Packaging · **S**

Add `pyproject.toml` so the project installs as a package with a console entry
point — `eyespot --camera gate` instead of `python main.py --camera gate`. Small
change, disproportionate improvement in how finished it feels.

### 4.3 Multi-camera support · **L**

Today one source is processed at a time. Watching a house properly means four
cameras at once. Requires a worker per stream, a shared model instance, and
careful thought about CPU contention — four streams on a four-core machine will
not each run at full speed, so frame-skipping strategy matters.

*Depends on: 2.2, so events from different cameras land in one place.*

### 4.4 Target lock in the web UI · **M**

Target lock is one of the more distinctive features and it exists only in the
CLI, where you click a frame in an OpenCV window. The web UI has no equivalent,
so the hosted demo never shows it off. Implementing it needs click coordinates
from a Streamlit image — an interactive component rather than a static one.

Worth doing largely because it is the feature most worth *showing*, and right
now it is invisible to anyone who only sees the deployed app.

---

## 6. Phase 5 — Multi-user product · **XL**

Covered in detail in [USER_REPORT_HOME_CCTV.md](USER_REPORT_HOME_CCTV.md)
section 7. In brief: Google sign-in is genuinely easy — Streamlit 1.63 has
native OIDC support and it is a few hours of work. Everything else is not.

The blocker is not authentication. It is that a cloud server cannot reach a
camera on someone's home network, so a multi-user product needs a local agent at
each user's home connecting outward, plus a database, encrypted credential
storage, always-on workers, media storage, and multi-tenant isolation. It cannot
run on free hosting, because continuous stream processing needs a machine that
never sleeps.

Two parts of it carry real responsibility: holding other people's CCTV passwords,
and ensuring one user can never see another's camera. A bug in either is a
privacy breach involving video of someone's home, not an inconvenience.

This is a product with running costs and obligations, not a weekend feature.
Recommended only if there is a genuine intention to operate it as a service.

---

## 7. Recommended order

The ordering is driven by value per unit of effort, and by what unblocks what.

| Order | Item | Size | Why here |
|---|---|---|---|
| 1 | **2.1 Alert system** | M | Transforms what the project *is*. Nothing depends on it, so it can start immediately. |
| 2 | **2.4 Run as a service** | S | Small, and alerts are pointless if the app is not running. |
| 3 | **2.2 Event log and history** | M | Makes alerts reviewable rather than transient. |
| 4 | **4.2 Packaging** | S | Cheap, and makes everything after it easier to install and demo. |
| 5 | **2.3 Zone counting** | M | The most recognisable CCTV feature; strong demo material. |
| 6 | **4.1 pytest and CI** | M | Do it once the feature set stops moving, or the tests get rewritten twice. |
| 7 | **4.4 Target lock in the web UI** | M | Makes the best feature visible in the hosted demo. |
| 8 | **2.5 Event-only recording** | S | Matters once the system has been running for weeks. |
| 9 | **3.1 / 3.3 / 3.4** | S–M | Pick by need rather than in sequence. |
| 10 | **4.3 Multi-camera** | L | Wait until single-camera monitoring is genuinely solid. |
| — | **3.2 Custom training** | L | Only when a specific object demands it. |
| — | **Phase 5 multi-user** | XL | Only with intent to operate a service. |

### If only one thing gets built

Build the alert system. Everything else on this list improves EyeSpot; alerts
change what it is for. A system that shows you a person when you happen to be
looking is a demo. A system that tells you a person appeared at 2:14am, with a
photo, is a security camera.

---

## 8. What is deliberately not on this list

Recording decisions taken, so they are not silently revisited later.

| Not planned | Reason |
|---|---|
| Vercel or serverless deployment | Structurally impossible — 250 MB bundle limit against ~930 MB of dependencies, and Streamlit needs a persistent server. See FINAL_REPORT 7.3. |
| Port-forwarding cameras to the internet | Actively unsafe. Exposed CCTV is scanned constantly and has been assembled into botnets. Use a tunnel instead. |
| Face recognition | A different problem from detection, with serious privacy and legal weight. Not to be added casually. |
| Cross-camera re-identification | Track IDs are valid within one stream only. Doing this properly is a research-grade problem. |
| Hugging Face Spaces migration | Ready and working under `deploy/`, but blocked on a PRO subscription. Revisit only if that changes. |
