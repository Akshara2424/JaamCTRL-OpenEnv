# Pre-Submission Checklist — JaamCTRL
# Complete every item before pasting your HF Space URL into the submission form.

---

## PHASE 1 — Local Verification (do first, on your machine)

### Code correctness

- [ ] `python -c "import sys; sys.path.insert(0,'.'); from env import JaamCTRLTrafficEnv; env = JaamCTRLTrafficEnv(task_id=3, mock_sumo=True); obs,_ = env.reset(); print('OK', obs['flat'].shape)"`
      Expected output: `OK (46,)`

- [ ] `python inference.py --mock --task 1 2>&1 | head -5`
      First line must start with `[START] {"event": "START", ...}`

- [ ] `python inference.py --mock 2>/dev/null | grep -E "^\[(START|STEP|END)\]" | wc -l`
      Should be > 0 (at least START + some STEPs + END per task)

- [ ] `python -c "import yaml; spec=yaml.safe_load(open('openenv.yaml')); assert len(spec['tasks'])==3; print('YAML OK')"`

- [ ] `python inference.py --mock --no-baseline 2>/dev/null | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    for tag in ('[START]','[STEP]','[END]'):
        if line.startswith(tag):
            json.loads(line.replace(tag+' ',''))  # will throw if malformed
print('JSON parse OK for all log lines')
"`

### Models

- [ ] At least one `.zip` model file exists in `agents/models/`
      `ls -lh agents/models/*.zip`

- [ ] Model loads without error:
      `python -c "from stable_baselines3 import PPO; m=PPO.load('agents/models/ppo_jaamctrl.zip'); print('model OK', m.policy)"`

- [ ] Run inference with real PPO model (not mock):
      `python inference.py --task 1 --mock` (use `--mock` if SUMO not installed locally)

### Dashboard

- [ ] `streamlit run app.py` opens without errors on `localhost:8501`
- [ ] All 3 tabs load (Task 1 / 2 / 3 or equivalent)
- [ ] Baseline comparison table renders
- [ ] Heatmap tab renders (even with synthetic data)
- [ ] No Python tracebacks visible in the Streamlit UI

---

## PHASE 2 — Docker Build (do second)

- [ ] `docker build -t jaamctrl .`
      Build must complete with exit code 0. Watch for any pip install failures.

- [ ] `docker run -p 7860:7860 -e MOCK_SUMO=1 jaamctrl`
      Open `http://localhost:7860` — Streamlit dashboard must load.

- [ ] `docker run -e MOCK_SUMO=1 -e INFERENCE_MODE=1 jaamctrl 2>/dev/null | head -20`
      Must show `[START]` on the first line.

- [ ] `docker run -e MOCK_SUMO=1 jaamctrl python inference.py --mock --task 3 2>/dev/null | grep "\[END\]"`
      Must return a `[END]` JSON line.

- [ ] Check image size is reasonable: `docker image ls jaamctrl`
      Anything under 4 GB is fine for HF Spaces.

---

## PHASE 3 — HuggingFace Space setup

### Create the Space

- [ ] Go to https://huggingface.co/new-space
- [ ] Space name: `jaamctrl` (or `jaamctrl-traffic`)
- [ ] SDK: **Docker** (not Gradio, not static)
- [ ] Hardware: **CPU basic** (free tier is fine; SUMO runs on CPU)
- [ ] Visibility: **Public** (required for submission)

### Push the repo

```bash
# If using Git directly
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/jaamctrl
git push hf main

# Or with HF CLI
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create jaamctrl --type space --space-sdk docker
git push hf main
```

- [ ] All files pushed (check Files tab on HF Spaces)
- [ ] Space starts building (yellow dot = building, green = live)
- [ ] Build log shows no errors: click "Build logs" in the Space settings

### README.md frontmatter

Your README must have this at the very top (HF parses it):

```yaml
---
title: JaamCTRL — AI Adaptive Traffic Signal Control
emoji: traffic-light
colorFrom: indigo
colorTo: cyan
sdk: docker
pinned: true
license: mit
---
```

- [ ] Verify frontmatter is the first thing in `README.md`
- [ ] `title:` field matches what you want shown on the Space card
- [ ] `sdk: docker` is set (not `gradio`)

### Space secrets (if needed)

- [ ] Go to Space Settings → Secrets
- [ ] Add `MOCK_SUMO = 0` if SUMO was successfully installed in Docker build
- [ ] Add `MOCK_SUMO = 1` if you want to force mock mode (safe fallback)
- [ ] Add `TEAM_NAME = YourTeamName` (shows in [END] JSON logs)

---

## PHASE 4 — Live Space verification (critical)

Wait for the green dot, then:

- [ ] Open your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/jaamctrl`
      Streamlit dashboard must load within 60 seconds.

- [ ] Click through all tabs — no crash, no blank screens

- [ ] Check the Space logs (Logs tab in HF) — no `ModuleNotFoundError` or `SUMO_HOME not found`

- [ ] Run inference remotely (optional but impressive for judges):
      ```bash
      docker run --rm \
        -e MOCK_SUMO=1 \
        -e INFERENCE_MODE=1 \
        YOUR_USERNAME/jaamctrl:latest \
        python inference.py --mock 2>/dev/null | grep "\[END\]"
      ```

- [ ] Copy the live Space URL and confirm it loads on a different device / incognito window
      Format: `https://huggingface.co/spaces/YOUR_USERNAME/jaamctrl`

---

## PHASE 5 — Submission form

- [ ] GitHub repo is **public**
- [ ] HF Space URL is **public** and loading
- [ ] `inference.py` is in the repo **root** (not inside `env/`)
- [ ] `openenv.yaml` is in the repo **root**
- [ ] `README.md` is in the repo **root** with correct frontmatter
- [ ] `requirements.txt` is in the repo **root**
- [ ] `Dockerfile` is in the repo **root**
- [ ] At least one model `.zip` is committed to `agents/models/`
      (or documented in README how to generate it)
- [ ] Git log shows meaningful commit history (not one giant commit)
      Good commit messages: `feat: add 3-intersection SUMO corridor network`
      Bad: `update files`

---

## PHASE 6 — Final checks (15 min before deadline)

- [ ] Read your README one more time — no broken links, no placeholder text like `YOUR_USERNAME`
- [ ] The real-world impact numbers are cited or clearly marked as simulation estimates
- [ ] The three task thresholds in `openenv.yaml` match what the agents actually achieve
      (check `logs/results.json` after running `python inference.py --mock`)
- [ ] Space title on HF matches what you submitted
- [ ] Your team name is correct everywhere
- [ ] Submission form URL is saved / screenshot taken

---

## Quick-fix commands for common last-minute issues

```bash
# SUMO not found in Docker
# → Add to Dockerfile: ENV SUMO_HOME=/usr/share/sumo
# → Rebuild and push

# Port mismatch (app not showing on HF Spaces)
# → Ensure CMD uses --server.port 7860, not 8501
# → HF Spaces Docker MUST use port 7860

# ModuleNotFoundError: env
# → Add to inference.py: sys.path.insert(0, str(Path(__file__).parent))
# → Already done in our inference.py — verify it's there

# Model zip not found
# → Set MOCK_SUMO=1 in HF Secrets so inference.py falls back to rule_based
# → The [END] line will still emit; grader will score on heuristic performance

# Streamlit crashes on HF Spaces
# → Add to Dockerfile CMD: --server.enableCORS false --server.enableXsrfProtection false
# → Already in our Dockerfile

# [START]/[STEP]/[END] format wrong
# → python inference.py --mock 2>/dev/null | python3 -c "
#    import sys,json
#    for l in sys.stdin:
#        l=l.strip()
#        for t in ('[START]','[STEP]','[END]'):
#            if l.startswith(t): json.loads(l[len(t)+1:])
#    print('FORMAT OK')"
```
