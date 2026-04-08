# Deploying JaamCTRL to Hugging Face Spaces

This guide walks you through deploying JaamCTRL as a Streamlit app on Hugging Face Spaces.

---

## Prerequisites

1. **Hugging Face Account**: [huggingface.co](https://huggingface.co) (free)
2. **Git**: For version control (optional, can create Space directly via web UI)
3. **GitHub Repository**: (Optional) If deploying from GitHub

---

## Deployment Option 1: Via Web UI (Easiest)

### Step 1: Create a New Space on Hugging Face

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in the details:
   - **Space name**: `jaamctrl`
   - **Owner**: Your username (e.g., `WitchedWick`)
   - **License**: MIT
   - **Space SDK**: `Streamlit`
   - **Visibility**: Public

4. Click **"Create Space"**

### Step 2: Upload Repository Files

Once the Space is created, you have two options:

#### Option A: Upload via Web Interface
1. Click **"Files"** → **"Upload files"**
2. Drag and drop (or select) these files from your local `JaamCTRL` folder:
   - `app.py`
   - `requirements.txt`
   - `HF_SPACES_README.md` → rename to `README.md`
   - `inference.py`
   - `openenv.yaml`
   - `env/` (entire folder)
   - `src/` (entire folder)
   - `sumo/` (folder with `.sumocfg`, `.xml` files)
   - `models/ppo_jaam_ctrl.zip`
   - `assets/bg.jpeg` (optional, for styling)
   - `.streamlit/config.toml`

3. Click **"Commit to main"**

#### Option B: Clone & Push via Git (Advanced)
```bash
# Clone the Space repo
git clone https://huggingface.co/spaces/WitchedWick/jaamctrl
cd jaamctrl

# Copy your JaamCTRL files into this directory
cp -r ../JaamCTRL/* .

# Commit and push
git add .
git commit -m "Initial JaamCTRL deployment"
git push
```

### Step 3: Configure README

The Space uses the **README.md** as its landing page description.

1. In the Space repository, edit or create `README.md`
2. Copy the contents from [HF_SPACES_README.md](HF_SPACES_README.md)
3. Commit the changes

---

## Deployment Option 2: From GitHub (Recommended)

If you have JaamCTRL on GitHub and want to sync:

### Step 1: Create New from GitHub

1. Go to [huggingface.co/spaces/create](https://huggingface.co/spaces/create)
2. Check **"Clone from repository URL"**
3. Enter: `https://github.com/your-org/jaamctrl`
4. Set **SDK** to `Streamlit`
5. Click **"Create Space"**

The Space will automatically sync with your GitHub repository.

### Step 2: Keep Updated

To update the Space after pushing changes to GitHub:
- **Automatic**: The Space will re-deploy on every commit to the main branch
- **Manual**: Click **"Restart"** button in the Space settings

---

## Verification

Once deployed, you should see:

1. **Space URL**: `https://huggingface.co/spaces/WitchedWick/jaamctrl`
2. **Streamlit App**: Loads the `app.py` dashboard
3. **Status**: Green "Running" badge
4. **Logs**: Click **"Logs"** to see any errors

---

## Environment Variables

If you want to configure the Space, add environment variables in **Space Settings** → **Variables**:

| Variable | Value | Purpose |
|---|---|---|
| `MOCK_SUMO` | `1` | Force mock mode (no SUMO needed) |
| `TEAM_NAME` | `JaamCTRL` | For inference logging |
| `STREAMLIT_SERVER_HEADLESS` | `true` | Required for Streamlit on Spaces |

*(These are auto-configured, but you can override.)*

---

## What Runs in the Space

### Included
- Streamlit dashboard (`app.py`)
- Interactive metrics & visualizations
- Rule-based & PPO control strategies
- Task difficulty selection (Easy/Medium/Hard)
- Mock SUMO (synthetic traffic simulation)
- Trained PPO model inference

### Not Available
- Real SUMO simulator (requires system `apt install sumo`)
- Live TraCI connection to running SUMO
- GPU acceleration (not needed for inference)

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'traci'"
**Cause**: SUMO is not installed  
**Fix**: The code auto-detects HF Spaces and uses `MOCK_SUMO=1`

### Error: "Cannot import streamlit"
**Cause**: `requirements.txt` is missing or malformed  
**Fix**: Ensure `requirements.txt` is in the Space root directory

### App is slow to start
**Cause**: First load compiles Streamlit & installs dependencies  
**Fix**: Wait 30–60 seconds for Space to warm up

### Space restarts constantly
**Cause**: Memory limit exceeded or infinite loop  
**Fix**: Check logs, comment out heavy computations, contact HF support

---

## Customization

### Change the README
Edit `README.md` in the Space to customize the landing page.

### Update the Model
If you train a new PPO model:
```bash
# Locally
python -m src.rl_agent  # trains and saves to models/ppo_jaam_ctrl.zip

# Push to Space
git commit -am "Update trained PPO model"
git push
```

### Modify Colors
Edit the `COLOUR PALETTE` section in `app.py` (lines with `YELLOW`, `PINK`, `MINT`).

---

## Monitoring

### View Space Metrics
- **Visitors**: [Space page] → **"Insights"**
- **Logs**: **"Logs"** tab
- **Uptime**: Tracked automatically

### Share Your Space
Once live, share the link:
```
Check out JaamCTRL — AI Traffic Signal Control:
https://huggingface.co/spaces/WitchedWick/jaamctrl
```

---

## Security & Privacy

- Space is **public** (anyone can view & use)
- No data is stored (stateless Streamlit app)
- Model runs locally in the user's browser context
- No external API calls for inference

---

## Pro Tips

1. **Add a license file**: Create `LICENSE` in Space root (MIT recommended)
2. **Add a space thumbnail**: Upload 400×400px PNG via Space settings
3. **Link to GitHub**: Add GitHub badge in README
4. **Use persistent storage** (optional): For training logs, use `/tmp/` (non-persistent) or add a persistent volume

---

## Support

- **Hugging Face Docs**: [hf.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- **GitHub Issues**: Post questions to your repo
- **Community**: Ask on [Hugging Face Forums](https://discuss.huggingface.co/)

---

**Your Space is now live!** Share it with the world.
