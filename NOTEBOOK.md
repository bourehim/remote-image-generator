# Kaggle notebook: exact cells

The notebook is only a launcher. There are no widgets, no output rendering, no
model code in cells. Two cells run the whole system.

## 0. Dataset layout (read this first)

Two layouts work; the scripts detect which one they are in.

**A. Launcher-only dataset** — the current `janemerser/remote-image-generator`
dataset. Its root is exactly:

```
/kaggle/input/datasets/janemerser/remote-image-generator/
├── NOTEBOOK.md
├── _common.py
├── bootstrap.py
├── install_cloudflared.sh
└── launch.py
```

There is **no `server/` and no `requirements/`** in this layout, so bootstrap
must be given the project source once (`--source`, or `RIMAGE_SOURCE`, or a
second full-repo dataset via `--project-dir`). After a successful bootstrap the
runnable copy is `/kaggle/working/remote-image`.

**B. Full-repository dataset** — upload the whole repo (`server/`,
`requirements/`, `kaggle/`, …). Then the plain bootstrap command works with no
extra flags.

## 1. Notebook settings

* **Add Input → Your Datasets →** the dataset above
* **Settings → Accelerator → GPU T4 x2** (or P100)
* **Settings → Internet → On** (cloudflared, pip and Hugging Face need it)
* **Settings → Persistence → Files only** (optional)
* **Add-ons → Secrets** (recommended):
  * `RIMAGE_API_TOKEN` — a random token, e.g. `openssl rand -base64 36`
  * `HF_TOKEN` — only if you plan to use FLUX.1-schnell
  * `CLOUDFLARE_TUNNEL_TOKEN` + `RIMAGE_PUBLIC_URL` — only for a named tunnel

  Without `RIMAGE_API_TOKEN`, `launch.py` generates one and writes it to
  `/kaggle/working/.rimage/api_token` (mode 0600). It is never printed.

## Cell 1 — bootstrap (run once per session)

**Layout A (launcher-only, the current dataset):**

```python
# Option 1: source from a git repository (public, or a token-bearing URL)
!python /kaggle/input/datasets/janemerser/remote-image-generator/bootstrap.py \
    --source https://github.com/<you>/remote-image-generator.git

# Option 2: a second dataset that contains the full repository
!python /kaggle/input/datasets/janemerser/remote-image-generator/bootstrap.py \
    --project-dir /kaggle/input/remote-image-generator-full

# Option 3: an archive inside the dataset (or any URL)
!python /kaggle/input/datasets/janemerser/remote-image-generator/bootstrap.py \
    --source /kaggle/input/datasets/janemerser/remote-image-generator/src.zip
```

**Layout B (full repository dataset):**

```python
!python /kaggle/input/remote-image-generator/kaggle/bootstrap.py
```

Expected output: detected launcher dir + project dir, pip install of the pinned
wheel set (torch untouched), torch / diffusers / transformers versions, GPU name
+ VRAM, and a per-model compatibility table. Exit code 3 with a hint means the
dataset has no server code — use `--source` as above.

## Cell 2 — start server + tunnel (keep running)

```python
!python /kaggle/working/remote-image/kaggle/launch.py
```

That path exists after bootstrap. You can also start it straight from the
launcher-only dataset — it finds `/kaggle/working/remote-image` automatically:

```python
!python /kaggle/input/datasets/janemerser/remote-image-generator/launch.py
```

This cell prints the public HTTPS URL, verifies it from the public internet,
checks that auth is enforced, and then keeps running. **Leave it running** —
interrupting it stops the server and the tunnel.

Copy the URL from the output, then on your Linux machine:

```bash
remote-image configure --url https://<random-words>.trycloudflare.com
```

The token prompt expects the value stored in `/kaggle/working/.rimage/api_token`
(open it from the notebook file browser, or from a cell with
`!cat /kaggle/working/.rimage/api_token` — that prints it, so do it only when
nobody is looking at your screen).

## Cell 3 (optional) — inspect logs

```python
!tail -n 60 /kaggle/working/rimage-logs/server.log
!tail -n 30 /kaggle/working/rimage-logs/cloudflared.log
```

## Cell 4 (optional) — health check from inside Kaggle

```python
!curl -s http://127.0.0.1:8000/health | python -m json.tool
```

## Shutdown / restart

* **Stop:** interrupt cell 2 (■ button). `launch.py` terminates cloudflared and
  uvicorn in that order. Closing the notebook tab alone is not enough; use
  "Stop Session" to be sure.
* **Restart:** re-run cell 2. A **new Quick Tunnel URL** appears, so run
  `remote-image configure --url <new-url>` again. The token stays the same if it
  came from a Kaggle Secret or the token file in the same session; a fresh
  session with no secret generates a new one.
* **New session:** installed packages and `/kaggle/working/remote-image` are
  gone, so run cell 1 again (with `--source` for layout A), then cell 2.
* **After 12 hours / quota exhaustion:** the session ends, the tunnel dies and
  `/kaggle/working` is preserved only as notebook output. Nothing is automatic.

## Things that do not persist (by design)

| Item | Persists? |
| --- | --- |
| Installed pip packages | No (re-run bootstrap) |
| `/kaggle/working/remote-image` staged copy | Only as notebook output |
| Running processes / tunnel URL | No |
| Model cache in `~/.cache/huggingface` | No (re-downloads on first use) |
| Token file | Only if the session itself is resumed |

Do not try to keep a session alive past Kaggle's limits; the server is designed
to be restarted cheaply.
