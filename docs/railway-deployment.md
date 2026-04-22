# Deploying Open Notebook on Railway

This guide walks you through deploying Open Notebook on [Railway](https://railway.app) from scratch. You'll end up with a fully working instance protected by login credentials, with no exposed database or unprotected API.

---

## What you'll create

Two Railway services in one project:

| Service | What it is | Publicly accessible? |
|---|---|---|
| **surrealdb** | The database | No — internal only |
| **open-notebook** | API + worker + frontend (all-in-one) | Yes — port 8502 |

All three internal processes (API on `:5055`, background worker, Next.js frontend on `:8502`) run inside the single `open-notebook` container managed by supervisord. Only the frontend port is exposed to the internet, so your API docs and raw API are never directly reachable.

---

## Prerequisites

- A [Railway account](https://railway.app) (free tier works for testing; paid for persistent volumes)
- Your Anthropic API key (or any other supported AI provider key)
- 10 minutes

---

## Step 1 — Create a new Railway project

1. Go to [railway.app](https://railway.app) → **New Project**
2. Choose **Empty project**
3. Name it something like `open-notebook`

---

## Step 2 — Add the SurrealDB service

1. In your project, click **+ New** → **Docker Image**
2. Image name: `surrealdb/surrealdb:v2`
3. Click the service → **Settings** tab:
   - **Start command:**
     ```
     start --log warn --user $SURREAL_USER --pass $SURREAL_PASS --bind 0.0.0.0:8000 file:/data/db
     ```
   - **Port:** `8000`
4. Go to the **Variables** tab and add:
   ```
   SURREAL_USER=root
   SURREAL_PASS=root
   ```
5. Go to the **Volumes** tab → **+ Add Volume**
   - Mount path: `/data`
   - This is critical — without a volume, your database is wiped on every deploy.
6. **Do NOT** generate a public domain for this service. It should stay private.

> Railway services in the same project can reach each other at
> `<service-name>.railway.internal`. Your SurrealDB will be reachable at
> `surrealdb.railway.internal:8000`.

---

## Step 3 — Add the Open Notebook service

1. Click **+ New** → **Docker Image**
2. Image name: `lfnovo/open_notebook:v1-latest`
3. Click the service → **Settings** tab:
   - **Port:** `8502`
4. Go to **Settings → Networking** → **Generate Domain** to get your public URL.
   Note it down — you'll need it for env vars (e.g. `https://open-notebook-production.up.railway.app`).

---

## Step 4 — Configure environment variables

In the **open-notebook** service → **Variables** tab, add every variable below.

### Secrets (generate these — do not use the defaults)

Generate two random secrets. You can use:
```bash
openssl rand -hex 32
```

| Variable | Value |
|---|---|
| `OPEN_NOTEBOOK_ENCRYPTION_KEY` | A random string, min 16 characters |
| `OPEN_NOTEBOOK_JWT_SECRET` | A different random string, min 32 characters |

### Admin account

This account is created automatically on first startup if no users exist.

| Variable | Value |
|---|---|
| `OPEN_NOTEBOOK_ADMIN_EMAIL` | Your email address |
| `OPEN_NOTEBOOK_ADMIN_PASSWORD` | A strong password (min 12 characters) |

### Database connection

| Variable | Value |
|---|---|
| `SURREAL_URL` | `ws://surrealdb.railway.internal:8000/rpc` |
| `SURREAL_USER` | `root` |
| `SURREAL_PASSWORD` | `root` |
| `SURREAL_NAMESPACE` | `open_notebook` |
| `SURREAL_DATABASE` | `open_notebook` |

### URLs (replace with your actual Railway domain)

| Variable | Value |
|---|---|
| `API_URL` | `https://your-project.up.railway.app` |
| `CORS_ORIGINS` | `https://your-project.up.railway.app` |
| `INTERNAL_API_URL` | `http://localhost:5055` |

### Lock down registration

This prevents random internet users from creating accounts and using your AI credits:

| Variable | Value |
|---|---|
| `OPEN_NOTEBOOK_DISABLE_REGISTRATION` | `true` |

### AI provider key

At least one is required to use AI features. If you provide an Anthropic key, Claude Sonnet is automatically set as the default model on first startup.

| Variable | Value |
|---|---|
| `ANTHROPIC_API_KEY` | `sk-ant-...` |

Other supported keys: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`

---

## Step 5 — Deploy

Railway deploys automatically when you save variables. Watch the **Deploy Logs** for the open-notebook service. A successful startup looks like:

```
✓ Database migrations completed
✓ Created default admin user: you@example.com
✓ Assigned unowned records to admin user
✓ Auto-configured claude-3-5-sonnet-20241022 as default language model
✓ API initialization completed successfully
```

The frontend waits for the API to be healthy before starting (via `wait-for-api.sh`), so allow 60–90 seconds for the first boot.

---

## Step 6 — First login

1. Open your Railway domain in a browser
2. You'll see the login page
3. Sign in with the `OPEN_NOTEBOOK_ADMIN_EMAIL` and `OPEN_NOTEBOOK_ADMIN_PASSWORD` you set
4. If you want to invite other users, go to **Settings → Users → Invite User**

---

## Security summary

| Protection layer | How it works |
|---|---|
| **Login required** | Every page and API endpoint requires a valid JWT token. Unauthenticated requests are redirected to login. |
| **Registration disabled** | `OPEN_NOTEBOOK_DISABLE_REGISTRATION=true` blocks self-signup. Only admins can invite new users. |
| **API never directly exposed** | Port 5055 (raw API) is internal only. The internet only reaches port 8502 (Next.js frontend). Swagger docs are unreachable from outside. |
| **Database internal only** | SurrealDB has no public Railway domain. Only the open-notebook container can reach it. |
| **CORS locked down** | `CORS_ORIGINS` set to your domain blocks cross-origin API abuse from other websites. |
| **Encrypted credentials** | AI provider API keys stored in the database are encrypted with `OPEN_NOTEBOOK_ENCRYPTION_KEY`. |

---

## Adding your own volumes (optional but recommended)

The open-notebook container stores LangGraph checkpoints and cached data in `/app/data`. To preserve these across deploys:

1. open-notebook service → **Volumes** → **+ Add Volume**
2. Mount path: `/app/data`

---

## Updating to a new version

1. open-notebook service → **Settings** → **Deploy** → Change image tag (e.g. `v1.2-latest`)
2. Or use `v1-latest` to always pull the latest v1.x image on redeploy

Migrations run automatically on startup — no manual steps needed.

---

## Troubleshooting

**Frontend shows "Cannot connect to API"**
- Check `INTERNAL_API_URL` is `http://localhost:5055` (not a Railway domain)
- Check deploy logs for API startup errors

**Login fails immediately**
- Verify `OPEN_NOTEBOOK_ADMIN_EMAIL` and `OPEN_NOTEBOOK_ADMIN_PASSWORD` match what you set
- Check logs for "Created default admin user" — if it's not there, the user seed may have failed

**Database connection errors**
- Confirm the SurrealDB service is running (no public domain needed, just healthy)
- Verify `SURREAL_URL` uses the `.railway.internal` hostname exactly

**AI features not working**
- Confirm at least one AI provider key is set
- Go to **Settings → API Keys** in the app to verify the key is recognized
- If using Anthropic, check logs for "Auto-configured claude-3-5-sonnet-20241022"

**"Open registration is disabled" error**
- Expected if `OPEN_NOTEBOOK_DISABLE_REGISTRATION=true` — use Settings → Users to invite users as admin

---

## Full environment variable reference

```env
# === REQUIRED ===
OPEN_NOTEBOOK_ENCRYPTION_KEY=<random-32-char-string>
OPEN_NOTEBOOK_JWT_SECRET=<different-random-32-char-string>

# === ADMIN ACCOUNT ===
OPEN_NOTEBOOK_ADMIN_EMAIL=you@example.com
OPEN_NOTEBOOK_ADMIN_PASSWORD=strong-password-here

# === DATABASE ===
SURREAL_URL=ws://surrealdb.railway.internal:8000/rpc
SURREAL_USER=root
SURREAL_PASSWORD=root
SURREAL_NAMESPACE=open_notebook
SURREAL_DATABASE=open_notebook

# === URLS (update with your Railway domain) ===
API_URL=https://your-project.up.railway.app
CORS_ORIGINS=https://your-project.up.railway.app
INTERNAL_API_URL=http://localhost:5055

# === SECURITY ===
OPEN_NOTEBOOK_DISABLE_REGISTRATION=true

# === AI PROVIDER (at least one required) ===
ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
# GOOGLE_API_KEY=...
# GROQ_API_KEY=gsk_...

# === OPTIONAL ===
# OPEN_NOTEBOOK_DEFAULT_LANGUAGE_MODEL=claude-3-5-sonnet-20241022
```
