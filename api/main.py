# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# bcrypt 5.x compatibility shim for passlib.                                   #
# passlib.handlers.bcrypt reads bcrypt.__about__.__version__ at import time;  #
# bcrypt 5 removed __about__ and passlib logs "(trapped) error reading bcrypt #
# version" on every process start / login.  Restore the attribute so the      #
# warning never fires.  Harmless no-op on older bcrypt.                       #
# --------------------------------------------------------------------------- #
try:
    import bcrypt  # type: ignore

    if not hasattr(bcrypt, "__about__"):
        _bcrypt_version = getattr(bcrypt, "__version__", "0.0.0")
        bcrypt.__about__ = type(  # type: ignore[attr-defined]
            "_About", (), {"__version__": _bcrypt_version}
        )()
except Exception:
    # bcrypt isn't installed, or a future version exposes __about__ again —
    # either way we don't want this shim to block startup.
    pass

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.auth import JWTAuthMiddleware
from api.routers import (
    auth,
    chat,
    config,
    context,
    credentials,
    embedding,
    embedding_rebuild,
    episode_profiles,
    insights,
    languages,
    models,
    notebooks,
    notes,
    podcasts,
    search,
    settings,
    sharepoint,
    source_chat,
    sources,
    speaker_profiles,
    transformations,
    users,
)
from api.routers import commands as commands_router
from open_notebook.database.async_migrate import AsyncMigrationManager
from open_notebook.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ExternalServiceError,
    InvalidInputError,
    NetworkError,
    NotFoundError,
    OpenNotebookError,
    RateLimitError,
)
from open_notebook.utils.encryption import get_secret_from_env


def _parse_cors_origins(raw: str) -> list[str]:
    """Parse CORS_ORIGINS env value into a list of origins."""
    value = raw.strip()
    if value == "*":
        return ["*"]
    return [origin.strip() for origin in value.split(",") if origin.strip()]


# Parsed once at module load; CORS_ORIGINS changes require a restart.
_cors_origins_raw = os.getenv("CORS_ORIGINS")
CORS_ALLOWED_ORIGINS = _parse_cors_origins(_cors_origins_raw or "*")
CORS_IS_DEFAULT_WILDCARD = _cors_origins_raw is None


def _cors_headers(request: Request) -> dict[str, str]:
    """
    Build CORS headers for error responses.

    Mirrors Starlette CORSMiddleware behavior: reflects the request Origin
    when the origin is allowed (or when wildcard is configured, since
    browsers reject `Access-Control-Allow-Origin: *` combined with
    credentials). Omits `Access-Control-Allow-Origin` for disallowed
    origins so the browser blocks the error body from leaking cross-origin.
    """
    origin = request.headers.get("origin")
    headers: dict[str, str] = {
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "*",
    }

    if origin and ("*" in CORS_ALLOWED_ORIGINS or origin in CORS_ALLOWED_ORIGINS):
        headers["Access-Control-Allow-Origin"] = origin
        headers["Vary"] = "Origin"

    return headers


# Import commands to register them in the API process
try:
    logger.info("Commands imported in API process")
except Exception as e:
    logger.error(f"Failed to import commands in API process: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for the FastAPI application.
    Runs database migrations automatically on startup.
    """
    # Startup: Security checks
    logger.info("Starting API initialization...")

    # Security check: Encryption key
    if not get_secret_from_env("OPEN_NOTEBOOK_ENCRYPTION_KEY"):
        logger.warning(
            "OPEN_NOTEBOOK_ENCRYPTION_KEY not set. "
            "API key encryption will fail until this is configured. "
            "Set OPEN_NOTEBOOK_ENCRYPTION_KEY to any secret string."
        )

    # Run database migrations
    try:
        migration_manager = AsyncMigrationManager()
        current_version = await migration_manager.get_current_version()
        logger.info(f"Current database version: {current_version}")

        if await migration_manager.needs_migration():
            logger.warning("Database migrations are pending. Running migrations...")
            await migration_manager.run_migration_up()
            new_version = await migration_manager.get_current_version()
            logger.success(
                f"Migrations completed successfully. Database is now at version {new_version}"
            )
        else:
            logger.info(
                "Database is already at the latest version. No migrations needed."
            )
    except Exception as e:
        logger.error(f"CRITICAL: Database migration failed: {str(e)}")
        logger.exception(e)
        # Fail fast - don't start the API with an outdated database schema
        raise RuntimeError(f"Failed to run database migrations: {str(e)}") from e

    # Seed default admin user and assign unowned records
    try:
        from open_notebook.database.repository import ensure_record_id, repo_query
        from open_notebook.domain.user import User

        admin_email = os.getenv("OPEN_NOTEBOOK_ADMIN_EMAIL", "admin@opennotebook.local")
        admin_password = os.getenv("OPEN_NOTEBOOK_ADMIN_PASSWORD", "admin123")

        existing_users = await User.get_all()
        if not existing_users:
            admin = await User.create_user(
                email=admin_email,
                password=admin_password,
                name="Admin",
                role="admin",
            )
            logger.success(f"Created default admin user: {admin_email} / {admin_password}")
            existing_users = [admin]

        # Assign records with no user_id to the first admin
        admins = [u for u in existing_users if u.role == "admin"]
        if admins:
            admin_rid = ensure_record_id(str(admins[0].id))
            for table in ("notebook", "source", "note", "chat_session"):
                await repo_query(
                    f"UPDATE {table} SET user_id = $uid WHERE user_id IS NONE",
                    {"uid": admin_rid},
                )
            logger.success("Assigned unowned records to admin user")
    except Exception as e:
        logger.warning(f"User seeding / record assignment failed (non-fatal): {e}")

    # Run podcast profile data migration (legacy strings -> Model registry)
    try:
        from open_notebook.podcasts.migration import migrate_podcast_profiles

        await migrate_podcast_profiles()
    except Exception as e:
        logger.warning(f"Podcast profile migration encountered errors: {e}")
        # Non-fatal: profiles can be migrated manually via UI

    # Auto-configure default language model when ANTHROPIC_API_KEY is set
    try:
        from open_notebook.ai.models import DefaultModels, Model
        from open_notebook.database.repository import repo_query

        defaults = await DefaultModels.get_instance()
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        if not defaults.default_chat_model and anthropic_key:
            model_name = os.getenv(
                "OPEN_NOTEBOOK_DEFAULT_LANGUAGE_MODEL", "claude-3-5-sonnet-20241022"
            )

            existing = await repo_query(
                "SELECT * FROM model WHERE name = $name AND provider = 'anthropic' AND type = 'language' LIMIT 1",
                {"name": model_name},
            )

            if existing:
                model = Model(**existing[0])
                logger.info(f"Found existing model record for {model_name}")
            else:
                model = Model(name=model_name, provider="anthropic", type="language")
                await model.save()
                logger.info(f"Created default model record: {model_name}")

            defaults.default_chat_model = str(model.id)
            defaults.default_transformation_model = str(model.id)
            defaults.default_tools_model = str(model.id)
            await defaults.update()
            logger.success(f"Auto-configured {model_name} as default language model")
        elif defaults.default_chat_model:
            logger.info("Default language model already configured, skipping auto-setup")
        else:
            logger.info("No ANTHROPIC_API_KEY found, skipping auto-model setup")
    except Exception as e:
        logger.warning(f"Auto-configuring default model failed (non-fatal): {e}")

    # Fix model types that were incorrectly stored as 'language' due to a
    # registration bug (all discovered OpenAI models defaulted to 'language').
    # This corrects existing records based on well-known name patterns.
    #
    # Gated behind a meta flag so it only runs once: every subsequent startup
    # sees the flag and skips the full scan + Python-side pattern matching,
    # saving ~200–500ms on boot.
    try:
        from api.credentials_service import infer_model_type
        from open_notebook.database.repository import ensure_record_id, repo_query

        # Meta table row: `meta:model_type_fixup` with `done: bool, version: int`.
        # If we ever change the inference rules, bump FIXUP_VERSION to re-run.
        FIXUP_VERSION = 1

        flag_rows = await repo_query(
            "SELECT done, version FROM meta:model_type_fixup"
        )
        flag = flag_rows[0] if flag_rows else {}
        already_done = bool(flag.get("done")) and int(flag.get("version") or 0) >= FIXUP_VERSION

        if already_done:
            logger.info("Model type fixup: already applied, skipping scan")
        else:
            all_lang_models = await repo_query(
                "SELECT id, name, provider, type FROM model WHERE type = 'language'"
            )
            fixed = 0
            for m in all_lang_models:
                correct_type = infer_model_type(m.get("name", ""), m.get("provider", ""))
                if correct_type != "language":
                    await repo_query(
                        "UPDATE $id SET type = $type",
                        {"id": ensure_record_id(m["id"]), "type": correct_type},
                    )
                    fixed += 1

            if fixed:
                logger.success(
                    f"Model type fixup: corrected {fixed} model record(s) to their proper types"
                )
            else:
                logger.info("Model type fixup: all model types already correct")

            # Record completion so we don't scan again next boot
            await repo_query(
                "UPSERT meta:model_type_fixup CONTENT { done: true, version: $v }",
                {"v": FIXUP_VERSION},
            )
    except Exception as e:
        logger.warning(f"Model type fixup failed (non-fatal): {e}")

    logger.success("API initialization completed successfully")

    # Yield control to the application
    yield

    # Shutdown: close connection pool
    from open_notebook.database.repository import close_connection_pool
    await close_connection_pool()
    logger.info("API shutdown complete")


app = FastAPI(
    title="Open Notebook API",
    description="API for Open Notebook - Research Assistant",
    lifespan=lifespan,
)

if CORS_IS_DEFAULT_WILDCARD:
    logger.warning(
        "CORS_ORIGINS is not set — API accepts cross-origin requests from any "
        "origin (default: '*'). For production deployments, set CORS_ORIGINS to "
        "your frontend origin(s), e.g. "
        "CORS_ORIGINS=https://notebook.example.com"
    )
else:
    logger.info(f"CORS allowed origins: {CORS_ALLOWED_ORIGINS}")

# JWT authentication middleware — validates tokens on all non-public routes
app.add_middleware(JWTAuthMiddleware)

# Add CORS middleware last (so it processes first)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom exception handler to ensure CORS headers are included in error responses
# This helps when errors occur before the CORS middleware can process them
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Custom exception handler that ensures CORS headers are included in error responses.
    This is particularly important for 413 (Payload Too Large) errors during file uploads.

    Note: If a reverse proxy (nginx, traefik) returns 413 before the request reaches
    FastAPI, this handler won't be called. In that case, configure your reverse proxy
    to add CORS headers to error responses.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={**(exc.headers or {}), **_cors_headers(request)},
    )


@app.exception_handler(NotFoundError)
async def not_found_error_handler(request: Request, exc: NotFoundError):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc)},
        headers=_cors_headers(request),
    )


@app.exception_handler(InvalidInputError)
async def invalid_input_error_handler(request: Request, exc: InvalidInputError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
        headers=_cors_headers(request),
    )


@app.exception_handler(AuthenticationError)
async def authentication_error_handler(request: Request, exc: AuthenticationError):
    return JSONResponse(
        status_code=401,
        content={"detail": str(exc)},
        headers=_cors_headers(request),
    )


@app.exception_handler(RateLimitError)
async def rate_limit_error_handler(request: Request, exc: RateLimitError):
    return JSONResponse(
        status_code=429,
        content={"detail": str(exc)},
        headers=_cors_headers(request),
    )


@app.exception_handler(ConfigurationError)
async def configuration_error_handler(request: Request, exc: ConfigurationError):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
        headers=_cors_headers(request),
    )


@app.exception_handler(NetworkError)
async def network_error_handler(request: Request, exc: NetworkError):
    return JSONResponse(
        status_code=502,
        content={"detail": str(exc)},
        headers=_cors_headers(request),
    )


@app.exception_handler(ExternalServiceError)
async def external_service_error_handler(request: Request, exc: ExternalServiceError):
    return JSONResponse(
        status_code=502,
        content={"detail": str(exc)},
        headers=_cors_headers(request),
    )


@app.exception_handler(OpenNotebookError)
async def open_notebook_error_handler(request: Request, exc: OpenNotebookError):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers=_cors_headers(request),
    )


# Include routers
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(config.router, prefix="/api", tags=["config"])
app.include_router(notebooks.router, prefix="/api", tags=["notebooks"])
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(transformations.router, prefix="/api", tags=["transformations"])
app.include_router(notes.router, prefix="/api", tags=["notes"])
app.include_router(embedding.router, prefix="/api", tags=["embedding"])
app.include_router(
    embedding_rebuild.router, prefix="/api/embeddings", tags=["embeddings"]
)
app.include_router(settings.router, prefix="/api", tags=["settings"])
app.include_router(context.router, prefix="/api", tags=["context"])
app.include_router(sources.router, prefix="/api", tags=["sources"])
app.include_router(insights.router, prefix="/api", tags=["insights"])
app.include_router(commands_router.router, prefix="/api", tags=["commands"])
app.include_router(podcasts.router, prefix="/api", tags=["podcasts"])
app.include_router(episode_profiles.router, prefix="/api", tags=["episode-profiles"])
app.include_router(speaker_profiles.router, prefix="/api", tags=["speaker-profiles"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(source_chat.router, prefix="/api", tags=["source-chat"])
app.include_router(credentials.router, prefix="/api", tags=["credentials"])
app.include_router(languages.router, prefix="/api", tags=["languages"])
app.include_router(users.router, prefix="/api", tags=["users"])
app.include_router(sharepoint.router, prefix="/api", tags=["sharepoint"])


@app.get("/")
async def root():
    return {"message": "Open Notebook API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
