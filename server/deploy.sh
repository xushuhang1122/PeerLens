#!/usr/bin/env bash
# PaperRadar MCP Server — one-click deploy (Ubuntu/Debian)
#
# Usage:
#   sudo EMBEDDING_API_KEY=sk-xxx bash server/deploy.sh
#
# With nginx + HTTPS:
#   sudo EMBEDDING_API_KEY=sk-xxx DOMAIN=mcp.example.com ENABLE_HTTPS=1 bash server/deploy.sh
#
# Configuration via env vars (all optional except EMBEDDING_API_KEY):
#   EMBEDDING_API_KEY   embedding API key (required)
#   EMBEDDING_BASE_URL  custom endpoint, e.g. https://api.openai.com/v1
#   EMBEDDING_MODEL     default: text-embedding-3-large
#   DOMAIN              public domain name; omit to skip nginx
#   ENABLE_HTTPS        set to 1 to run certbot (requires DOMAIN + DNS pointing here)
#   MCP_PORT            default: 8765
#   APP_USER            system user to run the service; default: paperradar
#   INSTALL_DIR         project root; default: current directory
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
INSTALL_DIR="${INSTALL_DIR:-$(pwd)}"
MCP_PORT="${MCP_PORT:-8765}"
APP_USER="${APP_USER:-paperradar}"
DOMAIN="${DOMAIN:-}"
ENABLE_HTTPS="${ENABLE_HTTPS:-0}"
EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-large}"

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Pre-flight checks ──────────────────────────────────────────────────────────
[[ $EUID -ne 0 ]]                          && die "Run with sudo."
[[ -z "$EMBEDDING_API_KEY" ]]              && die "EMBEDDING_API_KEY is required."
[[ ! -f "$INSTALL_DIR/server/mcp_server.py" ]] \
                                           && die "Run from the project root (got: $INSTALL_DIR)."
command -v python3 &>/dev/null             || die "python3 not found."

if [[ ! -d "$INSTALL_DIR/data/chroma_db" ]]; then
    warn "data/chroma_db not found. Copy your data/ directory before starting."
    warn "Example: rsync -av data/ user@server:${INSTALL_DIR}/data/"
fi

# ── System user ────────────────────────────────────────────────────────────────
if ! id "$APP_USER" &>/dev/null; then
    info "Creating system user: $APP_USER"
    useradd --system --no-create-home --shell /sbin/nologin "$APP_USER"
fi
chown -R "$APP_USER:$APP_USER" "$INSTALL_DIR"

# ── Python venv ────────────────────────────────────────────────────────────────
VENV="$INSTALL_DIR/venv"
info "Setting up venv at $VENV ..."
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r "$INSTALL_DIR/server/requirements.txt"
info "Python deps installed."

# ── .env file ──────────────────────────────────────────────────────────────────
ENV_FILE="$INSTALL_DIR/.env"
# Bind to loopback if nginx will front it, public otherwise.
if [[ -n "$DOMAIN" ]]; then
    MCP_HOST_BIND="127.0.0.1"
else
    MCP_HOST_BIND="0.0.0.0"
fi

if [[ ! -f "$ENV_FILE" ]]; then
    info "Writing $ENV_FILE"
    {
        echo "EMBEDDING_API_KEY=${EMBEDDING_API_KEY}"
        echo "EMBEDDING_MODEL=${EMBEDDING_MODEL}"
        [[ -n "$EMBEDDING_BASE_URL" ]] && echo "EMBEDDING_BASE_URL=${EMBEDDING_BASE_URL}"
        echo "MCP_HOST=${MCP_HOST_BIND}"
        echo "MCP_PORT=${MCP_PORT}"
    } > "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    chown "$APP_USER:$APP_USER" "$ENV_FILE"
else
    warn "$ENV_FILE already exists — not overwriting. Edit it manually if needed."
fi

# ── systemd service ────────────────────────────────────────────────────────────
SERVICE="paperradar-mcp"
SERVICE_FILE="/etc/systemd/system/${SERVICE}.service"
info "Installing systemd service: $SERVICE_FILE"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=PaperRadar MCP Server
After=network.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=${INSTALL_DIR}/.env
ExecStart=${VENV}/bin/python server/mcp_server.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE"
systemctl restart "$SERVICE"
info "Service started. Logs: sudo journalctl -u $SERVICE -f"

# ── nginx (only when DOMAIN is set) ───────────────────────────────────────────
if [[ -z "$DOMAIN" ]]; then
    echo ""
    echo "  No DOMAIN set — nginx skipped."
    echo "  MCP server listening on 0.0.0.0:${MCP_PORT}"
    echo ""
    echo "  Client .env:  REMOTE_MCP_URL=http://<your-server-ip>:${MCP_PORT}/mcp"
    echo "  Logs:         sudo journalctl -u $SERVICE -f"
    exit 0
fi

command -v nginx &>/dev/null || { info "Installing nginx ..."; apt-get install -y nginx; }

NGINX_CONF="/etc/nginx/sites-available/paperradar-mcp"
info "Writing nginx config for $DOMAIN"
cat > "$NGINX_CONF" <<EOF
server {
    listen 80;
    server_name ${DOMAIN};

    location /mcp {
        proxy_pass          http://127.0.0.1:${MCP_PORT};
        proxy_http_version  1.1;
        # Keep the SSE / streaming connection alive
        proxy_set_header    Connection "";
        proxy_set_header    Host \$host;
        proxy_set_header    X-Real-IP \$remote_addr;
        proxy_read_timeout  300s;
        proxy_send_timeout  300s;
        proxy_buffering     off;
    }
}
EOF

ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/paperradar-mcp
nginx -t
systemctl reload nginx

# ── HTTPS via certbot ─────────────────────────────────────────────────────────
if [[ "$ENABLE_HTTPS" == "1" ]]; then
    command -v certbot &>/dev/null \
        || { info "Installing certbot ..."; apt-get install -y certbot python3-certbot-nginx; }
    CERTBOT_EMAIL="admin@${DOMAIN}"
    certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$CERTBOT_EMAIL"
    MCP_URL="https://${DOMAIN}/mcp"
    info "HTTPS certificate issued."
else
    MCP_URL="http://${DOMAIN}/mcp"
    warn "HTTPS not enabled. Run with ENABLE_HTTPS=1 for a Let's Encrypt cert."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "  Deploy complete."
echo ""
echo "  MCP endpoint:   $MCP_URL"
echo "  Client .env:    REMOTE_MCP_URL=$MCP_URL"
echo ""
echo "  Manage service:"
echo "    sudo systemctl status  $SERVICE"
echo "    sudo systemctl restart $SERVICE"
echo "    sudo journalctl -u $SERVICE -f"
