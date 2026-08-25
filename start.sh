#!/usr/bin/env bash
# start.sh — start Akshara in a few common ways without memorizing flags.
#
#   ./start.sh               show a numbered menu and pick one
#   ./start.sh local         free & private — runs on your machine (Ollama)
#   ./start.sh cloud         uses whichever key your .env has
#   ./start.sh web           the chat UI in your browser
#   ./start.sh local-web     local model + browser UI together
#
# Everything typed after the preset goes straight to akshara, so the rest
# of the CLI keeps working exactly as the README describes:
#
#   ./start.sh local --yolo
#   ./start.sh cloud --resume "finish the TODO cleanup"
#   ./start.sh local --model llama3.3
#
# The browser UI can also run detached from your terminal:
#
#   ./start.sh web-start [local|cloud] [--port N]    start in the background
#   ./start.sh web-stop / web-status / web-restart / web-logs
#
# Presets only pin what makes them different; keys, models and URLs still
# come from .env (real environment variables beat .env, as always).

set -euo pipefail
cd "$(dirname "$0")"

# Bookkeeping for the background browser UI (gitignored).
WEB_PID_FILE=".akshara/web-ui.pid"
WEB_PORT_FILE=".akshara/web-ui.port"
WEB_LOG_FILE=".akshara/web-ui.log"

# Read ONE variable out of .env without sourcing it (values may carry a
# trailing "# comment", which sourcing would choke on). Empty if absent.
env_from_dotenv() {
    sed -nE "s/^[[:space:]]*$1=([^#[:space:]]*).*/\1/p" .env 2>/dev/null | head -1
}

# Which cloud dialect can we actually authenticate with? Mirrors the CLI's
# own guess: Anthropic key first, then OpenAI-style. Empty string = none.
pick_cloud_provider() {
    if [[ -n "${ANTHROPIC_API_KEY:-}${ANTHROPIC_AUTH_TOKEN:-}" ]] \
        || [[ -n "$(env_from_dotenv ANTHROPIC_API_KEY)" ]] \
        || [[ -n "$(env_from_dotenv ANTHROPIC_AUTH_TOKEN)" ]]; then
        echo anthropic
    elif [[ -n "${OPENAI_API_KEY:-}" ]] || [[ -n "$(env_from_dotenv OPENAI_API_KEY)" ]]; then
        echo openai
    elif [[ -n "${RESPONSES_API_KEY:-}" ]] || [[ -n "$(env_from_dotenv RESPONSES_API_KEY)" ]]; then
        echo responses
    else
        echo ""
    fi
}

# Local preset: is an Ollama server actually up? Warn (don't block) if not,
# naming the two usual fixes — server not running, or tag never pulled.
check_ollama() {
    local base host model
    base="${OLLAMA_BASE_URL:-$(env_from_dotenv OLLAMA_BASE_URL)}"
    base="${base:-http://localhost:11434/v1}"
    host="${base%/v1}"
    model="${OLLAMA_MODEL:-$(env_from_dotenv OLLAMA_MODEL)}"
    model="${model:-qwen3.8}"
    if ! curl -s -o /dev/null -m 2 "$host" 2>/dev/null; then
        echo "warning: nothing answered at $host" >&2
        echo "         is Ollama running there? ('ollama serve') and is the" >&2
        echo "         model pulled? ('ollama pull $model')" >&2
        echo "         starting anyway..." >&2
    fi
}

command -v uv >/dev/null 2>&1 || {
    echo "error: 'uv' not found — this project installs its Python with uv." >&2
    echo "       Get it from https://docs.astral.sh/uv/, then: uv sync" >&2
    exit 1
}

# --- the browser UI, running in the background ------------------------------
# 'web-start' detaches the server so the terminal stays yours; a pid file in
# .akshara/ remembers it so web-stop/web-status can find it again.

web_is_running() {
    [[ -f "$WEB_PID_FILE" ]] || return 1
    local pid
    pid="$(cat "$WEB_PID_FILE" 2>/dev/null)"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

# Find --port N (or --port=N) among launch arguments; default matches akshara.
web_port_from_args() {
    local prev="" a
    for a in "$@"; do
        if [[ "$prev" == "--port" ]]; then echo "$a"; return; fi
        if [[ "$a" == --port=* ]]; then echo "${a#--port=}"; return; fi
        prev="$a"
    done
    echo "8321"
}

web_start() {
    # Optional first word picks the model road: local | cloud | auto (default).
    local flavor="auto"
    case "${1:-}" in
        auto|cloud|local|local-web) flavor="$1"; shift ;;
    esac

    if web_is_running; then
        echo "already running — pid $(cat "$WEB_PID_FILE"), \
http://127.0.0.1:$(cat "$WEB_PORT_FILE" 2>/dev/null || echo 8321)"
        echo "(stop it first: ./start.sh web-stop)"
        return 0
    fi

    local flags=()
    case "$flavor" in
        cloud)
            local prov
            prov="$(pick_cloud_provider)"
            if [[ -z "$prov" ]]; then
                echo "no cloud API key in the environment or .env —" >&2
                echo "try './start.sh web-start local' instead (no key needed)" >&2
                return 1
            fi
            flags=(--provider "$prov")
            ;;
        local|local-web)
            check_ollama
            flags=(--provider ollama)
            ;;
        auto) ;; # no --provider: akshara guesses from .env, like './start.sh web'
    esac
    flags+=(--web)

    local port
    port="$(web_port_from_args "$@")"

    # If the port already answers, a fresh server would just die on bind —
    # usually an old instance started by hand. Say so instead of failing
    # mysteriously in the log.
    if curl -s -o /dev/null -m 1 "http://127.0.0.1:$port/" 2>/dev/null; then
        echo "something is already serving on port $port." >&2
        echo "stop that first, or choose another: ./start.sh web-start --port 8322" >&2
        return 1
    fi

    mkdir -p .akshara
    echo "---- $(date '+%Y-%m-%d %H:%M:%S') starting: ${flags[*]} $*" >>"$WEB_LOG_FILE"
    nohup uv run akshara "${flags[@]}" "$@" >>"$WEB_LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" >"$WEB_PID_FILE"
    echo "$port" >"$WEB_PORT_FILE"

    # Wait until the UI actually answers (uv needs a few seconds to boot).
    local i
    for i in $(seq 1 40); do
        if curl -s -o /dev/null -m 1 "http://127.0.0.1:$port/" 2>/dev/null; then
            echo "started (pid $pid) — open http://127.0.0.1:$port in your browser"
            return 0
        fi
        kill -0 "$pid" 2>/dev/null || break # died while booting
        sleep 0.5
    done
    echo "the UI didn't come up — last lines of $WEB_LOG_FILE:" >&2
    tail -n 5 "$WEB_LOG_FILE" >&2
    return 1
}

web_stop() {
    if ! web_is_running; then
        rm -f "$WEB_PID_FILE" "$WEB_PORT_FILE" # clear a stale record, if any
        echo "not running."
        return 0
    fi
    local pid gone=0 i
    pid="$(cat "$WEB_PID_FILE")"
    pkill -P "$pid" 2>/dev/null || true # children (uv may wrap python)
    kill "$pid" 2>/dev/null || true
    for i in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || { gone=1; break; }
        sleep 0.25
    done
    if [[ "$gone" == 0 ]]; then # still alive after 5s: force it
        pkill -9 -P "$pid" 2>/dev/null || true
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$WEB_PID_FILE" "$WEB_PORT_FILE"
    echo "stopped."
}

web_status() {
    local probe_port
    probe_port="$(web_port_from_args "$@")"
    if web_is_running; then
        echo "running — pid $(cat "$WEB_PID_FILE"), \
http://127.0.0.1:$(cat "$WEB_PORT_FILE" 2>/dev/null || echo '?')"
        echo "           (stop: ./start.sh web-stop · logs: ./start.sh web-logs)"
    elif curl -s -o /dev/null -m 1 "http://127.0.0.1:$probe_port/" 2>/dev/null; then
        # Not ours, yet the port answers — likely an instance someone
        # started in the foreground. Say so instead of a bare "stopped".
        rm -f "$WEB_PID_FILE" "$WEB_PORT_FILE"
        echo "not managed here — but SOMETHING is already serving on port $probe_port,"
        echo "probably an 'akshara --web' started by hand. This script can't stop that one;"
        echo "find its terminal (or kill its pid) or just use it as-is."
    else
        rm -f "$WEB_PID_FILE" "$WEB_PORT_FILE"
        echo "stopped.  (start: ./start.sh web-start)"
    fi
}

web_logs() {
    [[ -f "$WEB_LOG_FILE" ]] || { echo "no log yet — start it first: ./start.sh web-start" >&2; return 1; }
    tail -n "${1:-40}" "$WEB_LOG_FILE" # Ctrl-C stops following
}

web_manage() {
    local cmd="$1"; shift
    case "$cmd" in
        web-start)   web_start "$@" ;;
        web-stop)    web_stop ;;
        web-status)  web_status ;;
        web-restart) web_stop >/dev/null 2>&1; web_start "$@" ;;
        web-logs)    web_logs "$@" ;;
        *)           echo "unknown command: $cmd (try './start.sh --help')" >&2; return 1 ;;
    esac
}

usage() {
    cat <<'EOF'
Start Akshara with one command:

  ./start.sh              menu — pick a setup by number
  ./start.sh local        free & private — Ollama on this machine, no key
  ./start.sh cloud        uses whichever API key your .env has
  ./start.sh web          the chat UI in your browser (provider auto-picked)
  ./start.sh local-web    local model + browser UI

The browser UI can also run in the background, out of your terminal:

  ./start.sh web-start    start it detached ('web-start local' for Ollama,
                          'web-start cloud' to pin the cloud road)
  ./start.sh web-stop     stop it again
  ./start.sh web-status   is it running? where?
  ./start.sh web-restart  stop, then start
  ./start.sh web-logs     show what it has been saying

Anything after a preset or command passes through to akshara:

  ./start.sh local --yolo            no permission prompts (careful)
  ./start.sh cloud --resume          restore the newest checkpoint
  ./start.sh local --model qwen3.8   any tag you have pulled
  ./start.sh web-start --port 9000   different port for the UI

Keys and models come from .env — edit it to change them, or override on
the command line like any akshara flag.
EOF
}

# Turn a preset name into the akshara arguments that express it.
# Prints nothing on an unknown name (caller reports that).
flags_for() {
    case "$1" in
        local)
            check_ollama
            printf '%s\n' --provider ollama
            ;;
        cloud)
            local prov
            prov="$(pick_cloud_provider)"
            if [[ -z "$prov" ]]; then
                echo "no_key" # sentinel; caller turns it into advice
            else
                printf '%s\n' --provider "$prov"
            fi
            ;;
        web)
            printf '%s\n' --web
            ;;
        local-web)
            check_ollama
            printf '%s\n' --provider ollama --web
            ;;
    esac
}

menu_text() {
    cat <<'EOF'
How do you want to run Akshara?

  1) local      free & private — Ollama on this machine, no key needed
  2) cloud      use the API key already in your .env
  3) web        chat in your browser instead of the terminal
  4) local-web  local model, but in the browser
  5) web-start  browser chat in the background (./start.sh web-stop ends it)
EOF
}

# --- argument handling ------------------------------------------------------

if [[ $# -gt 0 ]]; then
    case "$1" in
        -h|--help|help|list|ls)
            usage
            exit 0
            ;;
        web-*)
            # Background browser-UI management (start/stop/status/restart/logs)
            web_manage "$@"
            exit $?
            ;;
        *)
            preset="$1"
            shift
            ;;
    esac
else
    menu_text
    if [[ -t 0 ]]; then
        read -rp "Pick 1-5 (or Ctrl-C to quit): " choice
        case "$choice" in
            1) preset="local" ;;
            2) preset="cloud" ;;
            3) preset="web" ;;
            4) preset="local-web" ;;
            5) preset="web-start" ;;
            *) echo "not a choice: $choice" >&2; exit 1 ;;
        esac
    else
        echo "(run './start.sh --help' for the preset list)" >&2
        exit 1
    fi
fi

# The detached browser UI has its own machinery (pid file, log, health
# check) — hand it over before the foreground launch path below.
if [[ "$preset" == "web-start" ]]; then
    web_start "$@"
    exit $?
fi

mapfile -t flags < <(flags_for "$preset") || true
if [[ ${#flags[@]} -eq 1 && "${flags[0]}" == "no_key" ]]; then
    echo "error: no cloud API key found in the environment or .env." >&2
    echo "       Copy .env.example to .env and fill in a key — or just go" >&2
    echo "       local, which needs no key at all:" >&2
    echo "" >&2
    echo "           ./start.sh local" >&2
    exit 1
fi
if [[ ${#flags[@]} -eq 0 ]]; then
    echo "unknown preset: $preset (try './start.sh --help')" >&2
    exit 1
fi

exec uv run akshara "${flags[@]}" "$@"
