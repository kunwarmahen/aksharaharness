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
# The same web UI can also live in a container, built and run by podman
# (or docker):
#
#   ./start.sh container-build                     build the image
#   ./start.sh container-start [local|cloud] [--port N]
#   ./start.sh container-stop / container-status / container-restart / container-logs
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
# Empty output = "no --port given" — callers then fall back to their default.
web_port_from_args() {
    local prev="" a
    for a in "$@"; do
        if [[ "$prev" == "--port" ]]; then echo "$a"; return; fi
        if [[ "$a" == --port=* ]]; then echo "${a#--port=}"; return; fi
        prev="$a"
    done
}

# Is anything listening on 127.0.0.1:PORT?
port_taken() {
    curl -s -o /dev/null -m 1 "http://127.0.0.1:$1/" 2>/dev/null
}

# Pick a publish port to use.
#   pick_port <N>           — explicit (e.g. --port N): honor it, refuse if busy.
#   pick_port - or (empty)  — no preference: default if free, else next free up.
# Prints the chosen port on stdout; reasons on stderr. Exits 1 on a conflict
# with an EXPLICIT port (we never override a user's choice).
pick_port() {
    local want="${1:-}"
    if [[ -n "$want" && "$want" != "-" ]]; then
        if port_taken "$want"; then
            echo "port $want is already in use by something else — this script will NOT kill it." >&2
            echo "pick a free one: ./start.sh ... --port 8322" >&2
            return 1
        fi
        echo "$want"; return 0
    fi
    if ! port_taken 8321; then
        echo 8321; return 0
    fi
    local probe
    for probe in 8322 8323 8325 8326 8327 8328 8329 8330 8331 8332 8333 8334 8335 8336 8337 8338 8339 8340 8341 8342; do
        if ! port_taken "$probe"; then
            echo "note: port 8321 is busy (your other server keeps it) — using $probe instead." >&2
            echo "$probe"; return 0
        fi
    done
    echo "no free port in 8321-8342 — pass one: ./start.sh ... --port N" >&2
    return 1
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
    if ! port="$(pick_port "$(web_port_from_args "$@")")"; then return 1; fi

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

# --- the web UI in a podman container ---------------------------------------
# The image carries code only (see Containerfile); keys come in at run time.
# The container lives in the podman daemon, so its "stop" and "is it up?"
# are asked of podman itself — no pid file needed.

CONTAINER_NAME="localhost/akshara-web"   # image name (podman's localhost/ tag)
CONTAINER_ID="akshara-web"               # running container name
CONTAINER_PORT_FILE=".akshara/akshara-web.port"

# podman, or docker if the host prefers that (README documents both).
container_engine() {
    if command -v podman >/dev/null 2>&1; then echo podman
    elif command -v docker >/dev/null 2>&1; then echo docker
    else
        echo "error: neither 'podman' nor 'docker' found on PATH." >&2
        echo "       (sudo apt install podman — or skip containers: './start.sh web-start')" >&2
        return 1
    fi
}

# True (0) if a container named akshara-web is RUNNING.
container_is_up() {
    local eng
    eng="$(container_engine)" || return 1
    "$eng" container inspect --format '{{.State.Running}}' "$CONTAINER_ID" 2>/dev/null \
        | grep -q true
}

# Stopped/exited containers still HOLD their name in podman ("name ... is
# already in use" on the next run) — so a container-stop followed by
# a container-start used to fail, until a retry or second attempt happened
# to land after cleanup. Remove any containers that own our name but are
# not running. Only ever touches our own name, never foreign containers.
container_reclaim_name() {
    local eng ids
    eng="$(container_engine)" || return 0
    ids="$("$eng" container ps -a --no-trunc --filter "name=^$CONTAINER_ID$" \
            --format '{{.ID}}  {{.State}}' 2>/dev/null \
            | awk 'tolower($2) !~ /running/ { print $1 }')"
    local id
    for id in $ids; do
        "$eng" container rm -f "$id" >/dev/null 2>&1 || true
        echo "(removed a stopped container of ours that still held the name: ${id:0:12})"
    done
    return 0
}

# True (0) if the image is already built.
container_image_ready() {
    local eng
    eng="$(container_engine)" || return 1
    "$eng" image exists "$CONTAINER_NAME" 2>/dev/null
}

# True (0) if something the image is built FROM is newer than the image.
# (Only a hint — podman builds are content-cached, so a "stale" image may
# still be perfectly current; this avoids asking twice for nothing.)
container_image_stale() {
    local eng img_ts code_ts f
    eng="$(container_engine)" || return 1
    img_ts="$("$eng" image inspect --format '{{.Created}}' "$CONTAINER_NAME" 2>/dev/null | head -1)"
    [[ "$img_ts" =~ ([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}) ]] || return 1
    img_ts="${BASH_REMATCH[1]}"
    code_ts=0
    for f in Containerfile .containerignore pyproject.toml uv.lock src; do
        [[ -e "$f" ]] && code_ts=$(( $(stat -c %Y "$f") > code_ts ? $(stat -c %Y "$f") : code_ts ))
    done
    code_ts=$(( code_ts / 3600 ))
    img_ts=$(( 10#$(date -u -d "$img_ts" +%s) / 3600 ))
    (( code_ts > img_ts ))
}

container_build() {
    local eng
    eng="$(container_engine)" || return 1
    # Build means BUILD. But if a container is already using the image,
    # ask before clobbering under its feet.
    if container_is_up; then
        local ans="n"
        [[ -t 0 ]] && read -rp "a container is running on this image — rebuild under its feet? [y/N] " ans
        case "${ans:-n}" in
            [yY]*) ;;
            *)
                echo "skipped — the running container keeps its image."
                echo "(stop it: ./start.sh container-stop, then container-build)"
                return 0
                ;;
        esac
    fi
    echo "building $CONTAINER_NAME (layers are cached — usually quick)"
    # --format docker: emits docker-format so the image's HEALTHCHECK survives.
    "$eng" build --format docker -t "$CONTAINER_NAME" . || {
        echo "build failed — fixing the Containerfile or your network, then retry." >&2
        return 1
    }
    echo "built."
    if container_is_up; then
        echo "(the running container still has the OLD image — ./start.sh container-restart to use the new one)"
    fi
}

container_start() {
    # Optional first word picks the model road: local | cloud (default).
    local flavor="cloud"
    case "${1:-}" in
        local|cloud) flavor="$1"; shift ;;
    esac

    local eng
    eng="$(container_engine)" || return 1

    if container_is_up; then
        echo "already running:"
        "$eng" container ps --filter "name=$CONTAINER_ID" --format '  {{.ID}}  {{.Ports}}  {{.Status}}'
        echo "(stop it first: ./start.sh container-stop)"
        return 0
    fi

    if ! container_image_ready; then
        echo "no image yet ($CONTAINER_NAME is not built)."
        if [[ -t 0 ]]; then
            read -rp "build it now? [Y/n] " ans || ans="n"
        else
            ans="y"
        fi
        case "${ans:-y}" in
            [nN]*)
                echo "building it first instead:"
                echo "  ./start.sh container-build"
                return 1
                ;;
        esac
        container_build || return 1
    elif container_image_stale; then
        echo "note: the image looks older than the code in this directory."
        echo "      (./start.sh container-build to rebuild — or ignore and use it as-is)"
    fi

    # A stopped container can still be sitting on the name (podman keeps them
    # until they are rm'd) — that would make 'container run' fail. Clear it.
    container_reclaim_name

    # Publish 8321 (fixed inside the container) on a free host port.
    # Default 8321; if your other server keeps it, we move up — never touch
    # whatever already owns 8321. An explicit --port is always honored.
    local port
    if ! port="$(pick_port "$(web_port_from_args "$@")")"; then return 1; fi

    local env_args=() cmd_args=()
    case "$flavor" in
        local)
            # Ollama stays on the host; containers reach it via its special name.
            env_args=(-e OLLAMA_BASE_URL=http://host.containers.internal:11434/v1
                      -e OLLAMA_MODEL="${OLLAMA_MODEL:-$(env_from_dotenv OLLAMA_MODEL)}")
            cmd_args=(--provider ollama)
            ;;
        cloud)
            # Mount .env read-only; keep-id lets the in-container user read
            # your 600 perms. (docker lacks keep-id — pass -e ANTHROPIC_API_KEY= instead.)
            if [[ ! -f .env ]]; then
                echo "no .env to mount — the container would boot keyless." >&2
                echo "   create .env (cp .env.example .env), or for now: ./start.sh web-start cloud" >&2
                return 1
            fi
            if [[ "$eng" == "podman" ]]; then
                env_args=(--userns=keep-id -v "$(pwd)/.env:/app/.env:ro")
            else
                env_args=(-v "$(pwd)/.env:/app/.env:ro")
            fi
            ;;
    esac

    # 8321 is fixed inside the image; publish it wherever we like.
    id="$("$eng" container run -d --name "$CONTAINER_ID" \
         -p "${port:-8321}:8321" "${env_args[@]}" \
         "$CONTAINER_NAME" --web --host 0.0.0.0 "${cmd_args[@]}" "$@")" || {
        echo "container run failed (output above)." >&2
        "$eng" container rm -f "$CONTAINER_ID" 2>/dev/null || true
        return 1
    }
    echo "started (container ${id:0:12})."

    # Wait until the UI actually answers.
    local i
    for i in $(seq 1 40); do
        if curl -s -o /dev/null -m 1 "http://127.0.0.1:$port/" 2>/dev/null; then
            echo "$port" >"$CONTAINER_PORT_FILE"
            echo "opened — http://127.0.0.1:$port in your browser"
            return 0
        fi
        if ! container_is_up; then
            echo "the container exited while booting. Its log:" >&2
            "$eng" container logs --tail 20 "$CONTAINER_ID" >&2
            return 1
        fi
        sleep 0.5
    done
    echo "still not answering after 20s — check: ./start.sh container-logs" >&2
    return 1
}

container_stop() {
    local eng
    eng="$(container_engine)" || return 1
    if container_is_up; then
        local id
        id="$("$eng" container stop --time 10 "$CONTAINER_ID")" || {
            echo "stop failed — force it: $eng container rm --force $CONTAINER_ID" >&2
            return 1
        }
        echo "stopped ($id)."
    else
        echo "not running."
    fi
    # Podman keeps stopped containers (and their name) around; drop the
    # record we don't need.
    rm -f "$CONTAINER_PORT_FILE"
    echo "(to fully delete the stopped container: ./start.sh container-rm)"
}

# Fully DELETE the container (only if it is not running). Frees the name.
container_rm() {
    local eng
    eng="$(container_engine)" || return 1
    if container_is_up; then
        echo "it is still running — stop it first: ./start.sh container-stop" >&2
        return 1
    fi
    if "$eng" container inspect "$CONTAINER_ID" >/dev/null 2>&1; then
        local id
        id="$("$eng" container rm -f "$CONTAINER_ID")"
        echo "removed ($id)."
    else
        echo "no stopped container to remove — none exists."
    fi
    rm -f "$CONTAINER_PORT_FILE"
}

container_status() {
    local eng
    eng="$(container_engine)" || return 1
    if container_image_ready; then
        echo "image:      $CONTAINER_NAME  (present)"
    else
        echo "image:      $CONTAINER_NAME  (NOT built — container-start will build it)"
    fi
    if container_is_up; then
        echo "container:  running"
        "$eng" container ps --filter "name=$CONTAINER_ID" --format '             {{.ID}}  {{.Ports}}  {{.Status}}'
        echo "             (stop: ./start.sh container-stop · logs: ./start.sh container-logs)"
    else
        rm -f "$CONTAINER_PORT_FILE"
        echo "container:  stopped.  (start: ./start.sh container-start)"
    fi
}

container_logs() {
    local eng
    eng="$(container_engine)" || return 1
    "$eng" container logs --tail "${1:-40}" "$CONTAINER_ID" || {
        echo "no such container yet — start it first: ./start.sh container-start" >&2
        return 1
    }
}

container_manage() {
    local cmd="$1"; shift
    case "$cmd" in
        container-build)    container_build ;;
        container-start)    container_start "$@" ;;
        container-stop)     container_stop ;;
        container-rm)       container_rm ;;
        container-status)   container_status ;;
        container-restart)  container_stop >/dev/null 2>&1; container_start "$@" ;;
        container-logs)     container_logs "$@" ;;
        *)                  echo "unknown command: $cmd (try './start.sh --help')" >&2; return 1 ;;
    esac
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

The same UI can also run inside a container (podman/docker):

  ./start.sh container-build    build the image (cached after the first time)
  ./start.sh container-start    run it detached — cloud key or Ollama both work
  ./start.sh container-stop     stop it
  ./start.sh container-status   is it running? is the image built?
  ./start.sh container-logs     watch its output

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
  6) container  the web UI inside a podman/docker container
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
        container-*)
            # Podman container image: build + run the web UI in a container
            container_manage "$@"
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
        read -rp "Pick 1-6 (or Ctrl-C to quit): " choice
        case "$choice" in
            1) preset="local" ;;
            2) preset="cloud" ;;
            3) preset="web" ;;
            4) preset="local-web" ;;
            5) preset="web-start" ;;
            6) preset="container" ;;
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

# 'container' (menu option 6): just start it; container_start offers to
# build the image first if it is missing.
if [[ "$preset" == "container" ]]; then
    container_start "$@"
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
