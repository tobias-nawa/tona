#!/usr/bin/env bash
# The pictures of docs/images/ (scripts/screenshot-tui.py), one command each; every picture is written as SVG and as
# PNG (the SVG rasterized with ImageMagick's RSVG delegate at 192 dpi):
#   scripts/make-screenshots.sh installer       installer-<page>.svg for the pages machine, tasks, languages,
#                                               models, details, help, room, go
#   scripts/make-screenshots.sh client-stub     tona-<page>.svg for the pages start, models, agents, settings,
#                                               help, from the recorded /status
#                                               (scripts/status-stub.py, port 8674)
#   scripts/make-screenshots.sh client-live     tona-start.svg from the real server: PORT (default 8662), no stub
# Binaries: TONA_INSTALL_BIN (default work/build/engine/tona; run as `tona setup`), TONA_CLIENT_BIN (default work/build/engine/tona).
# The installer runs in a throwaway HOME and install folder and stops before the download, so it changes nothing on this
# machine. The pages "server cannot be reached" and "update" show paths of this checkout or need a release install:
# they are not photographed (the picture tool refuses a screen with a real path).
set -euo pipefail
cd "$(dirname "$0")/.."

py="${TONA_SHOT_PYTHON:-work/venv-screenshot/bin/python}"
install_bin="${TONA_INSTALL_BIN:-work/build/engine/tona}"
client_bin="${TONA_CLIENT_BIN:-work/build/engine/tona}"
out="${TONA_SHOT_OUT:-docs/images}"
shot=("$py" scripts/screenshot-tui.py --size 80x24 --squeeze 1)
if [ -n "${TONA_SHOT_HOME_PARENT:-}" ]; then
    shot+=(--home-parent "$TONA_SHOT_HOME_PARENT")
fi

if [ ! -x "$py" ]; then
    python3 -m venv work/venv-screenshot
    work/venv-screenshot/bin/pip install -q --require-hashes -r scripts/screenshot-requirements.txt
fi

png() {
    # 192 dpi: the 80 x 24 terminal comes out about 1400 px wide, sharp on a README
    magick -density 192 "RSVG:$1" "${1%.svg}.png"
}

installer() {
    # Enter x2: Machine -> Tasks (coding and knowledge are on) -> Languages; Space, right, Space: Python and TypeScript;
    # Enter: the page of the models
    local models="enter enter space right space enter"
    local -A keys=([machine]="" [tasks]="enter" [languages]="enter enter" [models]="$models" [details]="$models i"
        [help]="$models ?" [room]="$models enter" [go]="$models enter enter")
    for page in machine tasks languages models details help room go; do
        "${shot[@]}" --lang en_US.UTF-8 --keys "${keys[$page]}" --out "$out/installer-$page.svg" \
            -- "$install_bin" setup --release --install-dir '{home}/tona'
        png "$out/installer-$page.svg"
    done
}

client() {
    # 20 s of waiting on the first page: the write-rate line fills up (one point per poll, 2 minutes of history)
    local url="$1"
    # The Agents page shows the working folder (here a throwaway one): replace it with a neutral path first
    local erase
    erase=$(printf 'backspace %.0s' $(seq 1 60))
    local -A keys=([start]="wait:20" [models]="wait:1 2" [agents]="wait:1 3 tab ${erase}text:~/projects/shop"
        [settings]="wait:1 4" [help]="wait:1 ?")
    for page in start models agents settings help; do
        "${shot[@]}" --lang en_US.UTF-8 --keys "${keys[$page]}" --out "$out/tona-$page.svg" \
            -- "$client_bin" --url "$url" --tab start
        png "$out/tona-$page.svg"
    done
}

case "${1:-}" in
installer)
    installer
    ;;
client-stub)
    python3 scripts/status-stub.py --port 8674 &
    stub=$!
    trap 'kill "$stub"' EXIT
    sleep 1
    client http://127.0.0.1:8674
    ;;
client-live)
    "${shot[@]}" --keys "wait:20" --out "${OUT:-$out/tona-start.svg}" \
        -- "$client_bin" --url "http://127.0.0.1:${PORT:-8662}" --tab start
    png "${OUT:-$out/tona-start.svg}"
    ;;
*)
    echo "usage: $0 installer|client-stub|client-live" >&2
    exit 2
    ;;
esac
