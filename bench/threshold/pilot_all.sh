#!/bin/bash
# pilot_all.sh: the pilot from one build (BUILD, default w76). The strong variant (layer 0 only) runs right after
# the first reference and must differ from it, otherwise the whole pilot stops.
set -eu
dir=$(cd "$(dirname "$0")" && pwd)
W76_WORK=${W76_WORK:-$dir/../../work/tmp/w76}
export BUILD=${BUILD:-w76}
runs=$W76_WORK/runs
"$dir/pilot.sh" ref1 TONA_IDX_SHARE=all --
"$dir/pilot.sh" l0 TONA_IDX_SHARE=0 --
python3 -I "$dir/check_differs.py" ref1 l0
"$dir/pilot.sh" closed TONA_IDX_SHARE=all -- --closed-book --families A
"$dir/pilot.sh" ref2 TONA_IDX_SHARE=all TONA_DRAFT_N=1 --
"$dir/pilot.sh" d8 TONA_IDX_SHARE=0,1,3,4,6,7,9,10 --
python3 -I "$dir/check_differs.py" ref1 d8 || echo "d8 identical to ref1: kept, but note it in the measurement file"
ls "$runs"
