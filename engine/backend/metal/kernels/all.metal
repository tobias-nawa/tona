// The one translation unit of the Metal kernels. Order matters: tables, decoders, then the ops that use them. The
// build compiles this file with xcrun metal; TONA_METAL_SOURCES expands the includes at run time (device.mm).
// Both compile without fast math: with it, the compiler contracts and reorders the sums of the batched mat-vecs
// differently for each token count B, and passes of different width lose bit-identity (126 of 129 rows on the
// full-size shapes; research/measurements/2026-11-11-mac-synth.md).
#include <metal_stdlib>
using namespace metal;
// clang-format off
#include "params.h"
#include "quant_tables.metal"
#include "decode.metal"
#include "matvec.metal"
#include "matvec_mma.metal"
#include "gemm.metal"
#include "moe_table.metal"
#include "ops.metal"
#include "deltanet.metal"
#include "qsa.metal"
#include "qsa_prefill.metal"
#include "moe.metal"
#include "ple.metal"
#include "mtp_pick.metal"
// clang-format on
