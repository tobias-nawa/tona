// Device code of the matvec kernels: row functions (float and int8 path), tables, chain phases and the grid barrier.
// Included by gpu/matvec.hip, the kernel units of the architectures (gpu/arch/*/kernel.hip), gpu/gemm.hip and
// gpu/prefill.hip. Everything is in the anonymous namespace: every translation unit has its own copies of the tables in
// device memory and uploads them with upload_tables() itself. Device functions live in headers of the same translation
// unit (there is no -fgpu-rdc), so that splitting costs no speed.
//
//   matvec_tables.h       constants, tables, block sizes, upload
//   matvec_dot_sub.h      float path: dot product of one sub-block per format
//   matvec_rows_float.h   float path: row functions
//   matvec_q8_core.h      int8 path: dot4, Dec, table struct
//   matvec_q8_kquants.h   int8 path: unpacking of Q8_0, Q2_0, K-quants, IQ4
//   matvec_q8_iq.h        int8 path: unpacking of IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S
//   matvec_q8_nvfp4.h     int8 path: unpacking of NVFP4
//   matvec_q8_row.h       int8 path: row function
//   matvec_task.h         DevTask, Batch, unit resolution
//   matvec_loops.h        unit loops
//   matvec_wide.h         wide passes (WMMA)
//   matvec_phase.h        run_phase
//   matvec_barrier.h      grid barrier
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "gpu/matvec.h"
#include "model/quant_tables.h"

// Not every translation unit uses every function (e.g. dot_sub<14> only with TONA_MATVEC_V1)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"

#include "gpu/matvec_barrier.h"
#include "gpu/matvec_dot_sub.h"
#include "gpu/matvec_loops.h"
#include "gpu/matvec_phase.h"
#include "gpu/matvec_q8_core.h"
#include "gpu/matvec_q8_iq.h"
#include "gpu/matvec_q8_kquants.h"
#include "gpu/matvec_q8_nvfp4.h"
#include "gpu/matvec_q8_row.h"
#include "gpu/matvec_rows_float.h"
#include "gpu/matvec_tables.h"
#include "gpu/matvec_task.h"
#include "gpu/matvec_wide.h"

#pragma clang diagnostic pop
