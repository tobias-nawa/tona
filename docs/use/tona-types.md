# Tona's own tensor types

GGUF type ids from 1000 on are Tona's own. Each one stores the bytes of a ggml type in another order, so a backend
reads them with fewer, aligned loads; the values are the ggml type's bit for bit. Foreign loaders (llama.cpp, other
engines) do not know these ids and refuse such a file with an unknown-type error instead of misreading it. Files with
these types are made by Tona's tools for Tona's own backends and are not published as general GGUF files.

| Id | Name | Block | Made from | Read by |
|---|---|---|---|---|
| 1018 | `TONA_IQ3_XXS_R` | 512 weights, 196 bytes | IQ3_XXS (18), rows of a multiple of 512 weights | `model/quant.cpp` (CPU, `tona-check`'s reference), Metal (`kernels/decode.metal`, `IQ3XXSR`) |
| 1019 | `TONA_Q6_K_T` | tiles of 8 rows; per row 256 weights in 210 bytes, as Q6_K | Q6_K (14), 2-D layer matrices with rows a multiple of 8 | `model/quant_tile.cpp` (`dequantize_rows`), Metal, the HIP backend (`gpu/matvec_q8_tile.h`: int8 row kernel, WMMA kernel, prefill GEMM and dequantization; no float path) |

## TONA_IQ3_XXS_R

Two IQ3_XXS blocks of 98 bytes (fp16 scale `d`, 64 index bytes, 32 bytes of sign and scale words) become one
super-block:

| Bytes | Content |
|---|---|
| 0 to 127 | index bytes of both blocks; the two of group g (8 weights) of the super-block at 2g |
| 128 to 191 | the 16 sign and scale words of both blocks, one 32-bit word per 32 weights |
| 192 to 195 | `d` of the first and of the second block |

With 4-byte aligned rows (512 weights per super-block, 196 bytes) every field is read with one aligned load; IQ3_XXS
needs single-byte loads from 2-byte aligned blocks. Why: the expert mat-vec on the Mac is limited by integer work
and addressing, not by bytes (`research/measurements/2026-11-12-mac-expert-decode-probes.md`).

Made by `tools/coder_requant/pipeline.py repack --input FILE --out NEW` (the gate and up experts by default,
`--pattern` for other tensors): a copy of the file (an APFS clone on macOS) with the chosen tensors repacked in place
and their type fields set to 1018; sizes and offsets stay. Only the Metal backend and the CPU reference read the
type; the reference backend (HIP) and the CPU expert path do not.

## TONA_Q6_K_T

**A row is not contiguous.** 8 consecutive rows of k weights form one tile of 8 × k / 256 × 210 bytes (the size of
the 8 Q6_K rows, so the tensor keeps its size), and the weights of the 8 rows are interleaved in the order of an
8 × 8 simdgroup MMA fragment. Read such a tensor only with `dequantize_rows` (`model/quant.h`); `dequantize_row`
refuses the type, and any code that computes a row's address as `data + row * row_bytes` misreads it. The loader
(`model/gguf.cpp`) and the repack tool allow the type only for 2-D matrices of a layer (`blk.*`) with rows a multiple
of 8 and k a multiple of 256, never for the tables read by row (`token_embd`, `output`, `per_layer_token_embd`, the
MTP head's tables) or for experts.

Entry e = 32 c + lane (c the 32-weight chunk of the row, lane 0 to 31) holds 8 weights: group q = 2 · bit 3 + bit 0
of the lane (weights 32 c + 8 q to + 7) of row r = 4 · bit 4 + bit 1..2 of the lane. That is the element the lane
holds in Metal's `simdgroup_matrix` (ggml-metal's `mul_mv_mma` lane layout). In the tile:

| Bytes | Content |
|---|---|
| 0 to 4k − 1 | per entry one 32-bit word: the low 4 bits of weight i at bits 4i |
| 4k to 6k − 1 | per entry one 16-bit word: the high 2 bits of weight i at bits 2i |
| 6k to 6.5k − 1 | per entry pair (e, e + 1: the same row and 16 weights) the int8 scale |
| 6.5k to 6.5625k − 1 | fp16 `d` of row r's super-block s (256 weights) at 2 (8 s + r) |

The value is `d * scale * (code - 32)` as in Q6_K. Why: the MMA mat-vec for 4 tokens spent 11 of 35 ms decoding
GGUF blocks in the fragment layout; with Q6_K in this order it drops by about Q6_K's share, while the 1-token kernel
gets faster too (`research/measurements/2026-11-12-mac-dense-mma-probes.md`). A layout that only reorders inside a
row keeps 1 of the 5 ms. Made by `tools/coder_requant/pipeline.py repack --tile` (without `--pattern`: every Q6_K layer matrix that qualifies).
