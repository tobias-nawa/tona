// GPU memory bandwidth while the SSD streams expert-sized blocks (O1), and GPU kernel time after idle gaps (O2).
// Plan and evaluation: research/measurements/2026-11-11-mac-overlap.md
// Build: swiftc -O bench/mac/overlap.swift -o work/build/mac/overlap
// Usage: overlap o1 FILE   (FILE: test file of several GiB, e.g. from ssdbench write)
//        overlap o2
//        overlap o2b
import Foundation
import Metal

let src = """
#include <metal_stdlib>
using namespace metal;
kernel void rd(device const uint4* a [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& n [[buffer(2)]],
               uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
    uint4 s = 0;
    for (uint i = gid; i < n; i += gsz) s += a[i];
    if ((s.x ^ s.y ^ s.z ^ s.w) == 0x9e3779b9u) out[0] = s.x;
}
kernel void alu(device float* out [[buffer(1)]], constant uint& n [[buffer(2)]],
                uint gid [[thread_position_in_grid]]) {
    float x = float(gid) * 1e-7f, y = 1.0f;
    for (uint i = 0; i < n; ++i) { y = fma(y, 0.999999f, x); x = fma(x, 1.000001f, y * 1e-9f); }
    if (y == 12345.0f) out[0] = x;
}
"""

func fail(_ m: String) -> Never {
    FileHandle.standardError.write((m + "\n").data(using: .utf8)!)
    exit(1)
}

func emit(_ rec: [String: Any]) {
    let d = try! JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys])
    print(String(data: d, encoding: .utf8)!)
}

func median(_ v: [Double]) -> Double { v.isEmpty ? 0 : v.sorted()[v.count / 2] }
func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

let dev = MTLCreateSystemDefaultDevice() ?? { fail("no device") }()
let queue = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let rdPipe = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "rd")!)
let aluPipe = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "alu")!)
let sink = dev.makeBuffer(length: 16, options: .storageModeShared)!
let loadBytes = 1 << 30
let loadBuf = dev.makeBuffer(length: loadBytes, options: .storageModeShared)!
memset(loadBuf.contents(), 1, loadBytes)

// one pass over the 1 GiB buffer, small grid (best in 2026-11-11-mac-bandwidth)
func encodeRead(_ cb: MTLCommandBuffer, passes: Int) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(rdPipe)
    enc.setBuffer(loadBuf, offset: 0, index: 0)
    enc.setBuffer(sink, offset: 0, index: 1)
    var n = UInt32(loadBytes / 16)
    enc.setBytes(&n, length: 4, index: 2)
    for _ in 0..<passes {
        enc.dispatchThreads(MTLSize(width: 24 * 4 * 256, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    enc.endEncoding()
}

func encodeAlu(_ cb: MTLCommandBuffer, iters: UInt32) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(aluPipe)
    enc.setBuffer(sink, offset: 0, index: 1)
    var n = iters
    enc.setBytes(&n, length: 4, index: 2)
    enc.dispatchThreads(MTLSize(width: 24 * 1024, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding()
}

func gpuSec(_ cb: MTLCommandBuffer) -> Double { cb.gpuEndTime - cb.gpuStartTime }

// SSD reader: QD threads, one pread of BLOCK bytes each, random block-aligned offsets, F_NOCACHE
final class SSDReader {
    let fd: Int32, blocks: Int, block = 2_046_000, qd = 16
    var running = false
    var bytes = [Int](repeating: 0, count: 16)
    var threads: [Thread] = []
    let lock = NSLock()

    init(path: String) {
        fd = open(path, O_RDONLY)
        if fd < 0 { fail("open \(path)") }
        _ = fcntl(fd, F_NOCACHE, 1)
        _ = fcntl(fd, F_RDAHEAD, 0)
        var st = stat()
        fstat(fd, &st)
        blocks = Int(st.st_size) / block
        if blocks < 1000 { fail("test file too small") }
    }

    func start() {
        running = true
        bytes = [Int](repeating: 0, count: qd)
        threads = (0..<qd).map { t in
            let th = Thread { [self] in
                let buf = UnsafeMutableRawPointer.allocate(byteCount: block, alignment: 16384)
                var s = UInt64(0x9e3779b97f4a7c15) &+ UInt64(t) &* 0x1000193
                var done = 0
                while running {
                    s ^= s << 13; s ^= s >> 7; s ^= s << 17
                    let off = off_t(Int(s % UInt64(blocks)) * block)
                    let r = pread(fd, buf, block, off)
                    if r != block { fail("pread \(r)") }
                    done += r
                }
                lock.lock(); bytes[t] = done; lock.unlock()
                buf.deallocate()
            }
            th.start()
            return th
        }
    }

    // stops the readers and returns the bytes read
    func stop() -> Int {
        running = false
        while true {
            lock.lock()
            let finished = threads.allSatisfy { $0.isFinished }
            lock.unlock()
            if finished { break }
            usleep(1000)
        }
        return bytes.reduce(0, +)
    }
}

// GPU read bandwidth in GB/s per command buffer of 4 passes (~13 ms) for `seconds`
func gpuPhase(seconds: Double) -> [Double] {
    var out: [Double] = []
    let end = now() + seconds
    while now() < end {
        let cb = queue.makeCommandBuffer()!
        encodeRead(cb, passes: 4)
        cb.commit()
        cb.waitUntilCompleted()
        out.append(Double(4 * loadBytes) / gpuSec(cb) / 1e9)
    }
    return out
}

func o1(_ path: String) {
    let ssd = SSDReader(path: path)
    _ = gpuPhase(seconds: 0.3)  // warm-up
    let gpuAlone = gpuPhase(seconds: 2)
    var t0 = now()
    ssd.start()
    let gpuDuring = gpuPhase(seconds: 2)
    let ssdDuring = Double(ssd.stop()) / (now() - t0) / 1e9
    t0 = now()
    ssd.start()
    usleep(2_000_000)
    let ssdAlone = Double(ssd.stop()) / (now() - t0) / 1e9
    let gpuAfter = gpuPhase(seconds: 2)
    emit(["kind": "overlap", "block": "O1", "unit": "GB/s",
          "gpu_alone_median": median(gpuAlone), "gpu_during_ssd_median": median(gpuDuring),
          "gpu_after_median": median(gpuAfter), "gpu_during_min": gpuDuring.min() ?? 0,
          "n_cb_during": gpuDuring.count, "ssd_during_gpu": ssdDuring, "ssd_alone": ssdAlone])
}

// probe = ALU kernel of ~0.5 ms at full clock; GPU time of the probe after a busy phase and an idle gap
func o2() {
    let iters: UInt32 = 4000
    func probe() -> Double {
        let cb = queue.makeCommandBuffer()!
        encodeAlu(cb, iters: iters)
        cb.commit()
        cb.waitUntilCompleted()
        return gpuSec(cb) * 1e6
    }
    func busy() {
        let cb = queue.makeCommandBuffer()!
        encodeRead(cb, passes: 4)  // ~13 ms of load
        cb.commit()
        cb.waitUntilCompleted()
    }
    busy()
    for gapMs in [0.0, 0.1, 0.3, 1, 3, 10, 30, 100] {
        var t: [Double] = []
        for _ in 0..<20 {
            busy()
            if gapMs > 0 { usleep(UInt32(gapMs * 1000)) }
            t.append(probe())
        }
        emit(["kind": "overlap", "block": "O2", "gap_ms": gapMs, "unit": "us probe GPU time",
              "median": median(t), "min": t.min()!, "max": t.max()!])
    }
}

// after `idleMs` of idle: 40 probes in their own command buffers, all committed at once; GPU time of each, so the
// clock ramp shows as the probe index where the time returns to the busy level
func o2b() {
    for idleMs in [100.0, 1000] {
        for rep in 0..<3 {
            usleep(UInt32(idleMs * 1000))
            let cbs: [MTLCommandBuffer] = (0..<40).map { _ in
                let cb = queue.makeCommandBuffer()!
                encodeAlu(cb, iters: 4000)
                cb.commit()
                return cb
            }
            cbs.last!.waitUntilCompleted()
            let t = cbs.map { gpuSec($0) * 1e6 }
            let t0 = cbs[0].gpuStartTime
            emit(["kind": "overlap", "block": "O2b", "idle_ms": idleMs, "rep": rep, "unit": "us probe GPU time",
                  "probe_us": t.map { ($0 * 10).rounded() / 10 },
                  "start_ms": cbs.map { (($0.gpuStartTime - t0) * 1e4).rounded() / 10 }])
        }
    }
}

let args = CommandLine.arguments
switch args.count > 1 ? args[1] : "" {
case "o1":
    if args.count < 3 { fail("o1 needs FILE") }
    o1(args[2])
case "o2": o2()
case "o2b": o2b()
default: fail("usage: overlap o1 FILE | o2")
}
