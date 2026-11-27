// GPU memory bandwidth on Apple silicon: compute kernels read, write and copy, plus a blit copy, over shared and
// private buffers. Time from the command buffer's GPU timestamps, one dispatch per command buffer.
// Plan and evaluation: research/measurements/2026-11-11-mac-bandwidth.md
// Build: swiftc -O bench/mac/gpubw.swift -o work/build/mac/gpubw
// Usage: gpubw [--info] [MiB per buffer, default 1024] [passes, default 10]
import Foundation
import Metal

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

// The xor test is never true for the fill pattern; it keeps the compiler from dropping the loads
kernel void rd(device const uint4* a [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& n [[buffer(2)]],
               uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
    uint4 s = 0;
    for (uint i = gid; i < n; i += gsz) s += a[i];
    if ((s.x ^ s.y ^ s.z ^ s.w) == 0x9e3779b9u) out[0] = s.x;
}

kernel void wr(device uint4* a [[buffer(0)]], constant uint& n [[buffer(2)]],
               uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
    for (uint i = gid; i < n; i += gsz) a[i] = uint4(i);
}

kernel void cp(device const uint4* a [[buffer(0)]], device uint4* b [[buffer(1)]], constant uint& n [[buffer(2)]],
               uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
    for (uint i = gid; i < n; i += gsz) b[i] = a[i];
}
"""

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(1)
}

func printInfo(_ dev: MTLDevice) {
    var families: [String] = []
    let known: [(String, MTLGPUFamily)] = [
        ("apple7", .apple7), ("apple8", .apple8), ("apple9", .apple9), ("metal3", .metal3),
    ]
    for (name, fam) in known where dev.supportsFamily(fam) { families.append(name) }
    let info: [String: Any] = [
        "kind": "info", "name": dev.name, "architecture": dev.architecture.name,
        "families": families, "unified": dev.hasUnifiedMemory,
        "recommended_max_working_set": dev.recommendedMaxWorkingSetSize,
        "max_buffer_length": dev.maxBufferLength,
        "max_threads_per_threadgroup": dev.maxThreadsPerThreadgroup.width,
        "ram": ProcessInfo.processInfo.physicalMemory,
        "os": ProcessInfo.processInfo.operatingSystemVersionString,
    ]
    let data = try! JSONSerialization.data(withJSONObject: info, options: [.sortedKeys])
    print(String(data: data, encoding: .utf8)!)
}

struct Bench {
    let dev: MTLDevice
    let queue: MTLCommandQueue
    let pipes: [String: MTLComputePipelineState]
    let bytes: Int

    // One command buffer per pass; GPU time of each pass in seconds
    func run(passes: Int, encode: (MTLCommandBuffer) -> Void) -> [Double] {
        var times: [Double] = []
        for _ in 0..<passes {
            guard let cb = queue.makeCommandBuffer() else { fail("no command buffer") }
            encode(cb)
            cb.commit()
            cb.waitUntilCompleted()
            if let err = cb.error { fail("command buffer: \(err)") }
            times.append(cb.gpuEndTime - cb.gpuStartTime)
        }
        return times
    }

    func dispatch(_ cb: MTLCommandBuffer, _ kernel: String, _ bufs: [MTLBuffer], grid: Int) {
        guard let enc = cb.makeComputeCommandEncoder(), let pipe = pipes[kernel] else { fail("encoder") }
        enc.setComputePipelineState(pipe)
        for (i, b) in bufs.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
        var n = UInt32(bytes / 16)
        enc.setBytes(&n, length: 4, index: 2)
        let tg = 256
        enc.dispatchThreads(MTLSize(width: grid, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
    }
}

func report(_ test: String, _ mode: String, grid: Int, moved: Int, _ times: [Double], warmup: Int) {
    let t = Array(times.dropFirst(warmup)).sorted()
    let gbs = t.map { Double(moved) / $0 / 1e9 }
    let rec: [String: Any] = [
        "kind": "gpu", "test": test, "storage": mode, "grid": grid, "bytes_moved": moved,
        "median_gbs": gbs[gbs.count / 2], "min_gbs": gbs.last!, "max_gbs": gbs.first!,
        "times_s": t,
    ]
    let data = try! JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys])
    print(String(data: data, encoding: .utf8)!)
}

func makeBuffer(_ dev: MTLDevice, _ q: MTLCommandQueue, bytes: Int, shared: Bool) -> MTLBuffer {
    if shared {
        guard let b = dev.makeBuffer(length: bytes, options: .storageModeShared) else { fail("alloc shared") }
        memset(b.contents(), 1, bytes)  // touch every page before timing
        return b
    }
    guard let b = dev.makeBuffer(length: bytes, options: .storageModePrivate),
        let cb = q.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder()
    else { fail("alloc private") }
    blit.fill(buffer: b, range: 0..<bytes, value: 1)
    blit.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return b
}

func main() {
    guard let dev = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
    var args = Array(CommandLine.arguments.dropFirst())
    if args.first == "--info" {
        printInfo(dev)
        return
    }
    printInfo(dev)
    let mib = args.isEmpty ? 1024 : Int(args.removeFirst())!
    let passes = args.isEmpty ? 10 : Int(args.removeFirst())!
    let warmup = 2
    let bytes = mib << 20
    guard let queue = dev.makeCommandQueue() else { fail("no queue") }
    let lib: MTLLibrary
    do { lib = try dev.makeLibrary(source: shaderSource, options: nil) } catch { fail("shader: \(error)") }
    var pipes: [String: MTLComputePipelineState] = [:]
    for k in ["rd", "wr", "cp"] { pipes[k] = try! dev.makeComputePipelineState(function: lib.makeFunction(name: k)!) }
    let bench = Bench(dev: dev, queue: queue, pipes: pipes, bytes: bytes)
    guard let out = dev.makeBuffer(length: 16, options: .storageModeShared) else { fail("alloc") }
    // Grids: 24 cores x 4 or x 16 threadgroups of 256; each thread strides through the buffer
    let grids = [24 * 4 * 256, 24 * 16 * 256]
    for shared in [true, false] {
        let mode = shared ? "shared" : "private"
        let a = makeBuffer(dev, queue, bytes: bytes, shared: shared)
        let b = makeBuffer(dev, queue, bytes: bytes, shared: shared)
        for g in grids {
            let n = passes + warmup
            report("read", mode, grid: g, moved: bytes,
                   bench.run(passes: n) { bench.dispatch($0, "rd", [a, out], grid: g) }, warmup: warmup)
            report("write", mode, grid: g, moved: bytes,
                   bench.run(passes: n) { bench.dispatch($0, "wr", [a], grid: g) }, warmup: warmup)
            report("copy", mode, grid: g, moved: 2 * bytes,
                   bench.run(passes: n) { bench.dispatch($0, "cp", [a, b], grid: g) }, warmup: warmup)
        }
        let blit = bench.run(passes: passes + warmup) { cb in
            let enc = cb.makeBlitCommandEncoder()!
            enc.copy(from: a, sourceOffset: 0, to: b, destinationOffset: 0, size: bytes)
            enc.endEncoding()
        }
        report("blit_copy", mode, grid: 0, moved: 2 * bytes, blit, warmup: warmup)
    }
}

main()
