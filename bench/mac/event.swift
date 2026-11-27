// GPU waits for a CPU signal inside a command buffer (MTLSharedEvent): latency, work before the wait, another queue
// during the wait, long waits (timeout check).
// Plan and evaluation: research/measurements/2026-11-11-mac-event.md
// Build: swiftc -O bench/mac/event.swift -o work/build/mac/event
// Usage: event [e1|e2a|e2b|e3]...   (default: e1 e2a e2b)
import Foundation
import Metal

let src = """
#include <metal_stdlib>
using namespace metal;
kernel void put(device atomic_uint* c [[buffer(0)]], constant uint& v [[buffer(1)]],
                uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0) atomic_store_explicit(c, v, memory_order_relaxed);
}
kernel void rd(device const uint4* a [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& n [[buffer(2)]],
               uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
    uint4 s = 0;
    for (uint i = gid; i < n; i += gsz) s += a[i];
    if ((s.x ^ s.y ^ s.z ^ s.w) == 0x9e3779b9u) out[0] = s.x;
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

func stats(_ v: [Double]) -> [String: Double] {
    let s = v.sorted()
    return ["median": s[s.count / 2], "p99": s[min(s.count - 1, s.count * 99 / 100)], "min": s.first!, "max": s.last!]
}

let dev = MTLCreateSystemDefaultDevice() ?? { fail("no device") }()
let queue = dev.makeCommandQueue()!
let queueB = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let put = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "put")!)
let rd = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "rd")!)
let flag = dev.makeBuffer(length: 16, options: .storageModeShared)!
let loadBytes = 1 << 30
let loadBuf = dev.makeBuffer(length: loadBytes, options: .storageModeShared)!
memset(loadBuf.contents(), 1, loadBytes)
let sink = dev.makeBuffer(length: 16, options: .storageModeShared)!

func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
func readFlag() -> UInt32 { flag.contents().load(as: UInt32.self) }

func encodePut(_ cb: MTLCommandBuffer, _ v: UInt32) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(put)
    enc.setBuffer(flag, offset: 0, index: 0)
    var x = v
    enc.setBytes(&x, length: 4, index: 1)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
}

// 10 reads of the 1 GiB buffer (~32 ms at 334 GB/s)
func encodeLoad(_ cb: MTLCommandBuffer) {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(rd)
    enc.setBuffer(loadBuf, offset: 0, index: 0)
    enc.setBuffer(sink, offset: 0, index: 1)
    var n = UInt32(loadBytes / 16)
    enc.setBytes(&n, length: 4, index: 2)
    for _ in 0..<10 {
        enc.dispatchThreads(MTLSize(width: 24 * 4 * 256, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    enc.endEncoding()
}

func gpuMs(_ cb: MTLCommandBuffer) -> Double { (cb.gpuEndTime - cb.gpuStartTime) * 1e3 }

func e1() {
    let ev = dev.makeSharedEvent()!
    flag.contents().storeBytes(of: UInt32(0), as: UInt32.self)
    let steps = 200
    let cb = queue.makeCommandBuffer()!
    for i in 1...steps {
        cb.encodeWaitForEvent(ev, value: UInt64(i))
        encodePut(cb, UInt32(i))
    }
    cb.commit()
    var lat: [Double] = []
    for i in 1...steps {
        let t0 = now()
        ev.signaledValue = UInt64(i)
        while readFlag() != UInt32(i) {}
        lat.append(Double(now() - t0) / 1e3)
    }
    cb.waitUntilCompleted()
    if let e = cb.error { fail("e1: \(e)") }
    emit(["kind": "event", "block": "E1", "unit": "us signal to visible write", "steps": steps,
          "stats": stats(Array(lat.dropFirst()))])
}

func e2a() {
    var alone: [Double] = [], withWait: [Double] = []
    for _ in 0..<5 {
        let cb0 = queue.makeCommandBuffer()!
        encodeLoad(cb0)
        encodePut(cb0, 1)
        cb0.commit()
        cb0.waitUntilCompleted()
        alone.append(gpuMs(cb0))
        let ev = dev.makeSharedEvent()!
        let cb = queue.makeCommandBuffer()!
        encodeLoad(cb)
        cb.encodeWaitForEvent(ev, value: 1)
        encodePut(cb, 2)
        cb.commit()
        usleep(5000)  // signal while the load still runs
        ev.signaledValue = 1
        cb.waitUntilCompleted()
        if let e = cb.error { fail("e2a: \(e)") }
        withWait.append(gpuMs(cb))
    }
    emit(["kind": "event", "block": "E2a", "unit": "ms GPU time", "load_alone": stats(alone),
          "load_wait_put": stats(withWait)])
}

func e2b() {
    var alone: [Double] = [], during: [Double] = [], waitA: [Double] = []
    for _ in 0..<5 {
        let b0 = queueB.makeCommandBuffer()!
        encodeLoad(b0)
        b0.commit()
        b0.waitUntilCompleted()
        alone.append(gpuMs(b0))
        let ev = dev.makeSharedEvent()!
        let a = queue.makeCommandBuffer()!
        a.encodeWaitForEvent(ev, value: 1)
        encodePut(a, 3)
        a.commit()
        let b = queueB.makeCommandBuffer()!
        encodeLoad(b)
        b.commit()
        usleep(40000)  // A waits longer than B's load runs
        ev.signaledValue = 1
        b.waitUntilCompleted()
        a.waitUntilCompleted()
        if let e = a.error ?? b.error { fail("e2b: \(e)") }
        during.append(gpuMs(b))
        waitA.append(gpuMs(a))
    }
    emit(["kind": "event", "block": "E2b", "unit": "ms GPU time", "b_alone": stats(alone), "b_during_wait": stats(during),
          "a_wait_cb": stats(waitA)])
}

func e3() {
    for delay in [0.5, 2.0, 8.0] {
        let ev = dev.makeSharedEvent()!
        let cb = queue.makeCommandBuffer()!
        cb.encodeWaitForEvent(ev, value: 1)
        encodePut(cb, 4)
        let t0 = now()
        cb.commit()
        usleep(UInt32(delay * 1e6))
        ev.signaledValue = 1
        cb.waitUntilCompleted()
        let wall = Double(now() - t0) / 1e9
        emit(["kind": "event", "block": "E3", "delay_s": delay, "wall_s": wall, "status": cb.status.rawValue,
              "error": cb.error.map { "\($0)" } ?? "", "gpu_ms": gpuMs(cb)])
    }
}

let blocks = CommandLine.arguments.count > 1 ? Array(CommandLine.arguments.dropFirst()) : ["e1", "e2a", "e2b"]
e2a()  // warm-up: pipelines, clocks
for b in blocks {
    switch b {
    case "e1": e1()
    case "e2a": e2a()
    case "e2b": e2b()
    case "e3": e3()
    default: fail("unknown block \(b)")
    }
}
