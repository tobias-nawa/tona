// Cost of dependent Metal dispatches: in one encoder, one encoder each, an indirect command buffer, command buffer
// round trips and pipelined command buffers. Kernel: thread 0 of one threadgroup increments a counter, so every
// dispatch depends on the previous one; the final counter checks the chain.
// Plan and evaluation: research/measurements/2026-11-11-mac-dispatch.md
// Build: swiftc -O bench/mac/dispatch.swift -o work/build/mac/dispatch
// Usage: dispatch [N dispatches, default 1000] [samples, default 20]
import Foundation
import Metal

let src = """
#include <metal_stdlib>
using namespace metal;
kernel void step(device uint* c [[buffer(0)]], uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0) c[0] = c[0] + 1;
}
"""

func fail(_ m: String) -> Never {
    FileHandle.standardError.write((m + "\n").data(using: .utf8)!)
    exit(1)
}

func emit(_ block: String, _ unit: String, _ samples: [Double], extra: [String: Any] = [:]) {
    let s = samples.sorted()
    var rec: [String: Any] = [
        "kind": "dispatch", "block": block, "unit": unit, "n_samples": s.count,
        "median_us": s[s.count / 2], "p99_us": s[min(s.count - 1, s.count * 99 / 100)], "min_us": s.first!,
        "max_us": s.last!,
    ]
    rec.merge(extra) { a, _ in a }
    let d = try! JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys])
    print(String(data: d, encoding: .utf8)!)
}

let dev = MTLCreateSystemDefaultDevice() ?? { fail("no device") }()
let args = CommandLine.arguments
let n = args.count > 1 ? Int(args[1])! : 1000
let samples = args.count > 2 ? Int(args[2])! : 20
let queue = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let fn = lib.makeFunction(name: "step")!
let pdesc = MTLComputePipelineDescriptor()
pdesc.computeFunction = fn
pdesc.supportIndirectCommandBuffers = true
let pipe = try! dev.makeComputePipelineState(descriptor: pdesc, options: [], reflection: nil)
let counter = dev.makeBuffer(length: 16, options: .storageModeShared)!
let one = MTLSize(width: 1, height: 1, depth: 1)
let tg = MTLSize(width: 32, height: 1, depth: 1)

func resetCounter() { counter.contents().storeBytes(of: UInt32(0), as: UInt32.self) }
func checkCounter(_ expect: Int) {
    let v = counter.contents().load(as: UInt32.self)
    if Int(v) != expect { fail("chain broken: counter \(v), expected \(expect)") }
}

// GPU time of one command buffer in µs
func gpuTime(_ encode: (MTLCommandBuffer) -> Void) -> Double {
    let cb = queue.makeCommandBuffer()!
    encode(cb)
    cb.commit()
    cb.waitUntilCompleted()
    if let e = cb.error { fail("\(e)") }
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e6
}

func blockD1() -> [Double] {
    (0..<samples).map { _ in
        resetCounter()
        let t = gpuTime { cb in
            let enc = cb.makeComputeCommandEncoder(dispatchType: .serial)!
            enc.setComputePipelineState(pipe)
            enc.setBuffer(counter, offset: 0, index: 0)
            for _ in 0..<n { enc.dispatchThreadgroups(one, threadsPerThreadgroup: tg) }
            enc.endEncoding()
        }
        checkCounter(n)
        return t / Double(n)
    }
}

func blockD2() -> [Double] {
    (0..<samples).map { _ in
        resetCounter()
        let t = gpuTime { cb in
            for _ in 0..<n {
                let enc = cb.makeComputeCommandEncoder(dispatchType: .serial)!
                enc.setComputePipelineState(pipe)
                enc.setBuffer(counter, offset: 0, index: 0)
                enc.dispatchThreadgroups(one, threadsPerThreadgroup: tg)
                enc.endEncoding()
            }
        }
        checkCounter(n)
        return t / Double(n)
    }
}

func makeICB() -> MTLIndirectCommandBuffer {
    let d = MTLIndirectCommandBufferDescriptor()
    d.commandTypes = .concurrentDispatch
    d.inheritPipelineState = false
    d.inheritBuffers = false
    d.maxKernelBufferBindCount = 1
    let icb = dev.makeIndirectCommandBuffer(descriptor: d, maxCommandCount: n, options: [])!
    for i in 0..<n {
        let c = icb.indirectComputeCommandAt(i)
        c.setComputePipelineState(pipe)
        c.setKernelBuffer(counter, offset: 0, at: 0)
        c.concurrentDispatchThreadgroups(one, threadsPerThreadgroup: tg)
        c.setBarrier()  // keeps the commands dependent, as the chain needs
    }
    return icb
}

func blockD3(_ icb: MTLIndirectCommandBuffer) -> [Double] {
    (0..<samples).map { _ in
        resetCounter()
        let t = gpuTime { cb in
            let enc = cb.makeComputeCommandEncoder(dispatchType: .serial)!
            enc.useResource(counter, usage: [.read, .write])
            enc.executeCommandsInBuffer(icb, range: 0..<n)
            enc.endEncoding()
        }
        checkCounter(n)
        return t / Double(n)
    }
}

func oneDispatchCB() -> MTLCommandBuffer {
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder(dispatchType: .serial)!
    enc.setComputePipelineState(pipe)
    enc.setBuffer(counter, offset: 0, index: 0)
    enc.dispatchThreadgroups(one, threadsPerThreadgroup: tg)
    enc.endEncoding()
    return cb
}

func blockD4() -> [Double] {
    resetCounter()
    let out: [Double] = (0..<500).map { _ in
        let t0 = DispatchTime.now().uptimeNanoseconds
        let cb = oneDispatchCB()
        cb.commit()
        cb.waitUntilCompleted()
        return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e3
    }
    checkCounter(500)
    return out
}

func blockD5() -> [Double] {
    (0..<samples).map { _ in
        resetCounter()
        let t0 = DispatchTime.now().uptimeNanoseconds
        var last: MTLCommandBuffer?
        for _ in 0..<n {
            let cb = oneDispatchCB()
            cb.commit()
            last = cb
        }
        last!.waitUntilCompleted()
        checkCounter(n)
        return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e3 / Double(n)
    }
}

_ = blockD1()  // warm-up: pipeline, driver, clocks
emit("D1", "us per dispatch", blockD1(), extra: ["n": n])
emit("D2", "us per dispatch", blockD2(), extra: ["n": n])
emit("D3", "us per dispatch", blockD3(makeICB()), extra: ["n": n])
emit("D4", "us per round trip", blockD4())
emit("D5", "us per command buffer", blockD5(), extra: ["n": n])
