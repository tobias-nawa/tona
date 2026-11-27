// How long may a command buffer wait on an MTLSharedEvent before macOS aborts it, and does the GPU accept work after
// the abort? Waits rise step by step and the tool stops at the first abort (at most one provoked timeout per run).
// Plan and evaluation: research/measurements/2026-11-11-mac-timeout.md
// Build: swiftc -O bench/mac/timeout.swift -o work/build/mac/timeout
// Usage: timeout [delays in s, default 2.5 3 3.5 4 4.5 5 6 7]
import Foundation
import Metal

let src = """
#include <metal_stdlib>
using namespace metal;
kernel void put(device atomic_uint* c [[buffer(0)]], constant uint& v [[buffer(1)]],
                uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0) atomic_store_explicit(c, v, memory_order_relaxed);
}
"""

func fail(_ m: String) -> Never {
    FileHandle.standardError.write((m + "\n").data(using: .utf8)!)
    exit(1)
}

func emit(_ rec: [String: Any]) {
    let d = try! JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys])
    print(String(data: d, encoding: .utf8)!)
    fflush(stdout)
}

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

let dev = MTLCreateSystemDefaultDevice() ?? { fail("no device") }()
let queue = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let put = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "put")!)
let flag = dev.makeBuffer(length: 16, options: .storageModeShared)!

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

func errInfo(_ cb: MTLCommandBuffer) -> [String: Any] {
    guard let e = cb.error as NSError? else { return ["status": cb.status.rawValue, "error": ""] }
    return ["status": cb.status.rawValue, "error": e.localizedDescription, "domain": e.domain, "code": e.code,
            "access_revoked": e.domain == MTLCommandBufferErrorDomain
                && e.code == MTLCommandBufferError.accessRevoked.rawValue]
}

// one small command buffer on queue q; did it complete and write v?
func probe(_ q: MTLCommandQueue, _ v: UInt32, _ label: String) {
    flag.contents().storeBytes(of: UInt32(0), as: UInt32.self)
    let cb = q.makeCommandBuffer()!
    encodePut(cb, v)
    let t0 = now()
    cb.commit()
    cb.waitUntilCompleted()
    var rec = errInfo(cb)
    rec.merge(["kind": "timeout", "block": "probe", "queue": label, "wall_s": now() - t0,
               "wrote": flag.contents().load(as: UInt32.self) == v]) { a, _ in a }
    emit(rec)
}

let delays = CommandLine.arguments.count > 1 ? CommandLine.arguments.dropFirst().map { Double($0)! }
    : [2.5, 3, 3.5, 4, 4.5, 5, 6, 7]
probe(queue, 1, "same, before")
for delay in delays {
    let ev = dev.makeSharedEvent()!
    let cb = queue.makeCommandBuffer()!
    cb.encodeWaitForEvent(ev, value: 1)
    encodePut(cb, 2)
    var errAt = -1.0
    let t0 = now()
    cb.addCompletedHandler { _ in errAt = now() - t0 }
    cb.commit()
    usleep(UInt32(delay * 1e6))
    ev.signaledValue = 1
    cb.waitUntilCompleted()
    usleep(10000)  // let the completed handler run
    var rec = errInfo(cb)
    rec.merge(["kind": "timeout", "block": "wait", "delay_s": delay, "completed_after_s": errAt]) { a, _ in a }
    emit(rec)
    if cb.error != nil {
        probe(queue, 3, "same, after abort")
        probe(dev.makeCommandQueue()!, 4, "new, after abort")
        break
    }
}
