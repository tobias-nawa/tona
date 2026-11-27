#include "update/release.h"

#include "update/state.h"

namespace tona::update {

VerifyOutcome verify_release(const UpdateEnv& env, std::string manifest_text, std::string signature_text,
                             uint64_t last_seq) {
    VerifyOutcome out;
    if (manifest_text.size() > kMaxManifestBytes) {
        out.error = "manifest is larger than 1 MiB";
        return out;
    }
    const SignatureCheck sig = verify_minisig(manifest_text, signature_text, env.keys);
    if (!sig.ok) {
        out.error = "signature check failed: " + sig.error;
        return out;
    }
    const ManifestContext ctx{env.platform, env.clock(), last_seq};
    ManifestResult parsed = parse_manifest(manifest_text, sig.trusted_comment, ctx);
    if (!parsed.ok) {
        out.error = "manifest rejected: " + parsed.error;
        return out;
    }
    out.ok = true;
    out.release = Release{std::move(parsed.manifest), std::move(manifest_text), std::move(signature_text)};
    return out;
}

CheckOutcome check_for_update(const UpdateEnv& env) {
    CheckOutcome out;
    const StateResult st = load_state(env.layout);
    if (!st.ok) {
        out.error = st.error;
        return out;
    }
    std::string sig, manifest;
    // Signature first: it is tiny and decides whether the larger manifest is worth reading at all
    const FetchResult sig_r = fetch_text(env.fetch, {env.base.signature_url(), kMaxSignatureBytes, 0, 30}, sig);
    if (!sig_r.ok) {
        out.error = "signature download failed: " + sig_r.error;
        return out;
    }
    const FetchResult man_r = fetch_text(env.fetch, {env.base.manifest_url(), kMaxManifestBytes, 0, 60}, manifest);
    if (!man_r.ok) {
        out.error = "manifest download failed: " + man_r.error;
        return out;
    }
    VerifyOutcome v = verify_release(env, std::move(manifest), std::move(sig), st.state.seq);
    if (!v.ok) {
        out.error = v.error;
        return out;
    }
    out.kind = v.release.manifest.seq == st.state.seq ? CheckOutcome::Kind::UpToDate : CheckOutcome::Kind::Available;
    out.release = std::move(v.release);
    return out;
}

}  // namespace tona::update
