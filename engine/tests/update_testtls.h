// A throw-away self-signed certificate for 127.0.0.1, made with the OpenSSL API at test time, so no private key
// has to be kept in the repository.
#pragma once

#include <openssl/pem.h>
#include <openssl/x509v3.h>

#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

namespace tona::test {

struct SelfSigned {
    std::string cert_path, key_path;
};

namespace detail {

struct PkeyDeleter {
    void operator()(EVP_PKEY* p) const { EVP_PKEY_free(p); }
};
struct X509Deleter {
    void operator()(X509* p) const { X509_free(p); }
};

inline void write_pem(const std::string& path, const std::function<int(FILE*)>& writer) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f || writer(f) != 1) throw std::runtime_error("cannot write " + path);
    std::fclose(f);
}

inline void add_san(X509* x, const std::string& ip) {
    X509V3_CTX ctx;
    X509V3_set_ctx_nodb(&ctx);
    X509V3_set_ctx(&ctx, x, x, nullptr, nullptr, 0);
    const std::string value = "IP:" + ip;
    X509_EXTENSION* ext = X509V3_EXT_conf_nid(nullptr, &ctx, NID_subject_alt_name, value.c_str());
    X509_add_ext(x, ext, -1);
    X509_EXTENSION_free(ext);
}

}  // namespace detail

// `san_ip` names the address the certificate is valid for; a different one makes host name checking fail
inline SelfSigned make_self_signed(const std::string& dir, const char* san_ip = "127.0.0.1") {
    std::unique_ptr<EVP_PKEY, detail::PkeyDeleter> key(EVP_PKEY_Q_keygen(nullptr, nullptr, "EC", "P-256"));
    std::unique_ptr<X509, detail::X509Deleter> x(X509_new());
    X509_set_version(x.get(), 2);
    ASN1_INTEGER_set(X509_get_serialNumber(x.get()), 1);
    X509_gmtime_adj(X509_getm_notBefore(x.get()), -60);
    X509_gmtime_adj(X509_getm_notAfter(x.get()), 3600);
    X509_set_pubkey(x.get(), key.get());
    // A name of our own: OpenSSL 4 returns the subject name as const, so it cannot be filled in place any more
    std::unique_ptr<X509_NAME, decltype(&X509_NAME_free)> name(X509_NAME_new(), &X509_NAME_free);
    const auto* common_name = reinterpret_cast<const uint8_t*>("tona-test");
    X509_NAME_add_entry_by_txt(name.get(), "CN", MBSTRING_ASC, common_name, -1, -1, 0);
    X509_set_subject_name(x.get(), name.get());
    X509_set_issuer_name(x.get(), name.get());
    detail::add_san(x.get(), san_ip);
    X509_sign(x.get(), key.get(), EVP_sha256());
    SelfSigned out{dir + "/cert.pem", dir + "/key.pem"};
    detail::write_pem(out.cert_path, [&](FILE* f) { return PEM_write_X509(f, x.get()); });
    detail::write_pem(out.key_path, [&](FILE* f) {
        return PEM_write_PrivateKey(f, key.get(), nullptr, nullptr, 0, nullptr, nullptr);
    });
    return out;
}

}  // namespace tona::test
