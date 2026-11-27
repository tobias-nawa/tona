// File-system primitives for the installer. Everything is relative to a directory descriptor and refuses symlinks
// (O_NOFOLLOW), so nothing a download or a stale directory contains can redirect a write.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include "update/layout.h"

namespace tona::update {

struct FsError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Owns a directory file descriptor
class Dir {
public:
    Dir() = default;
    explicit Dir(int fd) : fd_(fd) {}
    ~Dir();
    Dir(Dir&& o) noexcept : fd_(o.release()) {}
    Dir& operator=(Dir&& o) noexcept;
    Dir(const Dir&) = delete;
    Dir& operator=(const Dir&) = delete;
    int fd() const { return fd_; }
    int release();

    // Opens an existing directory by path; a symlink as last component is refused
    static Dir open(const std::string& path);
    // Creates the directory `name` (mode 0700, fails if it exists) and opens it
    Dir make_subdir(const std::string& name) const;
    Dir open_subdir(const std::string& name) const;

private:
    int fd_ = -1;
};

// Writes `name` in `dir` (O_EXCL, refuses an existing file or symlink): created 0600, finished with fsync and the
// final mode. Call write() until done, then finish().
class FileWriter {
public:
    FileWriter(const Dir& dir, const std::string& name);
    ~FileWriter();
    FileWriter(const FileWriter&) = delete;
    FileWriter& operator=(const FileWriter&) = delete;
    bool write(const char* data, size_t n) const;  // false on a write error
    void finish(int mode);

private:
    int fd_ = -1;
};

// Re-reads `name` from disk: it must be a regular file of exactly `size` bytes with this SHA-256 (hex).
// Returns an error text, empty if everything matches.
std::string verify_file(const Dir& dir, const std::string& name, uint64_t size, const std::string& sha256_hex);

// Atomically points `link_path` at `target`: symlink under a temporary name, then rename over the old one
void set_symlink_atomic(const std::string& target, const std::string& link_path);

// Content of the link, empty if it does not exist or is no symlink
std::string read_link(const std::string& link_path);

void fsync_dir(const Dir& dir);

// Atomic rename that fails if the target exists (renameat2 RENAME_NOREPLACE)
void rename_noreplace(const std::string& from, const std::string& to);

// Creates the install root (and versions/, runtime/, staging/) with mode 0700, or checks an existing one: it must be
// a real directory owned by the caller and closed to others. Throws FsError otherwise.
void ensure_private_root(const Layout& layout);

}  // namespace tona::update
