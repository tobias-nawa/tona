// The public keys the updater trusts. Keys are selected by the id inside a signature, never tried one by one.
#pragma once

#include <string_view>
#include <vector>

#include "update/minisig.h"

namespace tona::update {

// Base64 line of `minisign.pub` (starts with "RW"). Empty = not set yet: such a slot trusts nothing, so a build
// without the real keys accepts no release. Main key and spare key are generated and stored by the maintainer.
inline constexpr std::string_view kMainPublicKey = "RWQ3F7LonhcjgFwfaNP4/piW3DMh7vimCqxM3o9e8aedIDSX3GE6AY3n";
inline constexpr std::string_view kSparePublicKey = "RWR9M1PmEZ/s3UbiFyX8vcF/WRe7jhKwvFNJEptHcQFHBSeSEe9bQwPJ";

// Keys of the release build. Test keys never appear here; tests pass their own list.
std::vector<TrustedKey> release_keys();

}  // namespace tona::update
