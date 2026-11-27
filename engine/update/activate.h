// Switching between installed versions: the `current` and `previous` links and the clean-up of old versions
#pragma once

#include <string>
#include <string_view>

#include "update/layout.h"

namespace tona::update {

constexpr int kKeepVersions = 3;

// Version name a link points to ("" if the link is missing or points elsewhere)
std::string linked_version(const Layout& layout, const std::string& link_name);

// previous := the old current, current := versions/<version>. Both switches are atomic renames.
void activate_version(const Layout& layout, const std::string& version);

// current <-> previous. Returns false if there is no previous version.
bool swap_current_previous(const Layout& layout);

// Deletes versions beyond the newest kKeepVersions (never current or previous) and runtime directories that no
// remaining version uses
void prune(const Layout& layout);

// Numeric order of "a.b.c" version names; false if either is malformed
bool version_less(std::string_view a, std::string_view b);

}  // namespace tona::update
