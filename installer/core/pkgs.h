// Packages that building and running need: find them, check the version, name the command that installs the missing
// ones on the detected distribution. The installer itself only installs what needs no root (the toolchain to work/).
#pragma once

#include <string>
#include <vector>

#include "core/hw.h"

namespace tona::install {

struct Package {
    std::string id;       // program to look for (e.g. "cmake")
    std::string purpose;  // what it is needed for
    bool required = true;
    // package names per package manager; empty: not packaged there (see hint)
    std::string dnf{}, apt{}, pacman{}, brew{};  // {}: lets designated initializers omit them without warnings
    std::string hint{};                          // other source (e.g. an install script)
    // found on this machine
    bool found = false;
    std::string version{};
    std::string problem{};  // found but unusable (e.g. too old)
};

// the common packages plus those of the backend module, checked. A release install (`release`) builds nothing and
// downloads through the installer's own HTTPS code: of the common packages only the optional ones remain
std::vector<Package> check_packages(const Hardware& h, std::vector<Package> extra, bool release);
// package manager of the distribution: "dnf", "apt", "pacman", "brew" or empty
std::string package_manager(const Hardware& h);
// one command that installs all missing packages (empty: nothing missing or unknown distribution)
std::string install_command(const Hardware& h, const std::vector<Package>& pkgs);

}  // namespace tona::install
