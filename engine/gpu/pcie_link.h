// The PCIe link of the GPU from sysfs, and the planner's price of an expert that the GPU reads from pinned RAM over it.
// Parsing and arithmetic are pure (unit-tested without a GPU); read_pcie_link takes the sysfs directory of the device.
#pragma once

#include <string>
#include <string_view>

namespace tona::gpu {

struct PcieLink {
    double gts = 0;  // transfer rate per lane in GT/s (32 = PCIe 5.0)
    int width = 0;   // lanes
    bool known() const { return gts > 0 && width > 0; }
};

// "32.0 GT/s PCIe" -> 32.0; 0 if the text is not a link speed.
double parse_link_speed(std::string_view text);

// The link of the PCI device in `device_dir` (/sys/bus/pci/devices/<id>): the slower of the device and its upstream
// port. Speed is the capability (max_link_speed), not current_link_speed: the link drops to a low speed when the GPU
// idles and trains up under load. Width is current_link_width, because a badly seated card or a shared slot trains
// fewer lanes than it could and stays there. Unknown (known() false) if sysfs does not give it.
PcieLink read_pcie_link(const std::string& device_dir);

// Planner cost (us) of one expert read directly from pinned RAM: reference_us on the reference link (PCIe 5.0 x16),
// scaled with the inverse of the link's raw rate (GT/s x lanes). Both generations 3 to 5 use 128b/130b coding, so the
// ratio of raw rates is the ratio of payload rates. reference_us if the link is unknown.
double gd_cost_us(double reference_us, const PcieLink& link);

}  // namespace tona::gpu
