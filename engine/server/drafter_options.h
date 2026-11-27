// Checks the drafter options of the command line against what the model declares (ModelBase::drafters). The server
// knows no architecture: a new model declares its own support and needs no change here.
#pragma once

#include <string>

#include "model/arch.h"

namespace tona::server {

// What the model takes, for the messages
inline std::string drafters_accepted(const DrafterSupport& sup) {
    if (sup.dflash && sup.mtp_file) return "--dflash and --mtp";
    if (sup.dflash) return "--dflash";
    return sup.mtp_file ? "--mtp" : "no drafter";
}

// Empty when the options fit, else the message for the user; model is its name for the message
inline std::string drafter_error(const DrafterSupport& sup, const std::string& model, bool have_dflash, bool have_mtp) {
    if (have_dflash && have_mtp) return "only one of --dflash and --mtp";
    if ((have_dflash && !sup.dflash) || (have_mtp && !sup.mtp_file))
        return "a " + model + " model does not take this drafter; it takes " + drafters_accepted(sup);
    if (sup.required && !have_dflash && !have_mtp)
        return "a " + model + " model needs one of " + drafters_accepted(sup);
    return "";
}

}  // namespace tona::server
