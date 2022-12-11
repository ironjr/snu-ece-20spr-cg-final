#ifndef SYSTEM_GLOBAL_H
#define SYSTEM_GLOBAL_H

#include <vector>

#include "base/screen.h"
#include "base/timer.h"
#include "base/graphics_settings.h"

// #include "base/behavior.h"


// Systematic global variables that is required by other engine global variables
engine::ScreenInfo screen;
engine::Timer timer;
engine::GraphicsSettings graphicsSettings;

// TODO list of delegates to be called at startup or at each iteration.
// std::vector<engine::Behavior*> behaviorQueue = {};

#endif
