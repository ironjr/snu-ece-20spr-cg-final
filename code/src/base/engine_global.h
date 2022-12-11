#ifndef ENGINE_GLOBAL_H
#define ENGINE_GLOBAL_H

#include "base/game_manager.h"
#include "base/interface.h"


// Only one instance per class
// TODO make these thread safe
engine::Commands cmd;
engine::GameManager gameManager;

#endif
