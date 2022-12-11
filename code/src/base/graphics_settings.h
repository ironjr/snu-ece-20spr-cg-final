#ifndef GRAPHICS_SETTINGS_H
#define GRAPHICS_SETTINGS_H


namespace engine
{
struct GraphicsSettings
{
    // Graphics settings.
    bool useShadow = false;
    bool renderSun = false;
    bool renderGlow = false;
    int demoMode = 0;
};
}

#endif
