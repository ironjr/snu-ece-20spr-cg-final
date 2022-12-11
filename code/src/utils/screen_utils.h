#ifndef SCREEN_UTILS_H
#define SCREEN_UTILS_H

#include <string>

#include "FreeImage.h"

#include "base/system_global.h"


// Save Image to PNG file. Press V key to call.
// Argument:
//     filename: date.png (created in bin folder)
void saveImage(const std::string& filename)
{
    // Make the BYTE array, factor of 3 because it's RBG.
    BYTE* pixels = new BYTE[3 * screen.width * screen.height];
    glReadPixels(
        0, 0, screen.width, screen.height,
        GL_BGR, GL_UNSIGNED_BYTE,
        pixels
    );

    // Convert to FreeImage format & save to file
    FIBITMAP* image = FreeImage_ConvertFromRawBits(
        pixels,
        screen.width,
        screen.height,
        3 * screen.width,
        24,
        0xFF0000,
        0x00FF00,
        0x0000FF,
        false
    );
    FreeImage_Save(FIF_PNG, image, filename.c_str(), 0);

    // Free resources
    FreeImage_Unload(image);
    delete[] pixels;
}
#endif
