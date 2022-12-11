#ifndef TEXTURE_CUBE_H
#define TEXTURE_CUBE_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "base/asset.h"


namespace engine
{
class CubemapTexture : public Asset
{
public:
    unsigned int ID;
    int width;
    int height;
    int channels;


    CubemapTexture(const std::string& name, std::vector<std::string> faces) : Asset(name)
    {
        glGenTextures(1, &(this->ID));
        glBindTexture(GL_TEXTURE_CUBE_MAP, this->ID);

        // Set the texture filterinf parameters
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Set texture wrapping to GL_CLAMP_TO_EDGE
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        stbi_set_flip_vertically_on_load(false);

        for (unsigned int i = 0; i < faces.size(); i++)
        {
            unsigned char *data = stbi_load(
                faces[i].c_str(), &(this->width), &(this->height),
                &(this->channels), 0
            );
            if (data)
            {
                glTexImage2D(
                    GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, this->width,
                    this->height, 0, GL_RGB, GL_UNSIGNED_BYTE, data
                );
                stbi_image_free(data);
            }
            else
            {
                std::cout << "ERROR::TEXTURE::FAILED_TO_LOAD_TEXTURE: " << faces[i] << std::endl;
                stbi_image_free(data);
            }
        }
    }
};
}
#endif
