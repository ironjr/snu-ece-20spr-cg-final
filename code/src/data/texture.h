#ifndef TEXTURE_H
#define TEXTURE_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION   // use of stb functions once and for all
#include "ext/stb_image.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "base/asset.h"


namespace engine
{
class Texture : public Asset
{
public:
    unsigned int ID;
    int width;
    int height;
    int channels;


    Texture(const std::string& name, const std::string& filePath) : Asset(name)
    {   
        glGenTextures(1, &(this->ID));
        glBindTexture(GL_TEXTURE_2D, this->ID);

        // Set the texture wrapping parameters.
        // Set texture wrapping to GL_REPEAT (default wrapping method).
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        // Set texture filtering parameters.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_set_flip_vertically_on_load(true);

        unsigned char *data = stbi_load(
            filePath.c_str(), &(this->width), &(this->height), &(this->channels), 0
        );
        if (data)
        {   
            if (channels == 3)
                glTexImage2D(
                    GL_TEXTURE_2D, 0, GL_RGB, this->width, this->height, 0,
                    GL_RGB, GL_UNSIGNED_BYTE, data
                );
            else if (channels > 3)
                glTexImage2D(
                    GL_TEXTURE_2D, 0, GL_RGBA, this->width, this->height, 0,
                    GL_RGBA, GL_UNSIGNED_BYTE, data
                );
            else
                std::cout << "ERROR::TEXTURE::WRONG_CHANNEL_SIZE" << std::endl;
            
            glGenerateMipmap(GL_TEXTURE_2D);
        }
        else
        {
            std::cout << "ERROR::TEXTURE::FAILED_TO_LOAD_TEXTURE" << std::endl;
        }

        stbi_image_free(data);
    }
};


class DepthmapTexture : public Asset
{
public:
	unsigned int ID;
	unsigned int FBO;
	int width;
	int height;


	DepthmapTexture(const std::string& name, int shadow_width, int shadow_height)
        : Asset(name), width(shadow_width), height(shadow_height)
	{
		glGenTextures(1, &(this->ID));
		glBindTexture(GL_TEXTURE_2D, this->ID);
		glTexImage2D(
            GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, this->width, this->height, 0,
            GL_DEPTH_COMPONENT, GL_FLOAT, nullptr
        );

        // Set texture wrapping.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        // Set texture filtering parameters.
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

		// Attach depth texture as FBO's depth buffer.
		glGenFramebuffers(1, &(this->FBO));
		glBindFramebuffer(GL_FRAMEBUFFER, this->FBO);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, this->ID, 0);

		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

private:
};
}
#endif
