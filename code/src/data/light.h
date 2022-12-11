#pragma once
#ifndef LIGHT_H
#define LIGHT_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <string>
#include <vector>

#include "base/asset.h"


namespace engine
{
constexpr float DIRLIGHT_ELEVATION_MIN = 15.0f;
constexpr float DIRLIGHT_ELEVATION_MAX = 80.0f;
constexpr float DEFAULT_DIRLIGHT_SPEED_AZIMUTH = 1.0f; // degrees
constexpr float DEFAULT_DIRLIGHT_SPEED_ELEVATION = 0.07f; // degrees


enum class LightType
{
	DIRECTIONAL,
	POINT
};


class DirectionalLight : public Asset
{
public:
	float azimuth;
	float elevation;
	glm::vec3 lightDir; // Direction of light. If elevation is 90, it would be (0,-1,0)
	glm::vec3 lightColor; // This is I_d (I_s = I_d, I_a = 0.3 * I_d)


	DirectionalLight(
        const std::string& name, float azimuth, float elevation,
        glm::vec3 lightColor
    ) : Asset(name), azimuth(azimuth), elevation(elevation),
        lightColor(lightColor)
    {
		updateLightDir();
	}

	DirectionalLight(
        const std::string& name, glm::vec3 lightDir, glm::vec3 lightColor
    ) : Asset(name), lightDir(lightDir), lightColor(lightColor) {}

	glm::mat4 getViewMatrix(glm::vec3 cameraPosition) {
		// Directional light has no light position. Assume fake light position
        // depending on camera position.
		float lightDistance = 15.0f;
		glm::vec3 lightPos = cameraPosition - this->lightDir * lightDistance;
		return glm::lookAt(lightPos, cameraPosition, glm::vec3(0.0f, 1.0f, 0.0f));
	}

	glm::mat4 getProjectionMatrix()
    {
		// For simplicity, just use static projection matrix. (Actually we
        // have to be more accurate with considering camera's frustum)
		return glm::ortho(-15.0f, 15.0f, -15.0f, 15.0f, 0.1f, 50.0f);
	}

	void updateLightDir()
    {
        float phi = this->azimuth;
        float theta = 90.0f - this->elevation;

        float cosp = cos(glm::radians(phi));
        float sinp = sin(glm::radians(phi));
        float cost = cos(glm::radians(theta));
        float sint = sin(glm::radians(theta));

        this->lightDir = -glm::normalize(glm::vec3(
            sint * cosp,
            cost,
            sint * -sinp
        ));
	}

	// Processes input received from a keyboard input system. Expects the offset
    // value in both the x(azimuth) and y(elevation) direction.
	void processKeyboard(int azimuthEast, int elevationUp)
	{
        float azimuth = this->azimuth
            + azimuthEast * DEFAULT_DIRLIGHT_SPEED_AZIMUTH;
        float elevation = this->elevation
            + elevationUp * DEFAULT_DIRLIGHT_SPEED_ELEVATION;
        
        // Bound azimuth in (-180.0f, 180.0f].
        azimuth = glm::mod(glm::mod(azimuth, 360.0f) + 360.0f, 360.0f);
        if (azimuth > 180.0f)
            azimuth -= 360.0f;

		// Set elevation between 15 to 80 (degree)!
        if (elevation > DIRLIGHT_ELEVATION_MAX)
            elevation = DIRLIGHT_ELEVATION_MAX;
        if (elevation < DIRLIGHT_ELEVATION_MIN)
            elevation = DIRLIGHT_ELEVATION_MIN;

        this->azimuth = azimuth;
        this->elevation = elevation;
        updateLightDir();
	}
};
}
#endif
