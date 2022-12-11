#ifndef GAME_MANAGER_H
#define GAME_MANAGER_H

#include <glad/glad.h> 
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <utility>

#include "base/system_global.h"

#include "base/camera.h"
#include "base/interface.h"
#include "base/player_controller.h"
#include "base/transform.h"

#include "data/light.h"

using namespace std::string_literals;


namespace engine
{
class GameManager
{
public:
	Camera* defaultCamera;
	PlayerController* playerController;

    DirectionalLight* sun = nullptr;


	GameManager()
	{
		this->playerController = new PlayerController(
            "Our Hero"s,
            Transform(
                glm::vec3(4.9f, 5.2f, 4.3f),
                glm::vec3(-30.0f, -135.0f, 0.0f),
                1.0f
            ),
            nullptr
        );
		this->defaultCamera = this->playerController->playerCam;
	}

	~GameManager()
	{
		delete this->playerController;
	}

	void processCommands(const Commands* cmd)
	{
		this->playerController->processCommands(cmd);
        if (this->sun)
        {
            this->sun->processKeyboard(cmd->azimuthEast, cmd->elevationUp);
        }
	}

    void processMouseMovement(float xoffset, float yoffset)
    {
        this->playerController->processMouseMovement(yoffset, xoffset);
    }

    void processMouseScroll(float xoffset, float yoffset)
    {
        this->defaultCamera->updateFoV(yoffset);
    }
};
}

#endif
