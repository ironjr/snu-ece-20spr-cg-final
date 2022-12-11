#ifndef PLAYER_CONTROLLER_H
#define PLAYER_CONTROLLER_H

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "base/camera.h"
#include "base/engine_global.h"
#include "base/entity.h"
#include "base/interface.h"


namespace engine
{
constexpr float DEFAULT_PLAYER_SPEED               =   2.5f;
constexpr float DEFAULT_PLAYER_SPRINT_MULTIPLIER   =   3.0f;

constexpr float DEFAULT_PLAYER_PITCH_SENSITIVITY   =   0.07f;
constexpr float DEFAULT_PLAYER_YAW_SENSITIVITY     =   0.1f;
constexpr float DEFAULT_PLAYER_ROLL_SENSITIVITY    =   0.1f;


class PlayerController : public Entity
{
public:
	Camera* playerCam;
    float movementSpeed;
    float sprintMultiplier;
    glm::vec3 rotationSensitivity;


    // Constructors and a distructor.
	PlayerController(const std::string& name, Entity* parent = nullptr)
        : Entity(name, parent), playerCam(new Camera()),
        movementSpeed(DEFAULT_PLAYER_SPEED),
        sprintMultiplier(DEFAULT_PLAYER_SPRINT_MULTIPLIER),
        rotationSensitivity(glm::vec3(
            DEFAULT_PLAYER_PITCH_SENSITIVITY,
            DEFAULT_PLAYER_YAW_SENSITIVITY,
            DEFAULT_PLAYER_ROLL_SENSITIVITY
        )) {}

	PlayerController(const std::string& name, const Transform& tf, Entity* parent = nullptr)
        : Entity(name, parent, tf), playerCam(new Camera()),
        movementSpeed(DEFAULT_PLAYER_SPEED),
        sprintMultiplier(DEFAULT_PLAYER_SPRINT_MULTIPLIER),
        rotationSensitivity(glm::vec3(
            DEFAULT_PLAYER_PITCH_SENSITIVITY,
            DEFAULT_PLAYER_YAW_SENSITIVITY,
            DEFAULT_PLAYER_ROLL_SENSITIVITY
        ))
    {
        this->playerCam->setTransform(this->tf);
    }

	~PlayerController()
	{
        delete this->playerCam;
    }

    // Processes input received from any keyboard-like input system. Accepts
    // input parameter in the form of camera defined ENUM (to abstract it from
    // windowing systems).
	void processCommands(const Commands* cmd)
	{
		float deltaTime = timer.getDeltaTime();
        float speed = this->movementSpeed;
        if (cmd->sprint)
        {
            speed *= this->sprintMultiplier;
        }

        glm::vec3 cameraFront = this->playerCam->cameraFront;
        glm::vec3 cameraUp = this->playerCam->cameraUp;

        // Forward and backward movement.
        this->tf.translate(
            cmd->moveForward * deltaTime * speed * cameraFront
        );
        // Left and right movement.
        this->tf.translate(
            cmd->moveRight * deltaTime * speed * glm::normalize(glm::cross(
                cameraFront, cameraUp
            ))
        );
        // Up and down movement.
        this->tf.translate(
            cmd->moveUp * deltaTime * speed * glm::vec3(0.0f, 1.0f, 0.0f)
        );

        // Set camera position.
        this->playerCam->tf.position = this->tf.position;
	}

    // Processes input received from a mouse input system. Expects the offset
    // value in both the x and y direction.
    void processMouseMovement(float deltaPitch, float deltaYaw)
    {
        // Offsets are based on the screen coordinates
        deltaPitch *= this->rotationSensitivity.x;
        deltaYaw *= this->rotationSensitivity.y;
        this->playerCam->updateRotation(deltaPitch, deltaYaw);
    }
};
}

#endif
