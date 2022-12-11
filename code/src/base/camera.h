#ifndef CAMERA_H
#define CAMERA_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>

#include "base/coord.h"
#include "base/entity.h"


namespace engine
{
enum class CameraMode
{
    PERSPECTIVE,
    ORTHOGONAL
};

// Default camera values
constexpr float DEFAULT_CAMERA_YAW                 = -90.0f;
constexpr float DEFAULT_CAMERA_PITCH               =   0.0f;
constexpr float DEFAULT_CAMERA_ROLL                =   0.0f;

constexpr float DEFAULT_CAMERA_NEAR                =   0.1f;
constexpr float DEFAULT_CAMERA_FAR                 = 100.0f;

constexpr float DEFAULT_CAMERA_ZOOM                =  90.0f; // (FOV)
constexpr float DEFAULT_CAMERA_MAX_ZOOM            = 170.0f;
constexpr float DEFAULT_CAMERA_MIN_ZOOM            =  10.0f;
constexpr float DEFAULT_CAMERA_ZOOM_SENSITIVITY    =   1.0f;


// The base camera class. Camera is defined as an entity.
class Camera : public Entity
{
public:
    // Camera attributes
    glm::vec3 cameraFront;
    glm::vec3 cameraUp;
    glm::vec3 cameraRight;

    // Camera options
    float zoom;
    float zoomSensitivity;
    CameraMode mode;


    // Constructor with vectors.
    Camera(
        glm::vec3 position = WORLD3D_ORIGIN,
        float yaw = DEFAULT_CAMERA_YAW,
        float pitch = DEFAULT_CAMERA_PITCH,
        float roll = DEFAULT_CAMERA_ROLL,
        CameraMode mode = CameraMode::PERSPECTIVE
    ) : Entity(),
        zoom(DEFAULT_CAMERA_ZOOM),
        zoomSensitivity(DEFAULT_CAMERA_ZOOM_SENSITIVITY), mode(mode)
    {
        this->tf.position = position;
        this->tf.rotation = glm::vec3(pitch, yaw, roll);
        updateCameraVectors();
    }

    // Constructor with scalar values.
    Camera(
        float posX, float posY, float posZ,
        float yaw = DEFAULT_CAMERA_YAW,
        float pitch = DEFAULT_CAMERA_PITCH,
        float roll = DEFAULT_CAMERA_ROLL,
        CameraMode mode = CameraMode::PERSPECTIVE
    ) : Entity(), cameraFront(WORLD3D_WEST),
        zoom(DEFAULT_CAMERA_ZOOM),
        zoomSensitivity(DEFAULT_CAMERA_ZOOM_SENSITIVITY), mode(mode)
    {
        this->tf.position = glm::vec3(posX, posY, posZ);
        this->tf.rotation = glm::vec3(pitch, yaw, roll);
        updateCameraVectors();
    }

    // Returns the view matrix calculated using Euler angles and the LookAt matrix.
    glm::mat4 getViewMatrix()
    {
        return glm::lookAt(
            this->tf.position,
            this->tf.position + this->cameraFront,
            this->cameraUp
        );
    }

    // Returns the view matrix calculated using Euler Angles and the LookAt Matrix
    glm::mat4 getViewMatrixVerticalFixed()
    {   
        glm::vec3 frontProjected = glm::normalize(glm::vec3(
            this->cameraFront.x, 0.0f, this->cameraFront.z
        ));
        return glm::lookAt(
            this->tf.position,
            this->tf.position + frontProjected,
            WORLD3D_UP
        );
    }

    // Returns the projection matrix calculated using field of view and screen size.
    glm::mat4 getProjectionMatrix(unsigned int scrWidth, unsigned int scrHeight)
    {
        switch (this->mode)
        {
        case CameraMode::PERSPECTIVE:
            return glm::perspective(
                glm::radians(this->zoom),
                (float)scrWidth / scrHeight,
                DEFAULT_CAMERA_NEAR, DEFAULT_CAMERA_FAR
            );
        case CameraMode::ORTHOGONAL:
            return glm::ortho(
                0.0f, (float)scrWidth,
                0.0f, (float)scrHeight,
                DEFAULT_CAMERA_NEAR, DEFAULT_CAMERA_FAR
            );
        default:
            throw std::invalid_argument("ERROR::CAMERA::WRONG_CAMERA_MODE");
        }
    }

    void setPosition(glm::vec3 position)
    {
        this->tf.position = position;
        this->updateCameraVectors();
    }

    void setRotation(glm::vec3 rotation)
    {
        this->tf.rotation = rotation;
        this->updateCameraVectors();
    }

    void setTransform(const Transform& tf)
    {
        this->tf.position = tf.position;
        this->tf.rotation = tf.rotation;
        this->updateCameraVectors();
    }

    void setTransform(glm::vec3 position, glm::vec3 rotation)
    {
        this->tf.position = position;
        this->tf.rotation = rotation;
        this->updateCameraVectors();
    }

    // Processes input received from a mouse input system. Expects the offset
    // value in both the x and y direction.
    void updateRotation(float deltaPitch, float deltaYaw, GLboolean constrainPitch = GL_TRUE)
    {
        float pitch = this->tf.rotation.x + deltaPitch;
        float yaw = this->tf.rotation.y + deltaYaw;

        if (constrainPitch == GL_TRUE)
        {
            // Bound pitch in [-89.0f, 89.0f] to prevent view from flipping.
            if (pitch > 89.0f)
                pitch = 89.0f;
            if (pitch < -89.0f)
                pitch = -89.0f;
        }

        // Bound yaw in (-180.0f, 180.0f].
        yaw = glm::mod(glm::mod(yaw, 360.0f) + 360.0f, 360.0f);
        if (yaw > 180.0f)
            yaw -= 360.0f;

        this->tf.rotation.x = pitch;
        this->tf.rotation.y = yaw;
        this->updateCameraVectors();
    }

    // Processes input received from a mouse scroll-wheel event. Only requires
    // input on the vertical wheel-axis.
    void updateFoV(float amount)
    {
        float zoom = this->zoom + amount * this->zoomSensitivity;
        if (zoom > DEFAULT_CAMERA_MAX_ZOOM)
        {
            zoom = DEFAULT_CAMERA_MAX_ZOOM;
        }
        if (zoom < DEFAULT_CAMERA_MIN_ZOOM)
        {
            zoom = DEFAULT_CAMERA_MIN_ZOOM;
        }
        this->zoom = zoom;
    }

private:
    // Calculates the front vector from the Camera's (updated) Euler Angles.
    void updateCameraVectors()
    {
        float pitch = this->tf.rotation.x;
        float yaw = this->tf.rotation.y;
        float roll = this->tf.rotation.z;

        float cosp = cos(glm::radians(pitch));
        float sinp = sin(glm::radians(pitch));
        float cosy = cos(glm::radians(yaw));
        float siny = sin(glm::radians(yaw));
        float cosr = cos(glm::radians(roll));
        float sinr = sin(glm::radians(roll));

        // Rotations are applied in the order of yaw -> pitch -> roll.
        this->cameraFront = glm::normalize(glm::vec3(
            cosy * cosp * cosr - sinp * sinr,
            cosy * cosp * sinr + sinp * cosr,
            siny * cosp
        ));
        this->cameraRight
            = glm::normalize(glm::cross(this->cameraFront, WORLD3D_UP));
        this->cameraUp
            = glm::cross(this->cameraRight, this->cameraFront);
    }
};
}

#endif
