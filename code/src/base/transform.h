#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <glad/glad.h> 
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>


namespace engine
{
class Transform
{
public:
    glm::vec3 position;
    glm::vec3 rotation; // Euler angles (pitch, yaw, roll)
    glm::vec3 scale;


    // Constructors.
    Transform()
        : position(glm::vec3(0.0f)), rotation(glm::vec3(0.0f)), scale(glm::vec3(1.0f)) {}

    Transform(glm::vec3 position)
        : position(position), rotation(glm::vec3(0.0f)), scale(glm::vec3(1.0f)) {}

    Transform(glm::vec3 position, glm::vec3 rotation)
        : position(position), rotation(rotation), scale(glm::vec3(1.0f)) {}

    Transform(glm::vec3 position, glm::quat quaternion)
        : position(position), scale(glm::vec3(1.0f))
    {
        this->rotation = glm::degrees(glm::eulerAngles(quaternion));
    }

    Transform(glm::vec3 position, glm::vec3 rotation, float scale)
        : position(position), rotation(rotation), scale(glm::vec3(scale)) {}

    Transform(glm::vec3 position, glm::quat quaternion, float scale)
        : position(position), scale(glm::vec3(scale))
    {
        this->rotation = glm::degrees(glm::eulerAngles(quaternion));
    }

    Transform(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale)
        : position(position), rotation(rotation), scale(glm::vec3(scale)) {}

    Transform(glm::vec3 position, glm::quat quaternion, glm::vec3 scale)
        : position(position), scale(scale)
    {
        this->rotation = glm::degrees(glm::eulerAngles(quaternion));
    }

    // Copy constructor.
    Transform(const Transform& other)
        : position(other.position), rotation(other.rotation), scale(other.scale) {}

    // Copy assignment operator.
    Transform& operator=(const Transform& other)
    {
        if (&other == this)
            return *this;
        this->position = other.position;
        this->rotation = other.rotation;
        this->scale = other.scale;
        return *this;
    }

    // Getters.
    glm::mat4 getModelMatrix()
    {
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, this->position);
        model = glm::rotate(
            model,
            glm::radians(this->rotation.z),
            glm::vec3(0.0f, 0.0f, 1.0f)
        ); // Roll
        model = glm::rotate(
            model,
            glm::radians(this->rotation.x),
            glm::vec3(1.0f, 0.0f, 0.0f)
        ); // Pitch
        model = glm::rotate(
            model,
            glm::radians(this->rotation.y),
            glm::vec3(0.0f, 1.0f, 0.0f)
        ); // Yaw
        model = glm::scale(model, this->scale);
        return model;
    }

    // Interface for movements and transforms.
    void translate(glm::vec3 displacement)
    {
        this->position += displacement;
    }

    void rotate(float angle, glm::vec3 axis)
    {
        glm::quat quaternion = glm::quat(glm::radians(this->rotation));
        quaternion = glm::angleAxis(glm::radians(angle), axis) * quaternion;
        this->rotation = glm::eulerAngles(quaternion);
    }

    void rotate(glm::vec3 rotation)
    {
        glm::quat quaternion = glm::quat(glm::radians(this->rotation));
        quaternion = glm::quat(glm::radians(rotation)) * quaternion;
        this->rotation = glm::eulerAngles(quaternion);
    }

    void rotate(glm::quat quaternion)
    {
        quaternion = quaternion * glm::quat(glm::radians(this->rotation));
        this->rotation = glm::eulerAngles(quaternion);
    }

    void translateAndRotate(glm::vec3 displacement, float angle, glm::vec3 axis)
    {
        this->translate(displacement);
        this->rotate(angle, axis);
    }

    void translateAndRotate(glm::vec3 displacement, glm::vec3 rotation)
    {
        this->translate(displacement);
        this->rotate(rotation);
    }

    void translateAndRotate(glm::vec3 displacement, glm::quat quaternion)
    {
        this->translate(displacement);
        this->rotate(quaternion);
    }

    void rescale(glm::vec3 scale)
    {
        this->scale *= scale;
    }

private:
    glm::mat4 rotateEulerAngles(glm::vec3 euler)
    {
        glm::mat4 tf = glm::mat4(1.0f);
        tf = glm::rotate(
            tf, glm::radians(this->rotation.z), glm::vec3(0.0f, 0.0f, 1.0f)
        ); // Roll
        tf = glm::rotate(
            tf, glm::radians(this->rotation.x), glm::vec3(1.0f, 0.0f, 0.0f)
        ); // Pitch
        tf = glm::rotate(
            tf, glm::radians(this->rotation.y), glm::vec3(0.0f, 1.0f, 0.0f)
        ); // Yaw
        return tf;
    }

    glm::mat4 rotateEulerAngles(glm::mat4 tf, glm::vec3 euler)
    {
        tf = glm::rotate(
            tf, glm::radians(this->rotation.z), glm::vec3(0.0f, 0.0f, 1.0f)
        ); // Roll
        tf = glm::rotate(
            tf, glm::radians(this->rotation.x), glm::vec3(1.0f, 0.0f, 0.0f)
        ); // Pitch
        tf = glm::rotate(
            tf, glm::radians(this->rotation.y), glm::vec3(0.0f, 1.0f, 0.0f)
        ); // Yaw
        return tf;
    }
};
}
#endif
