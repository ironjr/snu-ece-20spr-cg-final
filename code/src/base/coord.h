#ifndef COORD_H
#define COORD_H

#include <glad/glad.h>
#include <glm/glm.hpp>


namespace engine
{
constexpr glm::vec3 WORLD3D_ORIGIN  = glm::vec3( 0.0f,  0.0f,  0.0f);
constexpr glm::vec3 WORLD3D_UP      = glm::vec3( 0.0f,  1.0f,  0.0f);
constexpr glm::vec3 WORLD3D_DOWN    = glm::vec3( 0.0f, -1.0f,  0.0f);
constexpr glm::vec3 WORLD3D_NORTH   = glm::vec3( 1.0f,  0.0f,  0.0f);
constexpr glm::vec3 WORLD3D_SOUTH   = glm::vec3(-1.0f,  0.0f,  0.0f);
constexpr glm::vec3 WORLD3D_EAST    = glm::vec3( 0.0f,  0.0f,  1.0f);
constexpr glm::vec3 WORLD3D_WEST    = glm::vec3( 0.0f,  0.0f, -1.0f);
}

#endif
