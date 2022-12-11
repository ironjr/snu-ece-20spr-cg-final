#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))  
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))  

#include <iostream>
#include <vector>


float getUniformRandomf()
{
    return (float)rand() / (float)RAND_MAX;
}

float getUniformRandomBetweenf(float a, float b)
{
    float r = getUniformRandomf();
    return a * (1 - r) + b * r;
}

glm::quat getRandomQuaternion()
{
    float angle = glm::mod(getUniformRandomBetweenf(0.0f, 360.0f), 360.0f);
    glm::vec3 axis = glm::normalize(glm::vec3(
        getUniformRandomf(),
        getUniformRandomf(),
        getUniformRandomf()
    ));
    return glm::angleAxis(glm::radians(angle), axis);
}

glm::mat4 getRandomRotation()
{
    return glm::mat4_cast(getRandomQuaternion());
}

float lerp(float a, float b, float r)
{
    return a * (1 - r) + b * r;
}

float clamp(float x, float xMin, float xMax)
{
    return x < xMin ? xMin : x > xMax ? xMax : x;
}
#endif
