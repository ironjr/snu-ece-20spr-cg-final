#ifndef ATTRIBUTE_H
#define ATTRIBUTE_H

#include "data/geometry.h"
#include "data/model.h"
#include "data/texture_cube.h"


namespace engine
{
class Entity;

class Attribute
{
public:
    Entity* entity = nullptr;

    Attribute() {}
    virtual ~Attribute() {}
};


class ModelAttribute : public Attribute
{
public:
    Model* model;

    ModelAttribute(Model* model) : Attribute(), model(model)
    {
        ++(model->assignments);
    }

    ~ModelAttribute()
    {
        --(this->model->assignments);
    }
};


class CubemapAttribute : public Attribute
{
public:
    Geometry* geometry;
    CubemapTexture* texture;

    CubemapAttribute(Geometry* geometry, CubemapTexture* texture)
        : Attribute(), geometry(geometry), texture(texture)
    {
        ++(geometry->assignments);
        ++(texture->assignments);
    }

    ~CubemapAttribute()
    {
        --(this->geometry->assignments);
        --(this->texture->assignments);
    }
};
}
#endif
