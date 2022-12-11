#ifndef SCENE_H
#define SCENE_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <map>
#include <stdexcept>
#include <vector>

#include "base/entity.h"

#include "data/geometry.h"
#include "data/model.h"
#include "data/shader.h"
#include "data/texture.h"
#include "data/texture_cube.h"


namespace engine
{
class Scene
{
public:
    std::map<std::string, Entity*> entities;


    Scene()
    {
        // this->root = new Entity();
    }

    ~Scene()
    {
        // delete root;
        for (auto& elem : this->shaders)
        {
            delete elem.second;
        }
        for (auto& elem : this->textures)
        {
            delete elem.second;
        }
        for (auto& elem : this->cubemapTextures)
        {
            delete elem.second;
        }
        for (auto& elem : this->geometries)
        {
            delete elem.second;
        }
        for (auto& elem : this->models)
        {
            delete elem.second;
        }
        for (auto& elem : this->entities)
        {
            delete elem.second;
        }
    }

    // Insert and delete interface.
    void addShader(Shader* shader)
    {
        auto ptr = this->shaders.insert(std::make_pair(shader->name, shader));
        if (!ptr.second)
        {
            throw std::runtime_error("ERROR::SCENE::Key already exists in shaders");
        }
    }

    void addShaders(std::vector<Shader*> shaders)
    {
        for (auto& shader : shaders)
        {
            this->addShader(shader);
        }
    }

    void addTexture(Texture* texture)
    {
        auto ptr = this->textures.insert(std::make_pair(texture->name, texture));
        if (!ptr.second)
        {
            throw std::runtime_error("ERROR::SCENE::Key already exists in textures");
        }
    }

    void addTextures(std::vector<Texture*> textures)
    {
        for (auto& texture : textures)
        {
            this->addTexture(texture);
        }
    }

    void addCubemapTexture(CubemapTexture* cubemap)
    {
        auto ptr = this->cubemapTextures.insert(std::make_pair(cubemap->name, cubemap));
        if (!ptr.second)
        {
            throw std::runtime_error("ERROR::SCENE::Key already exists in cubemapTextures");
        }
    }

    void addCubemapTextures(std::vector<CubemapTexture*> cubemaps)
    {
        for (auto& cubemap : cubemaps)
        {
            this->addCubemapTexture(cubemap);
        }
    }

    void addGeometry(Geometry* geometry)
    {
        auto ptr = this->geometries.insert(std::make_pair(geometry->name, geometry));
        if (!ptr.second)
        {
            throw std::runtime_error("ERROR::SCENE::Key already exists in geometries");
        }
    }

    void addGeometries(std::vector<Geometry*> geometries)
    {
        for (auto& geometry : geometries)
        {
            this->addGeometry(geometry);
        }
    }

    void addModel(Model* model)
    {
        auto ptr = this->models.insert(std::make_pair(model->name, model));
        if (!ptr.second)
        {
            throw std::runtime_error("ERROR::SCENE::Key already exists in models");
        }
    }

    void addModels(std::vector<Model*> models)
    {
        for (auto& model : models)
        {
            this->addModel(model);
        }
    }

    void addEntity(Entity* entity)
    {
        auto ptr = this->entities.insert(std::make_pair(entity->name, entity));
        if (!ptr.second)
        {
            throw std::runtime_error("ERROR::SCENE::Key already exists in entities");
        }
    }

    void addEntities(std::vector<Entity*> entities)
    {
        for (auto& entity : entities)
        {
            this->addEntity(entity);
        }
    }

    // Getters.
    Shader* getShader(const std::string& key)
    {
        return this->shaders[key];
    }

    Texture* getTexture(const std::string& key)
    {
        return this->textures[key];
    }

    CubemapTexture* getCubemapTexture(const std::string& key)
    {
        return this->cubemapTextures[key];
    }

    Geometry* getGeometry(const std::string& key)
    {
        return this->geometries[key];
    }

    Model* getModel(const std::string& key)
    {
        return this->models[key];
    }

    Entity* getEntity(const std::string& key)
    {
        return this->entities[key];
    }

    // TODO: Read json and parse for the scene.
    // Manipulate entity tree.
    // void addEntity(Entity* toAdd, Entity* parent = nullptr)
    // {
    //     if (!parent)
    //     {
    //         parent = this->root;
    //     }
    //     parent->addChild(toAdd);
    // }

    // Traverse entities for drawing.
    
private:
    // Entity* root;
    std::map<std::string, Shader*> shaders;
    std::map<std::string, Texture*> textures;
    std::map<std::string, CubemapTexture*> cubemapTextures;
    std::map<std::string, Geometry*> geometries;
    std::map<std::string, Model*> models;
};
}
#endif
