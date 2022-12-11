#ifndef ENTITY_H
#define ENTITY_H

#include <glad/glad.h> 
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <stdexcept>
#include <string>
#include <vector>

#include "base/attribute.h"
// #include "base/behavior.h"
#include "base/transform.h"
#include "base/system_global.h"
#include "utils/class_utils.h"


namespace engine
{
// Entity is the basic type of physical objects in the game. Entities in a
// scene is stored in the form of tree. Each entity possesses a name and a
// transform and a set of arbitrary attributions that do not have transforms.
class Entity
{
public:
    std::string name;
    Transform tf;
    std::vector<Attribute*> attribs = {};
    Entity* parent = nullptr;
    // std::vector<Entity*> children = {};


    // Constructors and a distructor.
    // There exists only one root entity for each scene. Every other entities
    // in a scene must have a non-null root entity.
    Entity() : name(std::string("root")), tf(Transform()) {}

    Entity(const std::string& name, Entity* parent)
        : name(name), tf(), parent(parent) {}

    Entity(const std::string& name, Entity* parent, const Transform& tf)
        : name(name), tf(tf), parent(parent) {}

    Entity(
        const std::string& name, Entity* parent, const Transform& tf,
        Attribute* attrib
    ) : name(name), tf(tf), parent(parent)
    {
        this->addAttribute(attrib);
    }

    Entity(
        const std::string& name, Entity* parent, const Transform& tf,
        std::vector<Attribute*> attribs
    ) : name(name), tf(tf), parent(parent)
    {
        this->addAttributes(attribs);
    }

    // Entity(
    //     const std::string& name, Entity* parent, const Transform& tf,
    //     Entity* child
    // ) : name(name), tf(tf), parent(parent)
    // {
    //     this->addChild(child);
    // }
    //
    // Entity(
    //     const std::string& name, Entity* parent, const Transform& tf,
    //     std::vector<Entity*> children
    // ) : name(name), tf(tf), parent(parent)
    // {
    //     this->addChildren(children);
    // }
    //
    // Entity(
    //     const std::string& name, Entity* parent, const Transform& tf,
    //     std::vector<Attribute*> attribs, std::vector<Entity*> children
    // ) : name(name), tf(tf), parent(parent)
    // {
    //     this->addAttributes(attribs);
    //     this->addChildren(children);
    // }

    // No copy constructor nor copy assignment are allowed.
    Entity(const Entity& other) = delete;
    Entity& operator=(const Entity& other) = delete;

    ~Entity()
    {
        for (auto& attrib : this->attribs)
        {
            if (!attrib)
            {
                delete attrib;
            }
        }
        // for (auto& child : this->children)
        // {
        //     if (!child)
        //     {
        //         // Dangling children are append to the parent.
        //         // parent->children.insert(child);
        //         // Dangling children are deleted. This will invoke cascading
        //         // deletion of entities.
        //         delete child;
        //     }
        // }
        // Only parent can delete its child, so the parent node will already
        // know what is going on.
    }

    // Getters and setters.
    template<typename T>
    T* getAttribute()
    {
        for (auto& attrib : this->attribs)
        {
            if (dynamic_cast<T*>(attrib))
            {
                return (T*)attrib;
            }
        }
        return nullptr;
    }

    // Object hierarchy manipulation.
    // void addChild(Entity* child)
    // {
    //     if (child->parent)
    //     {
    //         child->parent->popChild(child);
    //     }
    //     child->parent = this;
    //     this->children.push_back(child);
    // }
    //
    // void addChildren(std::vector<Entity*> children)
    // {
    //     for (auto& child : children)
    //     {
    //         this->addChild(child);
    //     }
    // }
    //
    // int deleteChild(Entity* child)
    // {
    //     for (auto it = this->children.begin(); it != this->children.end(); ++it)
    //     {
    //         if ((*it) == child)
    //         {
    //             it = this->children.erase(it)
    //             delete child;
    //             return 0;
    //         }
    //     }
    //     return -1;
    // }
    //
    // int deleteChildren(std::vector<Entity*> children)
    // {
    //     int flag = 0;
    //     for (auto& child: children)
    //     {
    //         flag |= this->deleteChild(child);
    //     }
    //     return flag;
    // }
    //
    // Entity* popChild(Entity* child)
    // {
    //     for (auto it = this->children.begin(); it != this->children.end(); ++it)
    //     {
    //         if ((*it) == child)
    //         {
    //             it = this->children.erase(it)
    //             return child;
    //         }
    //     }
    //     return nullptr;
    // }
    //
    // void deleteAllChildren() { this->deleteChildren(this->children); }

private:
    // All attributes are added at the initialization step.
    void addAttribute(Attribute* attrib)
    {
        if (attrib->entity)
        {
            throw std::logic_error("ERROR::ENTITY::Attributes are preallocated!");
        }
        // if (dynamic_cast<Behavior*>(attrib))
        // {
        //     behaviorQueue.push_back(attrib);
        // }
        attrib->entity = this;
        this->attribs.push_back(attrib);
    }

    void addAttributes(std::vector<Attribute*> attribs)
    {
        for (auto& attrib : attribs)
        {
            this->addAttribute(attrib);
        }
    }
};
}
#endif
