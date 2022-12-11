#ifndef BEHAVIOR_H
#define BEHAVIOR_H

#include "base/attribute.h"

// TODO
// Define the abstract class of behaviors and a single global list containing all activated behaviors.
namespace engine
{
class Behavior : Attribute
{
public:
    virtual ~Behavior();
    virtual void initiate() = 0;
    virtual void update() = 0;
};
}

#endif
