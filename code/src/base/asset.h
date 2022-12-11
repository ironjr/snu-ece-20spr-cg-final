#ifndef ASSET_H
#define ASSET_H


namespace engine
{
class Asset
{
public:
    std::string name;
    unsigned int assignments = 0;

    Asset(const std::string& name) : name(name) {}
};
}
#endif
