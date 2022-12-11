#ifndef CLASS_UTILS
#define CLASS_UTILS

#include <set>
#include <tuple>
#include <typeindex>


// https://stackoverflow.com/questions/500493/c-equivalent-of-javas-instanceof
#define _EMPTY_BASE_TYPE_DECL() using BaseTypes = std::tuple<>;
#define _BASE_TYPE_DECL(Class, BaseClass) \
    using BaseTypes = decltype(std::tuple_cat(std::tuple<BaseClass>(), Class::BaseTypes()));
#define _INSTANCE_OF_DECL_BODY(Class) \
    static const std::set<std::type_index> baseTypeContainer; \
    virtual bool instanceOfHelper(const std::type_index &_tidx) { \
        if (std::type_index(typeid(ThisType)) == _tidx) return true; \
        if (std::tuple_size<BaseTypes>::value == 0) return false; \
        return baseTypeContainer.find(_tidx) != baseTypeContainer.end(); \
    } \
    template <typename... T> \
    static std::set<std::type_index> getTypeIndexes(std::tuple<T...>) { \
        return std::set<std::type_index>{std::type_index(typeid(T))...}; \
    }

#define INSTANCE_OF_SUB_DECL(Class, BaseClass) \
    protected: \
    using ThisType = Class; \
    _BASE_TYPE_DECL(Class, BaseClass) \
    _INSTANCE_OF_DECL_BODY(Class)

#define INSTANCE_OF_BASE_DECL(Class) \
    protected: \
        using ThisType = Class; \
        _EMPTY_BASE_TYPE_DECL() \
        _INSTANCE_OF_DECL_BODY(Class) \
    public: \
        template <typename Of> \
        typename std::enable_if<std::is_base_of<Class, Of>::value, bool>::type instanceOf() { \
            return instanceOfHelper(std::type_index(typeid(Of))); \
        }

#define INSTANCE_OF_IMPL(Class) \
    const std::set<std::type_index> Class::baseTypeContainer = Class::getTypeIndexes(Class::BaseTypes());

#endif
