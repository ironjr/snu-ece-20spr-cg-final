#ifndef MESH_H
#define MESH_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <string>
#include <vector>

#include "base/asset.h"


namespace engine
{
struct Vertex
{
    glm::vec3 position;
    glm::vec2 texCoords;
    glm::vec3 normal;
    glm::vec3 tangent;
};


class Mesh : public Asset
{
public:
    unsigned int VAO;

    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;


    Mesh(
        const std::string& name, std::vector<Vertex> vertices,
        std::vector<unsigned int> indices
    ) : Asset(name)
    {
        this->vertices = vertices;
        this->indices = indices;

        // Now that we have all the required data, set the vertex buffers and
        // its attribute pointers.
        // initialize(); TODO Uncomment it for later usage.
    }

private:
    unsigned int VBO;
    unsigned int EBO;


    // Initializes all the buffer objects/arrays.
    void initialize()
    {
        // Create buffers and arrays.
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        // A great thing about structs is that their memory layout is
        // sequential for all its items. The effect is that we can simply pass
        // a pointer to the struct and it translates perfectly to a
        // glm::vec3/2 array which again translates to 3/2 floats which
        // translates to a byte array.
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);  

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        // Set the vertex attribute pointers.
        glEnableVertexAttribArray(0);	
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));
        glEnableVertexAttribArray(2);	
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tangent));

        glBindVertexArray(0);
    }
};
}

#endif
