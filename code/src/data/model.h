#ifndef MODEL_H
#define MODEL_H

#include <glad/glad.h> 
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

#include "base/asset.h"
#include "data/mesh.h"
#include "data/texture.h"


namespace engine
{
class Model : public Asset
{
public:
    Mesh* mesh = nullptr;

    // No texture this time
    Texture* diffuse = nullptr;
    Texture* normal = nullptr;
    Texture* specular = nullptr;
    bool ignoreShadow = false;


    Model(const std::string& name, const std::string& path) : Asset(name)
    {
        loadModel(path);
    }

    Model(
        const std::string& name, const std::string& path,
        Texture* diffuse, Texture* normal, Texture* specular
    ) : Asset(name), diffuse(diffuse), normal(normal), specular(specular)
    {
        loadModel(path);
    }

    Model(
        const std::string& name, const std::string& path,
        Texture* diffuse, Texture* normal, Texture* specular, bool ignoreShadow
    ) : Asset(name), diffuse(diffuse), normal(normal), specular(specular),
        ignoreShadow(ignoreShadow)
    {
        loadModel(path);
    }

    ~Model()
    {
        delete this->mesh;
    }

    // No need this time
    void bind()
    {
        glBindVertexArray(this->mesh->VAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, this->diffuse->ID);
        if (this->specular) {
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, this->specular->ID);
        }
        if (this->normal) {
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, this->normal->ID);
        }
    }
    
private:
    // Loads a model with supported ASSIMP extensions from file and stores the
    // resulting meshes in the meshes vector.
    void loadModel(std::string const &path)
    {
        // Read file via ASSIMP.
        Assimp::Importer importer;

        // TODO : to get additional 3 points, DO NOT use aiProcess_CalcTangentSpace!
        // const aiScene* scene = importer.ReadFile(
        //     path, aiProcess_Triangulate | aiProcess_CalcTangentSpace
        // );
        const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate);
        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            std::cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << std::endl;
            return;
        }

        // process ASSIMP's root node recursively
        processNode(scene->mRootNode->mChildren[0], scene);
    }

    // Original code : processes a node in a recursive fashion. Processes each
    //                 individual mesh located at the node and repeats this
    //                 process on its children nodes (if any).
    // Modified version : only process FIRST item of obj file. Do not consider
    //                    texture or material.
    void processNode(aiNode *node, const aiScene *scene)
    {
        std::cout << node->mNumMeshes << std::endl;
        aiMesh* mesh = scene->mMeshes[node->mMeshes[0]];
        this->mesh = processMesh(mesh, scene);
    }

    Mesh* processMesh(aiMesh* mesh, const aiScene* scene)
    {
        // Data to fill.
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        std::vector<Texture> textures;

        // Walk through each of the mesh's vertices.
        for (int i = 0; i < mesh->mNumVertices; ++i)
        {
            // An intermediate placeholder between ASSIMP and our vertex.
            Vertex vertex;
            vertex.position = glm::vec3(
                mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z
            );
            if (mesh->mNormals)
            {
                vertex.normal = glm::vec3(
                    mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z
                );
            }
            if (mesh->mTextureCoords[0])
            {
                // A vertex can contain up to 8 different texture coordinates.
                // We thus make the assumption that we won't use models where
                // a vertex can have multiple texture coordinates so we always
                // take the first set (0).
                vertex.texCoords = glm::vec2(
                    mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y
                );
            }
            else
                vertex.texCoords = glm::vec2(0.0f, 0.0f);

            // Tangents and bitangents.
            // vertex.tangent = glm::vec3(
            //     mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z
            // );
            // vertex.bitangent = glm::vec3(
            //     mesh->mBitangents[i].x, mesh->mBitangents[i].y, mesh->mBitangents[i].z
            // );
            vertex.tangent = glm::vec3(0.0f);
            vertices.push_back(vertex);
        }
        // Walk through each of the mesh's faces (a face is a mesh its
        // triangle) and retrieve the corresponding vertex indices.
        for (int i = 0; i < mesh->mNumFaces; ++i)
        {
            aiFace face = mesh->mFaces[i];
            // Retrieve all indices of the face and store them in the index
            // vector.
            for (int j = 0; j < face.mNumIndices; ++j)
            {
                indices.push_back(face.mIndices[j]);
            }

            // Tangent vectors are evaluated manually.
            // Assume the number of vertices is a multiple of 3
            // (triangluarization).
            unsigned int i0 = face.mIndices[0];
            unsigned int i1 = face.mIndices[1];
            unsigned int i2 = face.mIndices[2];

            glm::vec2 uv0 = vertices[i0].texCoords;
            glm::vec2 uv1 = vertices[i1].texCoords;
            glm::vec2 uv2 = vertices[i2].texCoords;
            glm::vec3 pos0 = vertices[i0].position;
            glm::vec3 pos1 = vertices[i1].position;
            glm::vec3 pos2 = vertices[i2].position;

            glm::vec3 edge0 = pos1 - pos0;
            glm::vec3 edge1 = pos2 - pos0;
            glm::vec2 deltaUV0 = uv1 - uv0;
            glm::vec2 deltaUV1 = uv2 - uv0;
            float eta = 1.0f / (deltaUV0.x * deltaUV1.y - deltaUV0.y * deltaUV1.x);

            glm::vec3 tangent = eta * glm::vec3(
                deltaUV1.y * edge0.x - deltaUV0.y * edge1.x,
                deltaUV1.y * edge0.y - deltaUV0.y * edge1.y,
                deltaUV1.y * edge0.z - deltaUV0.y * edge1.z
            );
            vertices[i0].tangent += tangent;
            vertices[i1].tangent += tangent;
            vertices[i2].tangent += tangent;
        }

        // Assign tangents.
        for (int i = 0; i < mesh->mNumVertices; ++i)
        {
            glm::vec3 normal = vertices[i].normal;
            glm::vec3 tangent = vertices[i].tangent;
            vertices[i].tangent = glm::normalize(
                tangent - glm::dot(tangent, normal) * normal
            );
        }
        
        // Return a mesh object created from the extracted mesh data.
        return new Mesh(this->name, vertices, indices);
    }
};
}
#endif
