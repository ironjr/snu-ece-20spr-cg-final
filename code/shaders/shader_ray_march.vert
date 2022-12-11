#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;


void main()
{
    TexCoord = aTexCoord;
    mat3 model = mat3(2.0);
    gl_Position = vec4(model * aPos, 1.0);
}
