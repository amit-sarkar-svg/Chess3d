#version 330 core

layout(location = 0) in vec3 aPos;      // vertex position
layout(location = 1) in vec3 aNormal;   // vertex normal
layout(location = 2) in vec2 aUV;       // vertex UV

out vec3 FragPos;       // world-space position
out vec3 Normal;        // world-space normal
out vec2 TexCoord;      // UV passed to fragment shader

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    // Transform vertex into world space
    vec4 worldPos = model * vec4(aPos, 1.0);
    FragPos = worldPos.xyz;

    // Transform normal (ignore scale/skew)
    Normal = mat3(transpose(inverse(model))) * aNormal;

    TexCoord = aUV;

    // Final clip-space position
    gl_Position = projection * view * worldPos;
}
