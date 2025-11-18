#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D texture0;

// Lighting uniforms
uniform vec3 lightDir = normalize(vec3(-0.5, -1.0, -0.4));
uniform vec3 lightColor = vec3(1.0, 1.0, 1.0);

// Highlight system
uniform bool enable_highlight = false;
uniform vec3 highlight_color = vec3(1.0, 0.8, 0.1);

// Material settings
uniform float ambient_strength = 0.35;
uniform float diffuse_strength = 0.65;

// Texture control
uniform bool use_texture = true;
uniform vec3 base_color = vec3(1.0, 1.0, 1.0);

// Debug: override output color to diagnose rendering
uniform bool debug_flat = false;
uniform vec3 debug_color = vec3(1.0, 1.0, 1.0);

void main()
{
    if (debug_flat) {
        FragColor = vec4(debug_color, 1.0);
        return;
    }
    // Base color: from texture or uniform fallback
    vec3 texColor = texture(texture0, TexCoord).rgb;
    vec3 baseColor = use_texture ? texColor : base_color;

    // Lighting
    vec3 N = normalize(Normal);
    vec3 L = normalize(-lightDir);

    float diff = max(dot(N, L), 0.0);

    vec3 ambient = ambient_strength * baseColor;
    vec3 diffuse = diffuse_strength * diff * baseColor;

    vec3 resultColor = ambient + diffuse;

    // Apply highlight overlay
    if (enable_highlight) {
        resultColor = mix(resultColor, highlight_color, 0.55);
    }

    FragColor = vec4(resultColor, 1.0);
}
