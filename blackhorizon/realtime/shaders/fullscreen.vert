#version 330 core

// Fullscreen triangle: three vertices generated from gl_VertexID cover the
// screen without any vertex buffer. v_ndc carries normalized device
// coordinates in [-1, 1] to the fragment shader.

out vec2 v_ndc;

void main() {
    vec2 corners[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0)
    );
    v_ndc = corners[gl_VertexID];
    gl_Position = vec4(v_ndc, 0.0, 1.0);
}
