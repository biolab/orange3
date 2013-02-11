#version 150

in float index;

void main()
{
    gl_Position = vec4(index, 0., 0., 0.);
}
