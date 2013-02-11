#version 150

layout(points) in;
layout(triangle_strip, max_vertices=144) out;

uniform int x_index;
uniform int y_index;
uniform int z_index;
uniform int color_index;
uniform int symbol_index;
uniform int size_index;

uniform bool use_2d_symbols;

uniform float jitter_size;
uniform bool jitter_continuous;
uniform bool x_discrete;
uniform bool y_discrete;
uniform bool z_discrete;

uniform samplerBuffer symbol_buffer;
uniform samplerBuffer data_buffer;

uniform int num_symbols_used;
uniform int[20] symbols_indices;
uniform int[20] symbols_sizes;
uniform int example_size;

// Colors are specified in case of a discrete attribute.
uniform int num_colors;
uniform vec3[50] colors;

out vec3 out_position;
out vec3 out_offset;
out vec3 out_color;
out vec3 out_normal;
out float out_index;

// http://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
// Should return pseudo-random value [-0.5, 0.5]
float rand(vec3 co){
    return fract(sin(dot(co.xyz, vec3(12.9898, 78.233, 42.42))) * 43758.5453) - 0.5;
}

void main()
{
    vec4 position = gl_in[0].gl_Position;
    const float scale = 0.001;

    out_index = position.x;
    int index = int(out_index * example_size);

    out_position = vec3(texelFetch(data_buffer, index+x_index).x,
                        texelFetch(data_buffer, index+y_index).x,
                        texelFetch(data_buffer, index+z_index).x);

	if (x_discrete || jitter_continuous)
		out_position.x += rand(out_position * out_index) * jitter_size / 100.;
	if (y_discrete || jitter_continuous)
		out_position.y += rand(out_position * out_index) * jitter_size / 100.;
	if (z_discrete || jitter_continuous)
		out_position.z += rand(out_position * out_index) * jitter_size / 100.;

    int symbol = 0;
    if (num_symbols_used > 1 && symbol_index > -1)
        symbol = clamp(int(texelFetch(data_buffer, index+symbol_index).x * num_symbols_used), 0, 9);
    if (!use_2d_symbols)
        symbol += 10;

    float size = texelFetch(data_buffer, index+size_index).x;
    if (size_index < 0 || size < 0.)
        size = 1.;

    float color = texelFetch(data_buffer, index+color_index).x;
    if (num_colors > 0)
        out_color = colors[int(color*num_colors)];
    else if (color_index > -1)
        out_color = vec3(0., 0., color);
    else
        out_color = vec3(0., 0., 0.8);

    for (int i = 0; i < symbols_sizes[symbol]; ++i)
    {
        out_offset = texelFetch(symbol_buffer, symbols_indices[symbol]+i*6+0).xyz * size * scale;
        out_normal = texelFetch(symbol_buffer, symbols_indices[symbol]+i*6+3).xyz;
        EmitVertex();
        out_offset = texelFetch(symbol_buffer, symbols_indices[symbol]+i*6+1).xyz * size * scale;
        out_normal = texelFetch(symbol_buffer, symbols_indices[symbol]+i*6+4).xyz;
        EmitVertex();
        out_offset = texelFetch(symbol_buffer, symbols_indices[symbol]+i*6+2).xyz * size * scale;
        out_normal = texelFetch(symbol_buffer, symbols_indices[symbol]+i*6+5).xyz;
        EmitVertex();

        EndPrimitive();
    }
}
