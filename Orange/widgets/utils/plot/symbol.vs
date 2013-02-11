// Each example is drawn using a symbol constructed out
// of triangles. Each vertex is specified by its offset
// from the center point, example's position, color, normal
// and index (stored in .w components of position and offset).
attribute vec4 position;
attribute vec4 offset;
attribute vec3 color;
attribute vec3 normal;

varying vec4 var_color;

uniform bool use_2d_symbols;
uniform bool encode_color;
uniform bool hide_outside;
uniform bool fade_outside;
uniform vec4 force_color;
uniform vec2 alpha_value; // vec2 instead of float, fixing a bug on windows
                           // (setUniformValue with float crashes)
uniform vec2 symbol_scale;

uniform vec3 scale;
uniform vec3 translation;

uniform mat4 modelview;
uniform mat4 projection;

void main(void) {
    vec3 offset_rotated = offset.xyz;
    offset_rotated.x *= symbol_scale.x;
    offset_rotated.y *= symbol_scale.x;
    offset_rotated.z *= symbol_scale.x;

    if (use_2d_symbols) {
        // Calculate inverse of rotations (in this case, inverse
        // is actually just transpose), so that polygons face
        // camera all the time.
        mat3 invs;

        invs[0][0] = modelview[0][0];
        invs[0][1] = modelview[1][0];
        invs[0][2] = modelview[2][0];

        invs[1][0] = modelview[0][1];
        invs[1][1] = modelview[1][1];
        invs[1][2] = modelview[2][1];

        invs[2][0] = modelview[0][2];
        invs[2][1] = modelview[1][2];
        invs[2][2] = modelview[2][2];

        offset_rotated = invs * offset_rotated;
    }

    vec3 pos = position.xyz;
    pos += translation;
    pos *= scale;
    vec4 off_pos = vec4(pos+offset_rotated, 1.);

    gl_Position = projection * modelview * off_pos;

    if (force_color.a > 0.)
    {
        var_color = force_color;
    }
    else if (encode_color)
    {
        var_color = vec4(position.w, offset.w, 0, 0);
    }
    else
    {
        pos = abs(pos);
        float manhattan_distance = max(max(pos.x, pos.y), pos.z)+0.5;
        float a = alpha_value.x;

        if (fade_outside)
            a = min(pow(min(1., 1. / manhattan_distance), 5.), a);

        if (use_2d_symbols)
        {
            // No lighting for 2d symbols.
            var_color = vec4(color, a);
        }
        else
        {
            // Calculate the amount of lighting this triangle receives (diffuse component only).
            // The calculations are physically wrong, but look better.
            vec3 light_direction = normalize(vec3(1., 1., 0.5));
            float diffuse = max(0., dot(normalize((modelview * vec4(normal, 0.)).xyz), light_direction));
            var_color = vec4(color+diffuse*0.7, a);
        }
        if (manhattan_distance > 1. && hide_outside)
            var_color.a = 0.;
    }
}
