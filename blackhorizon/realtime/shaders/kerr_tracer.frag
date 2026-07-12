#version 330 core

// Kerr geodesic ray tracer in Cartesian Kerr-Schild coordinates.
//
// This shader is a direct transcription of the validated Stage 1 physics
// in blackhorizon/kerr.py and blackhorizon/geodesics.py (see docs/DESIGN.md
// sections 2.2 and 2.3). One backward null geodesic is integrated per
// pixel with fixed-order RK4 and a distance-based step-size heuristic
// whose accuracy is validated in tests/test_realtime_reference.py against
// the adaptive Stage 1 tracer.
//
// State per ray: spatial position x and covariant spatial momentum p.
// The time momentum p_t is exactly conserved (stationary spacetime) and
// normalized to +1 for the past-directed camera rays, so it never needs
// to be integrated or stored.
//
// Compile-time defines injected by shader_source.py:
//   MAX_STEPS        hard upper bound of the integration loop

in vec2 v_ndc;
out vec4 frag_color;

uniform vec3 u_cam_position;
uniform vec3 u_cam_forward;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
uniform float u_tan_half_fov;
uniform float u_aspect;

uniform float u_spin;
uniform float u_horizon_radius;
uniform float u_escape_radius;
uniform int u_max_steps;
uniform float u_step_scale;
uniform float u_min_step;
uniform float u_max_step;
uniform float u_capture_margin;
uniform float u_momentum_bailout;
uniform int u_background_mode;

uniform int u_disk_enabled;
uniform float u_disk_inner_radius;
uniform float u_disk_outer_radius;
uniform float u_disk_peak_temperature;
uniform float u_disk_detail;
uniform float u_exposure;
uniform float u_observer_lapse;
uniform float u_bb_log_min;
uniform float u_bb_log_max;
uniform sampler2D u_temperature_lut;
uniform sampler2D u_blackbody_lut;

const float P_T = 1.0;

// Kerr-Schild geometry at a spatial point: radius r, scalar H, spatial
// null covector l, gradient of H, and the three gradient vectors
// dl[i] = (d l_x / d x_i, d l_y / d x_i, d l_z / d x_i).
struct Geometry {
    float r;
    float h;
    vec3 l;
    vec3 grad_h;
    mat3 grad_l;
};

float kerr_schild_radius(vec3 pos) {
    float a = u_spin;
    float a2 = a * a;
    float rho2 = dot(pos, pos);
    float q = rho2 - a2;
    float s = sqrt(q * q + 4.0 * a2 * pos.z * pos.z);
    return sqrt(max(0.5 * (q + s), 1e-12));
}

Geometry evaluate_geometry(vec3 pos) {
    Geometry g;
    float a = u_spin;
    float a2 = a * a;
    float rho2 = dot(pos, pos);
    float q = rho2 - a2;
    float s = max(sqrt(q * q + 4.0 * a2 * pos.z * pos.z), 1e-12);
    float r2 = max(0.5 * (q + s), 1e-12);
    float r = sqrt(r2);
    g.r = r;

    float r2_plus_a2 = r2 + a2;
    float qd = max(r2 * r2 + a2 * pos.z * pos.z, 1e-12);
    g.h = r * r2 / qd;

    float inv_r2a2 = 1.0 / r2_plus_a2;
    float inv_r = 1.0 / r;
    g.l = vec3(
        (r * pos.x + a * pos.y) * inv_r2a2,
        (r * pos.y - a * pos.x) * inv_r2a2,
        pos.z * inv_r
    );

    // Gradient of r from implicit differentiation of the radius quartic.
    float inv_s = 1.0 / s;
    vec3 grad_r = vec3(
        pos.x * r * inv_s,
        pos.y * r * inv_s,
        pos.z * r2_plus_a2 * inv_r * inv_s
    );

    // Gradient of H = r^3 / (r^4 + a^2 z^2) with M = 1.
    float inv_qd2 = 1.0 / (qd * qd);
    float dh_dr = r2 * (3.0 * a2 * pos.z * pos.z - r2 * r2) * inv_qd2;
    float dh_dz = -2.0 * a2 * pos.z * r * r2 * inv_qd2;
    g.grad_h = dh_dr * grad_r + vec3(0.0, 0.0, dh_dz);

    // Gradients of the spatial null covector (quotient rule); row i of
    // grad_l holds (d l_x / d x_i, d l_y / d x_i, d l_z / d x_i). GLSL
    // matrices are column-major, so rows are assembled via transpose.
    float two_r = 2.0 * r;
    vec3 row_x = vec3(
        (grad_r.x * pos.x + r) * inv_r2a2 - g.l.x * two_r * grad_r.x * inv_r2a2,
        (grad_r.x * pos.y - a) * inv_r2a2 - g.l.y * two_r * grad_r.x * inv_r2a2,
        -g.l.z * grad_r.x * inv_r
    );
    vec3 row_y = vec3(
        (grad_r.y * pos.x + a) * inv_r2a2 - g.l.x * two_r * grad_r.y * inv_r2a2,
        (grad_r.y * pos.y + r) * inv_r2a2 - g.l.y * two_r * grad_r.y * inv_r2a2,
        -g.l.z * grad_r.y * inv_r
    );
    vec3 row_z = vec3(
        grad_r.z * pos.x * inv_r2a2 - g.l.x * two_r * grad_r.z * inv_r2a2,
        grad_r.z * pos.y * inv_r2a2 - g.l.y * two_r * grad_r.z * inv_r2a2,
        (1.0 - g.l.z * grad_r.z) * inv_r
    );
    g.grad_l = transpose(mat3(row_x, row_y, row_z));
    return g;
}

// Hamilton's equations for (x, p) with p_t = P_T held constant.
void geodesic_rhs(vec3 pos, vec3 p, out vec3 dx, out vec3 dp) {
    Geometry g = evaluate_geometry(pos);
    float lp = -P_T + dot(g.l, p);
    float two_h_lp = 2.0 * g.h * lp;
    dx = p - two_h_lp * g.l;
    // grad_l is stored so that column j of the matrix times p gives the
    // contraction sum_j p_j d l_j / d x_i as (grad_l * p) with rows i.
    dp = g.grad_h * (lp * lp) + two_h_lp * (g.grad_l * p);
}

void rk4_step(inout vec3 pos, inout vec3 p, float h_step) {
    vec3 dx1, dp1, dx2, dp2, dx3, dp3, dx4, dp4;
    geodesic_rhs(pos, p, dx1, dp1);
    geodesic_rhs(pos + 0.5 * h_step * dx1, p + 0.5 * h_step * dp1, dx2, dp2);
    geodesic_rhs(pos + 0.5 * h_step * dx2, p + 0.5 * h_step * dp2, dx3, dp3);
    geodesic_rhs(pos + h_step * dx3, p + h_step * dp3, dx4, dp4);
    pos += (h_step / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4);
    p += (h_step / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4);
}

// Past-directed null momentum for a camera ray with contravariant spatial
// direction dir: solve g_mu_nu k^mu k^nu = 0 for k^t (past root), lower
// the index, and rescale so p_t = +1.
vec3 initial_momentum(vec3 pos, vec3 dir) {
    Geometry g = evaluate_geometry(pos);
    float l_dot_s = dot(g.l, dir);
    float g_tt = 2.0 * g.h - 1.0;
    float b = 4.0 * g.h * l_dot_s;
    float c = dot(dir, dir) + 2.0 * g.h * l_dot_s * l_dot_s;
    float disc = max(b * b - 4.0 * g_tt * c, 0.0);
    float sqrt_disc = sqrt(disc);
    float k_t = min((-b + sqrt_disc) / (2.0 * g_tt),
                    (-b - sqrt_disc) / (2.0 * g_tt));
    float l_dot_k = k_t + l_dot_s;
    float p_t = -k_t + 2.0 * g.h * l_dot_k;
    vec3 p = dir + 2.0 * g.h * l_dot_k * g.l;
    return p / abs(p_t);
}

// Novikov-Thorne disk emission at an equatorial crossing point.
// The redshift g combines gravitational shift, Doppler beaming and
// aberration via g = nu_obs / nu_em with nu_em = -(p_phys . u_emitter);
// for the past-directed traced momentum (p_t = +1) this reduces to
// nu_em = u^t (1 + dot(p_spatial, v_circular)). A blackbody at T shifts
// to a blackbody at g T with bolometric intensity scaling as g^4, which
// the T^4 brightness factor below absorbs automatically.
vec3 disk_emission(vec3 hit_pos, vec3 hit_p) {
    Geometry g = evaluate_geometry(hit_pos);
    float omega = 1.0 / (pow(g.r, 1.5) + u_spin);
    vec3 v = omega * vec3(-hit_pos.y, hit_pos.x, 0.0);
    float l_dot_v = 1.0 + dot(g.l, v);
    float norm2 = -1.0 + dot(v, v) + 2.0 * g.h * l_dot_v * l_dot_v;
    float u_t = inversesqrt(max(-norm2, 1e-9));
    float nu_emitted = u_t * (1.0 + dot(hit_p, v));
    float shift = u_observer_lapse / max(nu_emitted, 1e-6);

    float lut_u = clamp(
        (g.r - u_disk_inner_radius)
            / (u_disk_outer_radius - u_disk_inner_radius),
        0.0, 1.0
    );
    float t_norm = texture(u_temperature_lut, vec2(lut_u, 0.5)).r;
    float t_observed = shift * t_norm * u_disk_peak_temperature;

    float bb_u = clamp(
        (log(max(t_observed, 1.0)) - u_bb_log_min)
            / (u_bb_log_max - u_bb_log_min),
        0.0, 1.0
    );
    vec3 tint = texture(u_blackbody_lut, vec2(bb_u, 0.5)).rgb;

    float phi = atan(hit_pos.y, hit_pos.x);
    float detail = 1.0 + u_disk_detail * (
        0.18 * sin(9.0 * phi + 2.2 * g.r)
        + 0.12 * sin(23.0 * phi - 5.0 * g.r)
        + 0.15 * sin(3.5 * (g.r - u_disk_inner_radius))
    );
    float brightness = u_exposure
        * pow(t_observed / 6500.0, 4.0)
        * max(detail, 0.2);
    vec3 color = tint * brightness;
    return color / (1.0 + color);
}

float hash13(vec3 v) {
    v = fract(v * 0.1031);
    v += dot(v, v.zyx + 31.32);
    return fract((v.x + v.y) * v.z);
}

vec3 background_color(vec3 dir) {
    float theta = acos(clamp(dir.z, -1.0, 1.0));
    float phi = atan(dir.y, dir.x);
    if (u_background_mode == 0) {
        // Two-tone checkerboard: matches the Stage 1 imaging module.
        float cells = 12.0;
        float it = floor(theta / 3.14159265 * cells);
        float ip = floor((phi + 3.14159265) / 6.2831853 * 2.0 * cells);
        bool light = mod(it + ip, 2.0) >= 1.0;
        return light ? vec3(0.408, 0.502, 0.698) : vec3(0.094, 0.118, 0.204);
    }
    // Procedural starfield with a faint coordinate grid.
    vec3 cell = floor(dir * 220.0);
    float star = pow(hash13(cell), 60.0);
    vec3 color = vec3(0.012, 0.014, 0.028) + vec3(star);
    float grid_t = smoothstep(0.006, 0.0, abs(fract(theta / 0.19635) - 0.5) * 0.19635);
    float grid_p = smoothstep(0.006, 0.0, abs(fract(phi / 0.19635) - 0.5) * 0.19635);
    color += vec3(0.02, 0.03, 0.05) * max(grid_t, grid_p);
    return color;
}

void main() {
    // Camera ray through this pixel.
    vec2 ndc = vec2(v_ndc.x, v_ndc.y);
    vec3 dir = normalize(
        u_cam_forward
        + ndc.x * u_tan_half_fov * u_cam_right
        + ndc.y * u_tan_half_fov / u_aspect * u_cam_up
    );

    vec3 pos = u_cam_position;
    vec3 p = initial_momentum(pos, dir);
    float capture_radius = u_horizon_radius * (1.0 + u_capture_margin);

    int status = 0;
    vec3 dx_final = dir;
    vec3 disk_color = vec3(0.0);
    for (int i = 0; i < MAX_STEPS; i++) {
        if (i >= u_max_steps) {
            break;
        }
        float r = kerr_schild_radius(pos);
        // A diverging blueshift identifies the horizon for past-directed
        // rays; bail out as captured before fixed steps go unstable.
        if (r <= capture_radius
            || dot(p, p) >= u_momentum_bailout * u_momentum_bailout) {
            status = 1;
            break;
        }
        if (r >= u_escape_radius) {
            status = 2;
            break;
        }
        // Distance-based step heuristic, validated against the adaptive
        // Stage 1 tracer in tests/test_realtime_reference.py.
        float h_step = clamp(
            u_step_scale * (r - u_horizon_radius), u_min_step, u_max_step
        );
        // Bound the displacement of any RK stage for blueshifting rays.
        h_step = min(h_step, 1.0 / max(1.0, length(p)));
        vec3 prev_pos = pos;
        vec3 prev_p = p;
        rk4_step(pos, p, h_step);

        // Opaque thin disk: detect an equatorial plane crossing between
        // consecutive accepted positions, interpolate the crossing point
        // linearly, and terminate the ray if it lies within the disk.
        if (u_disk_enabled == 1 && prev_pos.z * pos.z < 0.0) {
            float t_cross = prev_pos.z / (prev_pos.z - pos.z);
            vec3 hit_pos = mix(prev_pos, pos, t_cross);
            float r_hit = kerr_schild_radius(hit_pos);
            if (r_hit >= u_disk_inner_radius
                && r_hit <= u_disk_outer_radius) {
                vec3 hit_p = mix(prev_p, p, t_cross);
                disk_color = disk_emission(hit_pos, hit_p);
                status = 3;
                break;
            }
        }
    }

    if (status == 3) {
        frag_color = vec4(disk_color, 1.0);
    } else if (status == 2) {
        Geometry g = evaluate_geometry(pos);
        float lp = -P_T + dot(g.l, p);
        dx_final = normalize(p - 2.0 * g.h * lp * g.l);
        frag_color = vec4(background_color(dx_final), 1.0);
    } else {
        // Captured rays and step-budget rays: the latter hug the photon
        // shell, where black is the visually correct limit.
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
